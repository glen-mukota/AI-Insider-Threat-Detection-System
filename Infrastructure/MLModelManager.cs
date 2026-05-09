using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using InsiderThreatDetection.Core.Models;

namespace InsiderThreatDetection.Infrastructure
{
    public class MLModelManager
    {
        private readonly MLContext _mlContext;
        private ITransformer? _model;
        private PredictionEngine<UserBehaviour, ThreatPrediction>? _predictionEngine;
        public BinaryClassificationMetrics? LastMetrics { get; private set; }

        private const float TestFraction = 0.2f;
        private const int RandomSeed = 42;
        private const float DecisionThreshold = 0.5f;

        private const string FeaturesColumn = "Features";
        private const string LabelColumn = "Label";

        // Statistics computed ONLY from benign training examples
        private float[]? _benignMeans;
        private float[]? _benignStdDevs;

        public float[]? BenignMeans => _benignMeans;

        private readonly string[] _featureNames = new[]
        {
            "employee_seniority_years", "is_contractor", "employee_classification",
            "total_printed_pages", "num_printed_pages_off_hours",
            "total_files_burned", "burned_from_other",
            "is_abroad", "trip_day_number", "hostility_country_level",
            "num_entries", "num_unique_campus",
            "late_exit_flag", "entry_during_weekend"
        };

        public MLModelManager()
        {
            _mlContext = new MLContext(seed: RandomSeed);
        }

        public void Train(string dataPath)
        {
            var data = _mlContext.Data.LoadFromTextFile<UserBehaviour>(
                path: dataPath, hasHeader: true, separatorChar: ',');

            var split = _mlContext.Data.TrainTestSplit(data, testFraction: TestFraction);
            float maliciousWeight = ComputeMaliciousWeight(split.TrainSet);

            (_benignMeans, _benignStdDevs) = ComputeBenignStats(split.TrainSet);

            var pipeline = BuildPipeline(maliciousWeight);
            _model = pipeline.Fit(split.TrainSet);

            var predictions = _model.Transform(split.TestSet);
            LastMetrics = _mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: LabelColumn);
            PrintMetrics(LastMetrics);

            _predictionEngine = _mlContext.Model.CreatePredictionEngine<UserBehaviour, ThreatPrediction>(_model);
        }

        public ThreatPrediction Predict(UserBehaviour input)
        {
            if (_predictionEngine == null)
                throw new InvalidOperationException("Model not trained or loaded.");
            var raw = _predictionEngine.Predict(input);
            return new ThreatPrediction
            {
                PredictedLabel = raw.Probability >= DecisionThreshold,
                Probability = raw.Probability,
                Score = raw.Score
            };
        }

        /// <summary>
        /// Returns the confusion matrix from the last evaluation.
        /// Format: [0][0]=TN, [0][1]=FP, [1][0]=FN, [1][1]=TP.
        /// </summary>
        public double[][]? GetConfusionMatrixCounts()
        {
            var counts = LastMetrics?.ConfusionMatrix?.Counts;
            if (counts == null) return null;
            // Convert IReadOnlyList<IReadOnlyList<double>> to double[][]
            return counts.Select(row => row.ToArray()).ToArray();
        }

        // ---------- EXPLAINABILITY (Contributions) ----------
        public List<(string Feature, float Contribution, float Value)> Explain(UserBehaviour input)
        {
            if (_model == null || _benignMeans == null)
                throw new InvalidOperationException("Model not trained.");

            float baselineProb = Predict(input).Probability;
            var result = new List<(string, float, float)>();

            for (int i = 0; i < _featureNames.Length; i++)
            {
                UserBehaviour perturbed = CloneBehaviour(input);
                SetFeatureValue(perturbed, _featureNames[i], _benignMeans[i]);
                float perturbedProb = Predict(perturbed).Probability;
                float contribution = baselineProb - perturbedProb;
                result.Add((_featureNames[i], contribution, GetFeatureValue(input, _featureNames[i])));
            }

            return result.OrderByDescending(x => Math.Abs(x.Item2)).ToList();
        }

        // ---------- MANAGER‑FRIENDLY EXPLANATION ----------
        public string GenerateHumanExplanation(UserBehaviour input, ThreatPrediction prediction)
        {
            if (_benignMeans == null || _benignStdDevs == null)
                return "Model statistics not available.";

            var sb = new StringBuilder();
            sb.AppendLine($"Threat Probability: {prediction.Probability:P0}");
            sb.AppendLine();

            // Identify features that deviate significantly from benign average (>2 std)
            var anomalies = new List<(string Feature, float Value, float Mean, string Direction)>();
            for (int i = 0; i < _featureNames.Length; i++)
            {
                float val = GetFeatureValue(input, _featureNames[i]);
                float mean = _benignMeans[i];
                float std = _benignStdDevs[i];
                if (std > 0 && Math.Abs(val - mean) > 2 * std)
                {
                    string dir = val > mean ? "higher" : "lower";
                    anomalies.Add((_featureNames[i], val, mean, dir));
                }
            }

            if (prediction.PredictedLabel)
            {
                sb.AppendLine("The activity is flagged as MALICIOUS.");
                if (anomalies.Any())
                {
                    sb.AppendLine("The following behaviours are unusual compared to typical employees:");
                    foreach (var a in anomalies.Take(3))
                    {
                        sb.AppendLine($"  - {HumanReadableName(a.Feature)}: {a.Value} (normal is around {a.Mean:F1})");
                    }
                }
                else
                {
                    sb.AppendLine("Although no single behaviour is extreme, the combination raised the overall risk.");
                }
            }
            else
            {
                sb.AppendLine("The activity is NORMAL.");
                if (anomalies.Any())
                {
                    sb.AppendLine("A few indicators were slightly outside the typical range, but they are not strong enough to raise an alert.");
                }
                else
                {
                    sb.AppendLine("All indicators are within expected ranges.");
                }
            }

            return sb.ToString();
        }

        private string HumanReadableName(string feature) => feature switch
        {
            "employee_seniority_years" => "Years of seniority",
            "is_contractor" => "Contractor status",
            "employee_classification" => "Job classification",
            "total_printed_pages" => "Printed pages",
            "num_printed_pages_off_hours" => "Off‑hours printing",
            "total_files_burned" => "Files burned to USB/external",
            "burned_from_other" => "Files burned from other devices",
            "is_abroad" => "Working abroad",
            "trip_day_number" => "Days on trip",
            "hostility_country_level" => "Hostility level of country",
            "num_entries" => "Building entries",
            "num_unique_campus" => "Unique campuses accessed",
            "late_exit_flag" => "Late exit",
            "entry_during_weekend" => "Weekend entry",
            _ => feature
        };

        // ---------- HELPERS ----------
        private UserBehaviour CloneBehaviour(UserBehaviour source) => new()
        {
            employee_seniority_years = source.employee_seniority_years,
            is_contractor = source.is_contractor,
            employee_classification = source.employee_classification,
            total_printed_pages = source.total_printed_pages,
            num_printed_pages_off_hours = source.num_printed_pages_off_hours,
            total_files_burned = source.total_files_burned,
            burned_from_other = source.burned_from_other,
            is_abroad = source.is_abroad,
            trip_day_number = source.trip_day_number,
            hostility_country_level = source.hostility_country_level,
            num_entries = source.num_entries,
            num_unique_campus = source.num_unique_campus,
            late_exit_flag = source.late_exit_flag,
            entry_during_weekend = source.entry_during_weekend,
            is_malicious = 0
        };

        private float GetFeatureValue(UserBehaviour input, string fn) => fn switch
        {
            "employee_seniority_years" => input.employee_seniority_years,
            "is_contractor" => input.is_contractor,
            "employee_classification" => input.employee_classification,
            "total_printed_pages" => input.total_printed_pages,
            "num_printed_pages_off_hours" => input.num_printed_pages_off_hours,
            "total_files_burned" => input.total_files_burned,
            "burned_from_other" => input.burned_from_other,
            "is_abroad" => input.is_abroad,
            "trip_day_number" => input.trip_day_number,
            "hostility_country_level" => input.hostility_country_level,
            "num_entries" => input.num_entries,
            "num_unique_campus" => input.num_unique_campus,
            "late_exit_flag" => input.late_exit_flag,
            "entry_during_weekend" => input.entry_during_weekend,
            _ => 0f
        };

        private void SetFeatureValue(UserBehaviour input, string fn, float v)
        {
            switch (fn)
            {
                case "employee_seniority_years": input.employee_seniority_years = v; break;
                case "is_contractor": input.is_contractor = v; break;
                case "employee_classification": input.employee_classification = v; break;
                case "total_printed_pages": input.total_printed_pages = v; break;
                case "num_printed_pages_off_hours": input.num_printed_pages_off_hours = v; break;
                case "total_files_burned": input.total_files_burned = v; break;
                case "burned_from_other": input.burned_from_other = v; break;
                case "is_abroad": input.is_abroad = v; break;
                case "trip_day_number": input.trip_day_number = v; break;
                case "hostility_country_level": input.hostility_country_level = v; break;
                case "num_entries": input.num_entries = v; break;
                case "num_unique_campus": input.num_unique_campus = v; break;
                case "late_exit_flag": input.late_exit_flag = v; break;
                case "entry_during_weekend": input.entry_during_weekend = v; break;
            }
        }

        private (float[] means, float[] stdDevs) ComputeBenignStats(IDataView trainData)
        {
            var benignRows = _mlContext.Data.CreateEnumerable<UserBehaviour>(trainData, reuseRowObject: false)
                .Where(r => r.is_malicious == 0)
                .ToList();

            int n = benignRows.Count;
            int f = 14;
            float[] sums = new float[f];
            float[] sqSums = new float[f];

            foreach (var r in benignRows)
            {
                float[] vals = new float[f];
                vals[0] = r.employee_seniority_years;
                vals[1] = r.is_contractor;
                vals[2] = r.employee_classification;
                vals[3] = r.total_printed_pages;
                vals[4] = r.num_printed_pages_off_hours;
                vals[5] = r.total_files_burned;
                vals[6] = r.burned_from_other;
                vals[7] = r.is_abroad;
                vals[8] = r.trip_day_number;
                vals[9] = r.hostility_country_level;
                vals[10] = r.num_entries;
                vals[11] = r.num_unique_campus;
                vals[12] = r.late_exit_flag;
                vals[13] = r.entry_during_weekend;
                for (int i = 0; i < f; i++)
                {
                    sums[i] += vals[i];
                    sqSums[i] += vals[i] * vals[i];
                }
            }

            float[] means = sums.Select(s => s / n).ToArray();
            float[] stds = new float[f];
            for (int i = 0; i < f; i++)
            {
                float m = means[i];
                stds[i] = (float)Math.Sqrt(sqSums[i] / n - m * m);
            }
            return (means, stds);
        }

        public void SaveModel(string modelPath)
        {
            if (_model == null) throw new InvalidOperationException("No trained model to save.");
            _mlContext.Model.Save(_model, null, modelPath);
        }

        public void LoadModel(string modelPath)
        {
            _model = _mlContext.Model.Load(modelPath, out _);
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<UserBehaviour, ThreatPrediction>(_model);
        }

        private IEstimator<ITransformer> BuildPipeline(float maliciousWeight)
        {
            return _mlContext.Transforms
                .Conversion.ConvertType(LabelColumn, nameof(UserBehaviour.is_malicious), DataKind.Boolean)
                .Append(_mlContext.Transforms.CustomMapping(
                    (UserBehaviour input, WeightOutput output) =>
                    {
                        output.Weight = input.is_malicious == 1 ? maliciousWeight : 1f;
                    }, contractName: null))
                .Append(_mlContext.Transforms.Concatenate(FeaturesColumn,
                    nameof(UserBehaviour.employee_seniority_years),
                    nameof(UserBehaviour.is_contractor),
                    nameof(UserBehaviour.employee_classification),
                    nameof(UserBehaviour.total_printed_pages),
                    nameof(UserBehaviour.num_printed_pages_off_hours),
                    nameof(UserBehaviour.total_files_burned),
                    nameof(UserBehaviour.burned_from_other),
                    nameof(UserBehaviour.is_abroad),
                    nameof(UserBehaviour.trip_day_number),
                    nameof(UserBehaviour.hostility_country_level),
                    nameof(UserBehaviour.num_entries),
                    nameof(UserBehaviour.num_unique_campus),
                    nameof(UserBehaviour.late_exit_flag),
                    nameof(UserBehaviour.entry_during_weekend)))
                .Append(_mlContext.Transforms.NormalizeMeanVariance(FeaturesColumn))
                .Append(_mlContext.BinaryClassification.Trainers.FastTree(
                    LabelColumn, FeaturesColumn, "Weight",
                    numberOfLeaves: 20, numberOfTrees: 100, minimumExampleCountPerLeaf: 10));
        }

        private float ComputeMaliciousWeight(IDataView trainData)
        {
            int malicious = 0, normal = 0;
            foreach (var row in _mlContext.Data.CreateEnumerable<UserBehaviour>(trainData, reuseRowObject: true))
            {
                if (row.is_malicious == 1) malicious++; else normal++;
            }
            Console.WriteLine($"Training set - Malicious: {malicious}, Normal: {normal}");
            return malicious == 0 ? 1f : (float)normal / malicious;
        }

        private void PrintMetrics(BinaryClassificationMetrics m)
        {
            Console.WriteLine("=== MODEL PERFORMANCE ===");
            Console.WriteLine($"Accuracy:  {m.Accuracy:P2}");
            Console.WriteLine($"Precision: {m.PositivePrecision:P2}");
            Console.WriteLine($"Recall:    {m.PositiveRecall:P2}");
            Console.WriteLine($"F1 Score:  {m.F1Score:P2}");
        }

        private class WeightOutput { public float Weight { get; set; } }
    }
}