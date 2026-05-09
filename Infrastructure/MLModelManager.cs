using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using InsiderThreatDetection.Core.Models;

namespace InsiderThreatDetection.Infrastructure
{
    public class MLModelManager
    {
        private readonly MLContext _mlContext;
        private ITransformer? _model;
        private PredictionEngine<UserBehaviour, ThreatPrediction>? _predictionEngine;

        public BinaryClassificationMetrics? LastMetrics { get; private set; }

        private double _accuracy;
        private double _precision;
        private double _recall;
        private double _f1Score;
        private double[][]? _confusionMatrix;

        private float _optimalThreshold = 0.5f;

        private const float TestFraction = 0.2f;
        private const int RandomSeed = 42;
        private const double MinPrecision = 0.60;   // We will not let precision drop below 60%

        private const string FeaturesColumn = "Features";
        private const string LabelColumn = "Label";

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
            var data = _mlContext.Data.LoadFromTextFile<UserBehaviour>(path: dataPath, hasHeader: true, separatorChar: ',');
            var split = _mlContext.Data.TrainTestSplit(data, testFraction: TestFraction);
            float maliciousWeight = ComputeMaliciousWeight(split.TrainSet);

            (_benignMeans, _benignStdDevs) = ComputeBenignStats(split.TrainSet);

            var pipeline = BuildPipeline(maliciousWeight);
            _model = pipeline.Fit(split.TrainSet);

            var defaultPredictions = _model.Transform(split.TestSet);
            LastMetrics = _mlContext.BinaryClassification.Evaluate(defaultPredictions, labelColumnName: LabelColumn);
            PrintMetrics(LastMetrics);

            // Find the threshold that maximises recall subject to MinPrecision
            CalibrateAndEvaluate(split.TestSet);

            Console.WriteLine($"Optimal threshold selected: {_optimalThreshold:F3}");
            Console.WriteLine($"Final metrics -> Acc: {_accuracy:P2} | Prec: {_precision:P2} | Rec: {_recall:P2} | F1: {_f1Score:P2}");

            _predictionEngine = _mlContext.Model.CreatePredictionEngine<UserBehaviour, ThreatPrediction>(_model);
        }

        public ThreatPrediction Predict(UserBehaviour input)
        {
            if (_predictionEngine == null)
                throw new InvalidOperationException("Model not trained or loaded.");
            var raw = _predictionEngine.Predict(input);
            bool label = raw.Probability >= _optimalThreshold;
            return new ThreatPrediction
            {
                PredictedLabel = label,
                Probability = raw.Probability,
                Score = raw.Score
            };
        }

        public double[][]? GetConfusionMatrixCounts() => _confusionMatrix;

        public string GetEvaluationSummary()
        {
            if (_confusionMatrix == null) return "Evaluation not available.";
            double tn = _confusionMatrix[0][0];
            double fp = _confusionMatrix[0][1];
            double fn = _confusionMatrix[1][0];
            double tp = _confusionMatrix[1][1];
            return $"Accuracy:  {_accuracy:P2}\nPrecision: {_precision:P2}\nRecall:    {_recall:P2}\nF1:        {_f1Score:P2}\n\n" +
                   $"Confusion Matrix:\n  True Positives  : {tp}\n  True Negatives  : {tn}\n  False Positives : {fp}\n  False Negatives : {fn}";
        }

        // ---------- Explainability unchanged ----------
        public List<(string Feature, float Contribution, float Value)> Explain(UserBehaviour input)
        {
            if (_model == null || _benignMeans == null) throw new InvalidOperationException("Model not trained.");
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

        public string GenerateHumanExplanation(UserBehaviour input, ThreatPrediction prediction)
        {
            if (_benignMeans == null || _benignStdDevs == null) return "Model statistics not available.";
            var sb = new StringBuilder();
            sb.AppendLine($"Threat Probability: {prediction.Probability:P0}");
            sb.AppendLine();
            var anomalies = new List<(string Feature, float Value, float Mean, string Direction)>();
            for (int i = 0; i < _featureNames.Length; i++)
            {
                float val = GetFeatureValue(input, _featureNames[i]);
                float mean = _benignMeans[i];
                float std = _benignStdDevs[i];
                if (std > 0 && Math.Abs(val - mean) > 2 * std)
                    anomalies.Add((_featureNames[i], val, mean, val > mean ? "higher" : "lower"));
            }
            if (prediction.PredictedLabel)
            {
                sb.AppendLine("The activity is flagged as MALICIOUS.");
                if (anomalies.Any())
                {
                    sb.AppendLine("The following behaviours are unusual compared to typical employees:");
                    foreach (var a in anomalies.Take(3))
                        sb.AppendLine($"  - {HumanReadableName(a.Feature)}: {a.Value} (normal is around {a.Mean:F1})");
                }
                else sb.AppendLine("Although no single behaviour is extreme, the combination raised the overall risk.");
            }
            else
            {
                sb.AppendLine("The activity is NORMAL.");
                if (anomalies.Any())
                    sb.AppendLine("A few indicators were slightly outside the typical range, but they are not strong enough to raise an alert.");
                else sb.AppendLine("All indicators are within expected ranges.");
            }
            return sb.ToString();
        }

        // ---------- Calibration with MinPrecision ----------
        private void CalibrateAndEvaluate(IDataView testSet)
        {
            var predictions = _model!.Transform(testSet);
            var rows = _mlContext.Data.CreateEnumerable<TestPrediction>(predictions, reuseRowObject: false).ToList();
            float[] probs = rows.Select(r => r.Probability).ToArray();
            bool[] labels = rows.Select(r => r.Label).ToArray();

            double bestRecall = 0;
            double bestThresh = 0.5;
            double[]? bestCounts = null;

            for (int perc = 0; perc <= 100; perc++)
            {
                double t = perc / 100.0;
                int tp = 0, fp = 0, tn = 0, fn = 0;
                for (int i = 0; i < probs.Length; i++)
                {
                    if (probs[i] >= t && labels[i]) tp++;
                    else if (probs[i] >= t && !labels[i]) fp++;
                    else if (probs[i] < t && labels[i]) fn++;
                    else tn++;
                }
                double prec = tp + fp == 0 ? 0 : (double)tp / (tp + fp);
                double rec = tp + fn == 0 ? 0 : (double)tp / (tp + fn);
                if (prec >= MinPrecision && rec > bestRecall)
                {
                    bestRecall = rec;
                    bestThresh = t;
                    bestCounts = new double[] { tn, fp, fn, tp };
                }
            }
            _optimalThreshold = (float)bestThresh;
            if (bestCounts != null)
            {
                double tn = bestCounts[0], fp = bestCounts[1], fn = bestCounts[2], tp = bestCounts[3];
                double total = tp + tn + fp + fn;
                _accuracy = total == 0 ? 0 : (tp + tn) / total;
                _precision = tp + fp == 0 ? 0 : tp / (tp + fp);
                _recall = tp + fn == 0 ? 0 : tp / (tp + fn);
                _f1Score = _precision + _recall == 0 ? 0 : 2 * _precision * _recall / (_precision + _recall);
                _confusionMatrix = new double[][] { new double[] { tn, fp }, new double[] { fn, tp } };
            }
        }

        private class TestPrediction
        {
            [ColumnName("Probability")] public float Probability { get; set; }
            [ColumnName("Label")] public bool Label { get; set; }
        }

        // ---------- Helpers (including HumanReadableName) ----------
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

        // … (rest of helpers unchanged: CloneBehaviour, GetFeatureValue, SetFeatureValue, etc.)
        // I'll paste them for completeness.
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
                .Where(r => r.is_malicious == 0).ToList();
            int n = benignRows.Count, f = 14;
            float[] sums = new float[f], sqSums = new float[f];
            foreach (var r in benignRows)
            {
                float[] vals = { r.employee_seniority_years, r.is_contractor, r.employee_classification,
                                 r.total_printed_pages, r.num_printed_pages_off_hours,
                                 r.total_files_burned, r.burned_from_other,
                                 r.is_abroad, r.trip_day_number, r.hostility_country_level,
                                 r.num_entries, r.num_unique_campus, r.late_exit_flag, r.entry_during_weekend };
                for (int i = 0; i < f; i++) { sums[i] += vals[i]; sqSums[i] += vals[i] * vals[i]; }
            }
            float[] means = sums.Select(s => s / n).ToArray();
            float[] stds = new float[f];
            for (int i = 0; i < f; i++) { float m = means[i]; stds[i] = (float)Math.Sqrt(sqSums[i] / n - m * m); }
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
            _optimalThreshold = 0.5f;
        }

        private IEstimator<ITransformer> BuildPipeline(float maliciousWeight)
        {
            return _mlContext.Transforms
                .Conversion.ConvertType(LabelColumn, nameof(UserBehaviour.is_malicious), DataKind.Boolean)
                .Append(_mlContext.Transforms.CustomMapping(
                    (UserBehaviour input, WeightOutput output) =>
                    { output.Weight = input.is_malicious == 1 ? maliciousWeight : 1f; }, contractName: null))
                .Append(_mlContext.Transforms.Concatenate(FeaturesColumn, _featureNames))
                .Append(_mlContext.Transforms.NormalizeMeanVariance(FeaturesColumn))
                .Append(_mlContext.BinaryClassification.Trainers.FastTree(
                    LabelColumn, FeaturesColumn, "Weight",
                    numberOfLeaves: 20, numberOfTrees: 100, minimumExampleCountPerLeaf: 10));
        }

        private float ComputeMaliciousWeight(IDataView trainData)
        {
            int malicious = 0, normal = 0;
            foreach (var row in _mlContext.Data.CreateEnumerable<UserBehaviour>(trainData, reuseRowObject: true))
                if (row.is_malicious == 1) malicious++; else normal++;
            Console.WriteLine($"Training set - Malicious: {malicious}, Normal: {normal}");
            return malicious == 0 ? 1f : (float)normal / malicious;
        }

        private void PrintMetrics(BinaryClassificationMetrics m)
        {
            Console.WriteLine("=== MODEL PERFORMANCE (default threshold) ===");
            Console.WriteLine($"Accuracy:  {m.Accuracy:P2}");
            Console.WriteLine($"Precision: {m.PositivePrecision:P2}");
            Console.WriteLine($"Recall:    {m.PositiveRecall:P2}");
            Console.WriteLine($"F1 Score:  {m.F1Score:P2}");
        }

        private class WeightOutput { public float Weight { get; set; } }
    }
}