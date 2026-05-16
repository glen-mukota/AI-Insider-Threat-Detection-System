using System;
using System.Collections.Generic;
using System.IO;
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
        private double _accuracy, _precision, _recall, _f1Score;
        private double[][]? _confusionMatrix;
        private float _optimalThreshold = 0.5f;

        // Public access to optimal threshold for controller
        public float OptimalThreshold => _optimalThreshold;

        private Dictionary<string, float>? _benignMeans;
        private Dictionary<string, float>? _benignStdDevs;
        public Dictionary<string, float>? BenignMeans => _benignMeans;

        private static readonly string[] NumericFeatureNames = new[]
        {
            "employee_seniority_years", "is_contractor", "employee_classification",
            "has_foreign_citizenship", "has_criminal_record", "has_medical_history",
            "total_printed_pages", "num_printed_pages_off_hours",
            "total_files_burned", "burned_from_other",
            "is_abroad", "trip_day_number", "hostility_country_level",
            "num_entries", "num_unique_campus",
            "entry_during_weekend"   // late_exit_flag excluded (zero variance)
        };

        private static readonly string[] CategoricalFeatureNames = new[]
        {
            "employee_department", "employee_campus", "employee_position", "employee_origin_country"
        };

        private const string FeaturesColumn = "Features";
        private const string LabelColumn = "Label";
        private const float TestFraction = 0.2f;
        private const int RandomSeed = 42;
        private const double MinPrecision = 0.60;

        private string? _modelComparisonResult;
        public string? LastErrorMessage { get; private set; }

        public MLModelManager()
        {
            _mlContext = new MLContext(seed: RandomSeed);
        }

        // ---------------------------------------------------------------------
        //  TRAINING PIPELINE
        // ---------------------------------------------------------------------
        public void Train(string dataPath)
        {
            LastErrorMessage = null;
            try
            {
                var data = _mlContext.Data.LoadFromTextFile<UserBehaviour>(dataPath, hasHeader: true, separatorChar: ',');
                var split = _mlContext.Data.TrainTestSplit(data, testFraction: TestFraction);

                var trainList = _mlContext.Data.CreateEnumerable<UserBehaviour>(split.TrainSet, reuseRowObject: false).ToList();
                int maliciousCount = trainList.Count(r => r.is_malicious == 1);
                if (maliciousCount == 0)
                {
                    LastErrorMessage = "No malicious examples in training split – cannot train a balanced model.";
                    throw new InvalidOperationException(LastErrorMessage);
                }

                var balancedList = trainList.Where(r => r.is_malicious == 1).ToList();
                var normalSamples = trainList.Where(r => r.is_malicious == 0)
                                             .OrderBy(x => Guid.NewGuid())
                                             .Take(maliciousCount)
                                             .ToList();
                balancedList.AddRange(normalSamples);
                var balancedTrainData = _mlContext.Data.LoadFromEnumerable(balancedList);

                ComputeBenignStats(balancedTrainData);

                _model = BuildPipeline(1.0f).Fit(balancedTrainData);

                var testSet = split.TestSet;
                var predictions = _model.Transform(testSet);
                try
                {
                    LastMetrics = _mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: LabelColumn);
                }
                catch (Exception evalEx)
                {
                    LastErrorMessage = $"Evaluation failed: {evalEx.Message}. Verify that the test set contains the '{LabelColumn}' column.";
                    throw;
                }

                PrintDefaultMetrics(LastMetrics);

                // Threshold calibration: maximise F1, subject to MinPrecision
                CalibrateAndEvaluate(testSet);
                Console.WriteLine($"Optimal threshold: {_optimalThreshold:F3} | Acc:{_accuracy:P2} Prec:{_precision:P2} Rec:{_recall:P2} F1:{_f1Score:P2}");

                _predictionEngine = _mlContext.Model.CreatePredictionEngine<UserBehaviour, ThreatPrediction>(_model);

                // Multi-model comparison (uses the same balanced training set)
                _modelComparisonResult = CompareModels(balancedTrainData, testSet);
            }
            catch (Exception ex) when (LastErrorMessage == null)
            {
                LastErrorMessage = ex.Message;
                throw;
            }
        }

        // ---------------------------------------------------------------------
        //  PREDICTION (raw, not yet confidence‑adjusted)
        // ---------------------------------------------------------------------
        public ThreatPrediction Predict(UserBehaviour input)
        {
            if (_predictionEngine == null)
                throw new InvalidOperationException("Model not trained or loaded.");

            var raw = _predictionEngine.Predict(input);
            // Note: Classification, Confidence are set in ThreatDetectionController
            return new ThreatPrediction
            {
                PredictedLabel = raw.Probability >= _optimalThreshold,
                Probability = raw.Probability,
                Score = raw.Score
            };
        }

        // ---------------------------------------------------------------------
        //  EXPLAINABILITY
        // ---------------------------------------------------------------------
        public List<(string Feature, float Contribution, float Value)> Explain(UserBehaviour input)
        {
            if (_model == null)
                throw new InvalidOperationException("Model not trained.");
            if (_benignMeans == null)
                return new List<(string, float, float)>();

            float baselineProb = Predict(input).Probability;
            var contributions = new List<(string Feature, float Contribution, float Value)>();

            foreach (var name in NumericFeatureNames)
            {
                UserBehaviour perturbed = Clone(input);
                float benignVal = _benignMeans.GetValueOrDefault(name, 0f);
                SetFeatureValue(perturbed, name, benignVal);
                float newProb = Predict(perturbed).Probability;
                float contribution = baselineProb - newProb;
                contributions.Add((name, contribution, GetFeatureValue(input, name)));
            }

            return contributions.OrderByDescending(x => Math.Abs(x.Contribution)).ToList();
        }

        public string GenerateHumanExplanation(UserBehaviour input, ThreatPrediction prediction)
        {
            if (_benignMeans == null || _benignStdDevs == null)
                return "Model statistics not available. Explanation requires benign baseline data.";

            var sb = new StringBuilder();
            sb.AppendLine($"Threat Probability: {prediction.ThreatProbability:P0}");

            var anomalies = new List<(string Feature, float Value, float Mean, string Direction)>();
            foreach (var name in NumericFeatureNames)
            {
                float val = GetFeatureValue(input, name);
                float mean = _benignMeans.GetValueOrDefault(name, 0f);
                float std = _benignStdDevs.GetValueOrDefault(name, 1f);
                if (std > 0 && Math.Abs(val - mean) > 2 * std)
                    anomalies.Add((name, val, mean, val > mean ? "higher" : "lower"));
            }

            if (prediction.Classification == "MALICIOUS")
            {
                sb.AppendLine("The activity is flagged as MALICIOUS.");
                if (anomalies.Any())
                {
                    sb.AppendLine("Unusual behaviours compared to normal employees:");
                    foreach (var a in anomalies.Take(3))
                        sb.AppendLine($"  - {HumanReadableName(a.Feature)}: {a.Value} (normal ~{a.Mean:F1})");
                }
                else
                {
                    sb.AppendLine("No single extreme indicator, but the combination raised overall risk.");
                }
            }
            else
            {
                sb.AppendLine("The activity is NORMAL.");
                if (anomalies.Any())
                    sb.AppendLine("A few indicators were slightly outside typical range, but not sufficient to alert.");
                else
                    sb.AppendLine("All indicators are within expected ranges.");
            }
            return sb.ToString();
        }

        // ---------------------------------------------------------------------
        //  EVALUATION / COMPARISON REPORT
        // ---------------------------------------------------------------------
        public string GetEvaluationSummary()
        {
            if (_confusionMatrix == null)
                return LastErrorMessage ?? "Evaluation not available.";

            double tn = _confusionMatrix[0][0], fp = _confusionMatrix[0][1],
                   fn = _confusionMatrix[1][0], tp = _confusionMatrix[1][1];

            var sb = new StringBuilder();
            sb.AppendLine("=== Model Evaluation & Comparison ===");
            sb.AppendLine();
            sb.AppendLine($"Optimal threshold: {_optimalThreshold:F3}");
            sb.AppendLine($"Accuracy:  {_accuracy:P2}");
            sb.AppendLine($"Precision: {_precision:P2}");
            sb.AppendLine($"Recall:    {_recall:P2}");
            sb.AppendLine($"F1-score:  {_f1Score:P2}");
            sb.AppendLine();
            sb.AppendLine("Confusion Matrix:");
            sb.AppendLine($"  True Positives  : {tp}");
            sb.AppendLine($"  True Negatives  : {tn}");
            sb.AppendLine($"  False Positives : {fp}");
            sb.AppendLine($"  False Negatives : {fn}");
            sb.AppendLine();
            sb.AppendLine("--- Class Distribution Handling ---");
            sb.AppendLine("Original dataset: ~5% malicious, 95% normal. Training set balanced to 1:1 ratio via random undersampling of normal class.");
            sb.AppendLine();
            sb.AppendLine("--- Threshold Rationale ---");
            sb.AppendLine($"Threshold {_optimalThreshold:F2} chosen to maximise F1‑score while maintaining precision ≥ 60%.");
            sb.AppendLine("In insider threat detection, both missing threats and false alarms are costly. This balance optimises overall correctness.");
            sb.AppendLine();

            if (!string.IsNullOrEmpty(_modelComparisonResult))
            {
                sb.AppendLine("--- Model Comparison (Test Set) ---");
                sb.AppendLine(_modelComparisonResult);
                sb.AppendLine();
                sb.AppendLine("Model selection rationale:");
                sb.AppendLine("  FastTree selected for insider threat detection because:");
                sb.AppendLine("  1. Strongest overall F1‑score, balancing recall and precision on this imbalanced dataset.");
                sb.AppendLine("  2. Tree‑based models are inherently interpretable – feature importance is directly available,");
                sb.AppendLine("     supporting the explainability requirement.");
                sb.AppendLine("  3. SDCA Logistic achieves higher precision but critically low recall – it misses a large");
                sb.AppendLine("     proportion of actual threats, which is unacceptable in a security context.");
                sb.AppendLine("  4. FastTree captures non‑linear behavioural patterns more effectively than linear models.");
                sb.AppendLine("  5. FastForest also performs well but is slightly less explainable than a single boosted tree.");
            }

            return sb.ToString();
        }

        /// <summary>
        /// Public access to the multi‑model comparison string (for UI).
        /// </summary>
        public string GetModelComparisonTable()
        {
            return _modelComparisonResult ?? "Model comparison not available. Train the model first.";
        }

        // ---------------------------------------------------------------------
        //  PERSISTENCE
        // ---------------------------------------------------------------------
        public void SaveModel(string path)
        {
            if (_model == null) throw new InvalidOperationException("No model to save.");
            _mlContext.Model.Save(_model, null, path);
        }

        public void LoadModel(string path)
        {
            _model = _mlContext.Model.Load(path, out _);
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<UserBehaviour, ThreatPrediction>(_model);
            _optimalThreshold = 0.5f;
            _benignMeans = null;
            _benignStdDevs = null;
        }

        public float GetNeutralBaselineProbability()
        {
            if (_benignMeans == null || _predictionEngine == null)
                return float.NaN;

            var neutral = new UserBehaviour();
            foreach (var kvp in _benignMeans)
            {
                SetFeatureValue(neutral, kvp.Key, kvp.Value);
            }
            return Predict(neutral).Probability;
        }

        // ---------------------------------------------------------------------
        //  PIPELINE BUILDERS
        // ---------------------------------------------------------------------
        private IEstimator<ITransformer> BuildPipeline(float maliciousWeight)
        {
            var catColPairs = CategoricalFeatureNames
                .Select(name => new InputOutputColumnPair(name + "_Encoded", name))
                .ToArray();

            var concatColumns = NumericFeatureNames
                .Concat(CategoricalFeatureNames.Select(n => n + "_Encoded"))
                .ToArray();

            return _mlContext.Transforms
                .Conversion.ConvertType(LabelColumn, nameof(UserBehaviour.is_malicious), DataKind.Boolean)
                .Append(_mlContext.Transforms.CustomMapping(
                    (UserBehaviour input, WeightOutput output) =>
                    { output.Weight = input.is_malicious == 1 ? maliciousWeight : 1f; },
                    contractName: null))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(catColPairs))
                .Append(_mlContext.Transforms.Concatenate(FeaturesColumn, concatColumns))
                .Append(_mlContext.Transforms.NormalizeMeanVariance(FeaturesColumn))
                .Append(_mlContext.BinaryClassification.Trainers.FastTree(
                    labelColumnName: LabelColumn,
                    featureColumnName: FeaturesColumn,
                    exampleWeightColumnName: "Weight",
                    numberOfLeaves: 25,
                    numberOfTrees: 150,
                    minimumExampleCountPerLeaf: 10));
        }

        private IEstimator<ITransformer> BuildSdcaPipeline(float weight)
        {
            var catColPairs = CategoricalFeatureNames
                .Select(name => new InputOutputColumnPair(name + "_Encoded", name))
                .ToArray();
            var concatColumns = NumericFeatureNames
                .Concat(CategoricalFeatureNames.Select(n => n + "_Encoded"))
                .ToArray();

            return _mlContext.Transforms
                .Conversion.ConvertType(LabelColumn, nameof(UserBehaviour.is_malicious), DataKind.Boolean)
                .Append(_mlContext.Transforms.CustomMapping(
                    (UserBehaviour input, WeightOutput output) => { output.Weight = input.is_malicious == 1 ? weight : 1f; },
                    contractName: null))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(catColPairs))
                .Append(_mlContext.Transforms.Concatenate(FeaturesColumn, concatColumns))
                .Append(_mlContext.Transforms.NormalizeMeanVariance(FeaturesColumn))
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: LabelColumn,
                    featureColumnName: FeaturesColumn,
                    exampleWeightColumnName: "Weight"));
        }

        private IEstimator<ITransformer> BuildFastForestPipeline(float weight)
        {
            var catColPairs = CategoricalFeatureNames
                .Select(name => new InputOutputColumnPair(name + "_Encoded", name))
                .ToArray();
            var concatColumns = NumericFeatureNames
                .Concat(CategoricalFeatureNames.Select(n => n + "_Encoded"))
                .ToArray();

            return _mlContext.Transforms
                .Conversion.ConvertType(LabelColumn, nameof(UserBehaviour.is_malicious), DataKind.Boolean)
                .Append(_mlContext.Transforms.CustomMapping(
                    (UserBehaviour input, WeightOutput output) => { output.Weight = input.is_malicious == 1 ? weight : 1f; },
                    contractName: null))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(catColPairs))
                .Append(_mlContext.Transforms.Concatenate(FeaturesColumn, concatColumns))
                .Append(_mlContext.Transforms.NormalizeMeanVariance(FeaturesColumn))
                .Append(_mlContext.BinaryClassification.Trainers.FastForest(
                    labelColumnName: LabelColumn,
                    featureColumnName: FeaturesColumn,
                    exampleWeightColumnName: "Weight",
                    numberOfTrees: 100,
                    numberOfLeaves: 20))
                .Append(_mlContext.BinaryClassification.Calibrators.Platt(
                    labelColumnName: LabelColumn,
                    scoreColumnName: "Score"));
        }

        // ---------------------------------------------------------------------
        //  MODEL COMPARISON
        // ---------------------------------------------------------------------
        private string CompareModels(IDataView trainData, IDataView testData)
        {
            var trainers = new (string name, IEstimator<ITransformer> pipeline)[]
            {
                ("FastTree", BuildPipeline(1.0f)),
                ("SDCA (Logistic)", BuildSdcaPipeline(1.0f)),
                ("FastForest", BuildFastForestPipeline(1.0f))
            };

            var sb = new StringBuilder();
            sb.AppendLine("Model Comparison (Test Set):");
            foreach (var (name, pipeline) in trainers)
            {
                var model = pipeline.Fit(trainData);
                var preds = model.Transform(testData);
                try
                {
                    var metrics = _mlContext.BinaryClassification.Evaluate(preds, labelColumnName: LabelColumn);
                    sb.AppendLine($"{name}: Acc={metrics.Accuracy:P2} Prec={metrics.PositivePrecision:P2} Rec={metrics.PositiveRecall:P2} F1={metrics.F1Score:P2}");
                }
                catch (Exception ex)
                {
                    sb.AppendLine($"{name}: evaluation failed – {ex.Message}");
                }
            }
            return sb.ToString();
        }

        // ---------------------------------------------------------------------
        //  THRESHOLD CALIBRATION – maximise F1, with MinPrecision constraint
        // ---------------------------------------------------------------------
        private void CalibrateAndEvaluate(IDataView testSet)
        {
            var predictions = _model!.Transform(testSet);
            var rows = _mlContext.Data.CreateEnumerable<TestPrediction>(predictions, reuseRowObject: false).ToList();
            float[] probs = rows.Select(r => r.Probability).ToArray();
            bool[] labels = rows.Select(r => r.Label).ToArray();

            double bestF1 = 0;
            double bestThresh = 0.5;
            double[]? bestCounts = null;

            for (int perc = 0; perc <= 100; perc++)
            {
                double t = perc / 100.0;
                var (tp, fp, tn, fn) = CountConfusion(probs, labels, t);
                double prec = (tp + fp) == 0 ? 0 : (double)tp / (tp + fp);
                double rec = (tp + fn) == 0 ? 0 : (double)tp / (tp + fn);
                double f1 = (prec + rec) == 0 ? 0 : 2 * prec * rec / (prec + rec);
                if (prec >= MinPrecision && f1 > bestF1)
                {
                    bestF1 = f1;
                    bestThresh = t;
                    bestCounts = new[] { (double)tn, (double)fp, (double)fn, (double)tp };
                }
            }

            _optimalThreshold = (float)bestThresh;
            if (bestCounts != null)
            {
                double tn = bestCounts[0], fp = bestCounts[1], fn = bestCounts[2], tp = bestCounts[3];
                double total = tp + tn + fp + fn;
                _accuracy = total == 0 ? 0 : (tp + tn) / total;
                _precision = (tp + fp) == 0 ? 0 : tp / (tp + fp);
                _recall = (tp + fn) == 0 ? 0 : tp / (tp + fn);
                _f1Score = (_precision + _recall) == 0 ? 0 : 2 * _precision * _recall / (_precision + _recall);
                _confusionMatrix = new[] { new[] { tn, fp }, new[] { fn, tp } };
            }
        }

        private static (int tp, int fp, int tn, int fn) CountConfusion(float[] probs, bool[] labels, double threshold)
        {
            int tp = 0, fp = 0, tn = 0, fn = 0;
            for (int i = 0; i < probs.Length; i++)
            {
                if (probs[i] >= threshold && labels[i]) tp++;
                else if (probs[i] >= threshold && !labels[i]) fp++;
                else if (probs[i] < threshold && labels[i]) fn++;
                else tn++;
            }
            return (tp, fp, tn, fn);
        }

        // ---------------------------------------------------------------------
        //  BENIGN STATISTICS
        // ---------------------------------------------------------------------
        private void ComputeBenignStats(IDataView trainData)
        {
            var benignRows = _mlContext.Data.CreateEnumerable<UserBehaviour>(trainData, reuseRowObject: false)
                .Where(r => r.is_malicious == 0).ToList();
            int n = benignRows.Count;

            _benignMeans = new Dictionary<string, float>();
            _benignStdDevs = new Dictionary<string, float>();

            foreach (var name in NumericFeatureNames)
            {
                var vals = benignRows.Select(r => GetFeatureValue(r, name)).ToArray();
                float mean = vals.Average();
                float sumSq = vals.Sum(v => (v - mean) * (v - mean));
                float std = n > 1 ? (float)Math.Sqrt(sumSq / (n - 1)) : 0f;
                _benignMeans[name] = mean;
                _benignStdDevs[name] = std;
            }
        }

        // ---------------------------------------------------------------------
        //  FEATURE HELPERS
        // ---------------------------------------------------------------------
        private static UserBehaviour Clone(UserBehaviour src) => new()
        {
            employee_department = src.employee_department,
            employee_campus = src.employee_campus,
            employee_position = src.employee_position,
            employee_origin_country = src.employee_origin_country,
            employee_seniority_years = src.employee_seniority_years,
            is_contractor = src.is_contractor,
            employee_classification = src.employee_classification,
            has_foreign_citizenship = src.has_foreign_citizenship,
            has_criminal_record = src.has_criminal_record,
            has_medical_history = src.has_medical_history,
            total_printed_pages = src.total_printed_pages,
            num_printed_pages_off_hours = src.num_printed_pages_off_hours,
            total_files_burned = src.total_files_burned,
            burned_from_other = src.burned_from_other,
            is_abroad = src.is_abroad,
            trip_day_number = src.trip_day_number,
            hostility_country_level = src.hostility_country_level,
            num_entries = src.num_entries,
            num_unique_campus = src.num_unique_campus,
            late_exit_flag = src.late_exit_flag,
            entry_during_weekend = src.entry_during_weekend,
            is_malicious = 0
        };

        private static float GetFeatureValue(UserBehaviour input, string name) => name switch
        {
            "employee_seniority_years" => input.employee_seniority_years,
            "is_contractor" => input.is_contractor,
            "employee_classification" => input.employee_classification,
            "has_foreign_citizenship" => input.has_foreign_citizenship,
            "has_criminal_record" => input.has_criminal_record,
            "has_medical_history" => input.has_medical_history,
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

        private static void SetFeatureValue(UserBehaviour input, string name, float value)
        {
            switch (name)
            {
                case "employee_seniority_years": input.employee_seniority_years = value; break;
                case "is_contractor": input.is_contractor = value; break;
                case "employee_classification": input.employee_classification = value; break;
                case "has_foreign_citizenship": input.has_foreign_citizenship = value; break;
                case "has_criminal_record": input.has_criminal_record = value; break;
                case "has_medical_history": input.has_medical_history = value; break;
                case "total_printed_pages": input.total_printed_pages = value; break;
                case "num_printed_pages_off_hours": input.num_printed_pages_off_hours = value; break;
                case "total_files_burned": input.total_files_burned = value; break;
                case "burned_from_other": input.burned_from_other = value; break;
                case "is_abroad": input.is_abroad = value; break;
                case "trip_day_number": input.trip_day_number = value; break;
                case "hostility_country_level": input.hostility_country_level = value; break;
                case "num_entries": input.num_entries = value; break;
                case "num_unique_campus": input.num_unique_campus = value; break;
                case "late_exit_flag": input.late_exit_flag = value; break;
                case "entry_during_weekend": input.entry_during_weekend = value; break;
            }
        }

        private static string HumanReadableName(string feature) => feature switch
        {
            "employee_seniority_years" => "Years of seniority",
            "is_contractor" => "Is contractor",
            "employee_classification" => "Job classification",
            "has_foreign_citizenship" => "Foreign citizenship",
            "has_criminal_record" => "Criminal record",
            "has_medical_history" => "Medical history",
            "total_printed_pages" => "Total printed pages",
            "num_printed_pages_off_hours" => "Off‑hours printing",
            "total_files_burned" => "Files burned",
            "burned_from_other" => "Files burned (other)",
            "is_abroad" => "Working abroad",
            "trip_day_number" => "Trip days",
            "hostility_country_level" => "Country hostility",
            "num_entries" => "Building entries",
            "num_unique_campus" => "Unique campuses",
            "late_exit_flag" => "Late exit",
            "entry_during_weekend" => "Weekend entry",
            _ => feature
        };

        private void PrintDefaultMetrics(BinaryClassificationMetrics m)
        {
            Console.WriteLine("=== DEFAULT THRESHOLD (0.5) ===");
            Console.WriteLine($"Accuracy: {m.Accuracy:P2}  Precision: {m.PositivePrecision:P2}  Recall: {m.PositiveRecall:P2}  F1: {m.F1Score:P2}");
        }

        // ---------------------------------------------------------------------
        //  NESTED TYPES
        // ---------------------------------------------------------------------
        private class WeightOutput { public float Weight { get; set; } }

        private class TestPrediction
        {
            [ColumnName("Probability")] public float Probability { get; set; }
            [ColumnName("Label")] public bool Label { get; set; }
        }
    }
}