// =============================================================================
//  ThreatDetectionController.cs
//  Insider Threat Detection System – COS720 2026
//
//  Application Layer Façade — orchestrates:
//    DataPreprocessor → MLModelManager → AuditLogger
//
//  CIA Triad:
//    Confidentiality – all processing local; audit logs never transmitted.
//    Integrity        – input validated and cleaned before prediction.
//    Availability     – meaningful exceptions with LastErrorMessage so UI
//                       can recover without crashing.
//
//  Design: Dependency Inversion — UI depends only on this controller,
//  never directly on MLModelManager or DataPreprocessor.
// =============================================================================

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using InsiderThreatDetection.Core.Models;
using InsiderThreatDetection.Core.Services;
using InsiderThreatDetection.Infrastructure;

namespace InsiderThreatDetection.ApplicationLayer
{
    public class ThreatDetectionController
    {
        // ── Dependencies ───────────────────────────────────────────────────────
        private readonly MLModelManager _modelManager;
        private readonly DataPreprocessor _preprocessor;

        private UserBehaviour? _normalProfile;

        public bool IsModelReady { get; private set; }
        public string? LastErrorMessage => _modelManager.LastErrorMessage;

        // ── Required feature columns for live CSV prediction ───────────────────
        // Prediction files do not need an is_malicious label because real
        // production records are unlabeled.
        private static readonly string[] RequiredColumns =
        {
            "employee_department", "employee_campus", "employee_position",
            "employee_origin_country", "employee_seniority_years",
            "is_contractor", "employee_classification",
            "has_foreign_citizenship", "has_criminal_record", "has_medical_history",
            "total_printed_pages", "num_printed_pages_off_hours",
            "total_files_burned", "burned_from_other",
            "is_abroad", "trip_day_number", "hostility_country_level",
            "num_entries", "num_unique_campus", "late_exit_flag",
            "entry_during_weekend"
        };

        // ── Constructor ────────────────────────────────────────────────────────
        public ThreatDetectionController()
        {
            _modelManager = new MLModelManager();
            _preprocessor = new DataPreprocessor();
        }

        // ── Training ───────────────────────────────────────────────────────────

        /// <summary>
        /// Preprocesses the raw CSV, trains the model, and returns a preprocessing
        /// report for display in the UI.
        /// </summary>
        public DataPreprocessor.PreprocessingReport TrainModel(string rawDataPath)
        {
            IsModelReady = false;

            // Step 1: Preprocess raw data
            var (cleanedPath, report) = _preprocessor.Preprocess(rawDataPath);
            AuditLogger.Instance.LogPreprocessing(report.ToString());

            try
            {
                // Step 2: Train ML model on cleaned data
                _modelManager.Train(cleanedPath);

                // Step 3: Find a representative normal profile for demo
                _normalProfile = FindNormalProfile(cleanedPath);

                IsModelReady = true;
                return report;
            }
            finally
            {
                // Clean up temp file (CIA Confidentiality)
                try { if (File.Exists(cleanedPath)) File.Delete(cleanedPath); }
                catch { /* Non-critical */ }
            }
        }

        // ── Prediction ─────────────────────────────────────────────────────────
        public ThreatPrediction Predict(UserBehaviour input)
        {
            EnsureModelReady();
            var cleaned = _preprocessor.CleanSingleRecord(input);
            var prediction = _modelManager.Predict(cleaned);

            AuditLogger.Instance.LogPrediction(
                "UI", prediction.Classification,
                prediction.Probability, prediction.RiskLevel);

            return prediction;
        }

        // ── Explainability ─────────────────────────────────────────────────────
        public List<(string Feature, float Contribution, float Value, float BenignMean)>
            Explain(UserBehaviour input)
        {
            EnsureModelReady();
            return _modelManager.Explain(input);
        }

        public string GenerateHumanExplanation(UserBehaviour input, ThreatPrediction prediction)
        {
            EnsureModelReady();
            return _modelManager.GenerateHumanExplanation(input, prediction);
        }

        // ── Evaluation ─────────────────────────────────────────────────────────
        public float GetBaselineProbability() => _modelManager.GetNeutralBaselineProbability();
        public float GetCalibratedThreshold() => _modelManager.GetCalibratedThreshold();
        public string GetEvaluationSummary() => _modelManager.GetEvaluationSummary();
        public string GetModelComparisonTable() => _modelManager.GetModelComparisonTable();

        // ── Persistence ────────────────────────────────────────────────────────
        public void SaveModel(string path)
        {
            EnsureModelReady();
            _modelManager.SaveModel(path);
        }

        public void LoadModel(string path)
        {
            _modelManager.LoadModel(path);
            IsModelReady = true;
            _normalProfile = null;
        }

        // ── CSV upload: single record ───────────────────────────────────────────

        /// <summary>
        /// Reads, validates, parses, and cleans a single behavioural record
        /// from a CSV file (data row at <paramref name="rowIndex"/>, 1-based).
        /// Returns the cleaned UserBehaviour ready for prediction.
        /// </summary>
        public UserBehaviour LoadSingleRowFromCsv(string filePath, int rowIndex = 1)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("CSV file not found.", filePath);

            var lines = File.ReadAllLines(filePath);
            if (lines.Length < 2)
                throw new InvalidOperationException(
                    "CSV must contain a header row and at least one data row.");

            var headers = SplitCsvLine(lines[0]).Select(h => h.Trim()).ToArray();

            var missing = RequiredColumns
                .Where(c => !headers.Contains(c, StringComparer.OrdinalIgnoreCase))
                .ToList();
            if (missing.Any())
                throw new InvalidOperationException(
                    $"CSV is missing required columns: {string.Join(", ", missing)}");

            if (lines.Length <= rowIndex)
                throw new InvalidOperationException(
                    $"CSV does not contain row {rowIndex}. File has {lines.Length - 1} data row(s).");

            var values = SplitCsvLine(lines[rowIndex]);
            if (values.Length != headers.Length)
                throw new InvalidOperationException(
                    $"Column count mismatch on row {rowIndex}: " +
                    $"header has {headers.Length} columns, data row has {values.Length}.");

            var raw = ParseRow(headers, values)
                ?? throw new InvalidOperationException(
                    "Failed to parse the CSV row. Check that all numeric fields are valid numbers.");

            return _preprocessor.CleanSingleRecord(raw);
        }

        /// <summary>
        /// Predicts all rows in a CSV (up to maxRows) and returns results.
        /// Used for batch CSV prediction.
        /// </summary>
        public List<(int Row, UserBehaviour Input, ThreatPrediction Prediction)>
            PredictAllRowsFromCsv(string filePath, int maxRows = 50)
        {
            EnsureModelReady();

            if (!File.Exists(filePath))
                throw new FileNotFoundException("CSV file not found.", filePath);

            var lines = File.ReadAllLines(filePath);
            if (lines.Length < 2)
                throw new InvalidOperationException("CSV must have a header and at least one data row.");

            var headers = SplitCsvLine(lines[0]).Select(h => h.Trim()).ToArray();
            var missing = RequiredColumns
                .Where(c => !headers.Contains(c, StringComparer.OrdinalIgnoreCase))
                .ToList();
            if (missing.Any())
                throw new InvalidOperationException(
                    $"CSV is missing required columns: {string.Join(", ", missing)}");

            var results = new List<(int, UserBehaviour, ThreatPrediction)>();
            int limit = Math.Min(lines.Length - 1, maxRows);

            for (int i = 1; i <= limit; i++)
            {
                var values = SplitCsvLine(lines[i]);
                if (values.Length != headers.Length) continue;

                var raw = ParseRow(headers, values);
                if (raw == null) continue;

                var cleaned = _preprocessor.CleanSingleRecord(raw);
                var prediction = _modelManager.Predict(cleaned);

                AuditLogger.Instance.LogCsvPrediction(
                    Path.GetFileName(filePath), i,
                    prediction.Classification, prediction.Probability, prediction.RiskLevel);

                results.Add((i, cleaned, prediction));
            }

            return results;
        }

        // ── Sample profiles ────────────────────────────────────────────────────

        /// <summary>
        /// Returns a pre-defined sample UserBehaviour profile by name.
        /// Profiles are designed to be representative of real dataset patterns,
        /// so predictions are accurate and meaningful for demonstration.
        /// </summary>
        public UserBehaviour GetProfile(string profileName)
        {
            return profileName switch
            {
                "Normal Office Worker" =>
                    _normalProfile
                    ?? _preprocessor.CleanSingleRecord(new UserBehaviour
                    {
                        // Representative benign employee from the dataset.
                        // Kept as a stable fallback when a saved model is loaded
                        // and the training-time normal profile is not available.
                        employee_department = "Engineering Department",
                        employee_campus = "Campus C",
                        employee_position = "Design Engineer",
                        employee_origin_country = "Georgia",
                        employee_seniority_years = 22,
                        is_contractor = 0,
                        employee_classification = 2,
                        has_foreign_citizenship = 0,
                        has_criminal_record = 0,
                        has_medical_history = 0,
                        total_printed_pages = 0,
                        num_printed_pages_off_hours = 0,
                        total_files_burned = 0,
                        burned_from_other = 0,
                        is_abroad = 0,
                        trip_day_number = 0,
                        hostility_country_level = 0,
                        num_entries = 1,
                        num_unique_campus = 1,
                        late_exit_flag = 0,
                        entry_during_weekend = 1
                    }),

                "Suspicious Printing Activity" => _preprocessor.CleanSingleRecord(new UserBehaviour
                {
                    // Representative malicious row: moderate off-hours printing
                    // paired with identity/background risk, not an artificial max.
                    employee_department = "R&D Department",
                    employee_campus = "Campus A",
                    employee_position = "Integration and Testing Engineer",
                    employee_origin_country = "Israel",
                    employee_seniority_years = 23,
                    is_contractor = 0,
                    employee_classification = 2,
                    has_foreign_citizenship = 1,
                    has_criminal_record = 0,
                    has_medical_history = 1,
                    total_printed_pages = 92,
                    num_printed_pages_off_hours = 17,
                    total_files_burned = 0,
                    burned_from_other = 0,
                    is_abroad = 0,
                    trip_day_number = 0,
                    hostility_country_level = 0,
                    num_entries = 0,
                    num_unique_campus = 0,
                    late_exit_flag = 0,
                    entry_during_weekend = 0
                }),

                "Excessive Facility Access" => _preprocessor.CleanSingleRecord(new UserBehaviour
                {
                    // Representative malicious row: repeated entries across
                    // three campuses plus removable-media activity.
                    employee_department = "Engineering Department",
                    employee_campus = "Campus B",
                    employee_position = "Systems Engineer",
                    employee_origin_country = "South Africa",
                    employee_seniority_years = 10,
                    is_contractor = 0,
                    employee_classification = 2,
                    has_foreign_citizenship = 0,
                    has_criminal_record = 0,
                    has_medical_history = 0,
                    total_printed_pages = 0,
                    num_printed_pages_off_hours = 0,
                    total_files_burned = 93,
                    burned_from_other = 0,
                    is_abroad = 0,
                    trip_day_number = 0,
                    hostility_country_level = 0,
                    num_entries = 4,
                    num_unique_campus = 3,
                    late_exit_flag = 0,
                    entry_during_weekend = 1
                }),

                "Critical Insider Threat" => _preprocessor.CleanSingleRecord(new UserBehaviour
                {
                    // Representative malicious row: hostile foreign travel plus
                    // substantial removable-media activity.
                    employee_department = "Information Technology",
                    employee_campus = "Campus A",
                    employee_position = "Enterprise Systems Developer (ERP / CRM / SAP)",
                    employee_origin_country = "Morocco",
                    employee_seniority_years = 2,
                    is_contractor = 0,
                    employee_classification = 2,
                    has_foreign_citizenship = 0,
                    has_criminal_record = 0,
                    has_medical_history = 1,
                    total_printed_pages = 0,
                    num_printed_pages_off_hours = 0,
                    total_files_burned = 138,
                    burned_from_other = 0,
                    is_abroad = 1,
                    trip_day_number = 7,
                    hostility_country_level = 3,
                    num_entries = 0,
                    num_unique_campus = 0,
                    late_exit_flag = 0,
                    entry_during_weekend = 0
                }),

                _ => throw new ArgumentException($"Unknown profile: '{profileName}'")
            };
        }

        // ── Audit log ──────────────────────────────────────────────────────────
        public string GetAuditLogPath() => AuditLogger.Instance.GetCurrentLogPath();
        public string GetRecentAuditEntries() => AuditLogger.Instance.GetRecentEntries(50);

        // ── Private helpers ────────────────────────────────────────────────────
        private void EnsureModelReady()
        {
            if (!IsModelReady)
                throw new InvalidOperationException(
                    "The model is not ready. Please upload a dataset and train the model first.");
        }

        /// <summary>
        /// Scans the training data for a representative normal profile to use
        /// as the "Normal Office Worker" sample.
        /// </summary>
        private UserBehaviour? FindNormalProfile(string dataPath)
        {
            try
            {
                var lines = File.ReadAllLines(dataPath);
                if (lines.Length < 2) return null;
                var headers = SplitCsvLine(lines[0]);

                for (int i = 1; i < Math.Min(lines.Length, 5001); i++)
                {
                    var user = ParseRow(headers, SplitCsvLine(lines[i]));
                    if (user == null || user.is_malicious != 0) continue;

                    var pred = _modelManager.Predict(user);
                    // Find someone the model is confident is normal
                    if (!pred.PredictedLabel && pred.Probability < 0.2f)
                        return user;
                }
            }
            catch { /* Non-critical */ }
            return null;
        }

        private static UserBehaviour? ParseRow(string[] headers, string[] values)
        {
            if (headers.Length != values.Length) return null;
            try
            {
                return new UserBehaviour
                {
                    employee_department = GetString(headers, values, "employee_department"),
                    employee_campus = GetString(headers, values, "employee_campus"),
                    employee_position = GetString(headers, values, "employee_position"),
                    employee_origin_country = GetString(headers, values, "employee_origin_country"),
                    employee_seniority_years = GetFloat(headers, values, "employee_seniority_years"),
                    is_contractor = GetFloat(headers, values, "is_contractor"),
                    employee_classification = GetFloat(headers, values, "employee_classification"),
                    has_foreign_citizenship = GetFloat(headers, values, "has_foreign_citizenship"),
                    has_criminal_record = GetFloat(headers, values, "has_criminal_record"),
                    has_medical_history = GetFloat(headers, values, "has_medical_history"),
                    total_printed_pages = GetFloat(headers, values, "total_printed_pages"),
                    num_printed_pages_off_hours = GetFloat(headers, values, "num_printed_pages_off_hours"),
                    total_files_burned = GetFloat(headers, values, "total_files_burned"),
                    burned_from_other = GetFloat(headers, values, "burned_from_other"),
                    is_abroad = GetFloat(headers, values, "is_abroad"),
                    trip_day_number = GetFloat(headers, values, "trip_day_number"),
                    hostility_country_level = GetFloat(headers, values, "hostility_country_level"),
                    num_entries = GetFloat(headers, values, "num_entries"),
                    num_unique_campus = GetFloat(headers, values, "num_unique_campus"),
                    late_exit_flag = GetFloat(headers, values, "late_exit_flag"),
                    entry_during_weekend = GetFloat(headers, values, "entry_during_weekend"),
                    is_malicious = GetFloat(headers, values, "is_malicious")
                };
            }
            catch { return null; }
        }

        private static string GetString(string[] headers, string[] values, string col)
        {
            int idx = Array.FindIndex(
                headers, h => h.Trim().Equals(col, StringComparison.OrdinalIgnoreCase));
            return (idx >= 0 && idx < values.Length) ? values[idx].Trim() : string.Empty;
        }

        private static float GetFloat(string[] headers, string[] values, string col)
        {
            int idx = Array.FindIndex(
                headers, h => h.Trim().Equals(col, StringComparison.OrdinalIgnoreCase));
            if (idx < 0 || idx >= values.Length) return 0f;
            return float.TryParse(values[idx].Trim(), NumberStyles.Float,
                CultureInfo.InvariantCulture, out float v) ? v : 0f;
        }

        private static string[] SplitCsvLine(string line)
        {
            var fields = new List<string>();
            bool inQuotes = false;
            var current = new System.Text.StringBuilder();

            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];
                if (c == '"')
                {
                    if (inQuotes && i + 1 < line.Length && line[i + 1] == '"')
                    {
                        current.Append('"');
                        i++;
                    }
                    else
                    {
                        inQuotes = !inQuotes;
                    }
                }
                else if (c == ',' && !inQuotes)
                {
                    fields.Add(current.ToString());
                    current.Clear();
                }
                else
                {
                    current.Append(c);
                }
            }

            fields.Add(current.ToString());
            return fields.ToArray();
        }
    }
}
