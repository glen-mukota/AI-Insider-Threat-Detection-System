// =============================================================================
//  ThreatDetectionController.cs
//  Insider Threat Detection System – COS720 2026
//
//  This is the Application Layer facade (Façade pattern).
//  It orchestrates: DataPreprocessor → MLModelManager → AuditLogger.
//
//  CIA Triad:
//    Confidentiality – all data processed locally; audit logs are local.
//    Integrity        – input validated and cleaned before prediction.
//    Availability     – meaningful exceptions with LastErrorMessage so UI
//                       can recover gracefully without crashing.
//
//  Software Engineering: Dependency Inversion – UI depends on this interface,
//  not on MLModelManager or DataPreprocessor directly.
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
        // -----------------------------------------------------------------------
        //  DEPENDENCIES
        // -----------------------------------------------------------------------
        private readonly MLModelManager _modelManager;
        private readonly DataPreprocessor _preprocessor;

        private UserBehaviour? _normalProfile;

        public bool IsModelReady { get; private set; }
        public string? LastErrorMessage => _modelManager.LastErrorMessage;

        // -----------------------------------------------------------------------
        //  REQUIRED COLUMNS
        // -----------------------------------------------------------------------
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
            "entry_during_weekend", "is_malicious"
        };

        // -----------------------------------------------------------------------
        //  CONSTRUCTOR
        // -----------------------------------------------------------------------
        public ThreatDetectionController()
        {
            _modelManager = new MLModelManager();
            _preprocessor = new DataPreprocessor();
        }

        // -----------------------------------------------------------------------
        //  TRAINING (with preprocessing)
        // -----------------------------------------------------------------------

        /// <summary>
        /// Preprocesses the raw CSV, trains the model, and returns a preprocessing report.
        /// </summary>
        public DataPreprocessor.PreprocessingReport TrainModel(string rawDataPath)
        {
            IsModelReady = false;

            // ── Step 1: Preprocess raw data ────────────────────────────────────
            var (cleanedPath, report) = _preprocessor.Preprocess(rawDataPath);
            AuditLogger.Instance.LogPreprocessing(report.ToString());

            try
            {
                // ── Step 2: Train ML model on cleaned data ─────────────────────
                _modelManager.Train(cleanedPath);

                // ── Step 3: Find a representative normal profile for demo ──────
                _normalProfile = FindNormalProfile(cleanedPath);

                IsModelReady = true;
                return report;
            }
            finally
            {
                // Clean up temp file
                try { if (File.Exists(cleanedPath)) File.Delete(cleanedPath); }
                catch { /* Non-critical */ }
            }
        }

        // -----------------------------------------------------------------------
        //  PREDICTION
        // -----------------------------------------------------------------------
        public ThreatPrediction Predict(UserBehaviour input)
        {
            EnsureModelReady();
            var cleaned = _preprocessor.CleanSingleRecord(input);
            var prediction = _modelManager.Predict(cleaned);

            AuditLogger.Instance.LogPrediction(
                "UI", prediction.Classification, prediction.Probability, prediction.RiskLevel);

            return prediction;
        }

        // -----------------------------------------------------------------------
        //  EXPLAINABILITY
        // -----------------------------------------------------------------------
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

        // -----------------------------------------------------------------------
        //  EVALUATION
        // -----------------------------------------------------------------------
        public float GetBaselineProbability() => _modelManager.GetNeutralBaselineProbability();
        public string GetEvaluationSummary() => _modelManager.GetEvaluationSummary();
        public string GetModelComparisonTable() => _modelManager.GetModelComparisonTable();

        // -----------------------------------------------------------------------
        //  PERSISTENCE
        // -----------------------------------------------------------------------
        public void SaveModel(string path)
        {
            EnsureModelReady();
            _modelManager.SaveModel(path);
        }

        public void LoadModel(string path)
        {
            _modelManager.LoadModel(path);
            IsModelReady = true;
            _normalProfile = null; // Will be re-derived on first use if available
        }

        // -----------------------------------------------------------------------
        //  CSV UPLOAD – SINGLE RECORD
        // -----------------------------------------------------------------------

        /// <summary>
        /// Reads a single behavioural record from a CSV file (row at <paramref name="rowIndex"/>),
        /// applies single-record cleaning, and returns the UserBehaviour object.
        /// </summary>
        public UserBehaviour LoadSingleRowFromCsv(string filePath, int rowIndex = 1)
        {
            // Security: validate the file is a plain text CSV (Integrity)
            if (!File.Exists(filePath))
                throw new FileNotFoundException("CSV file not found.", filePath);

            var lines = File.ReadAllLines(filePath);
            if (lines.Length < 2)
                throw new InvalidOperationException("CSV must contain a header row and at least one data row.");

            var headers = lines[0].Split(',').Select(h => h.Trim()).ToArray();

            var missing = RequiredColumns
                .Where(c => !headers.Contains(c, StringComparer.OrdinalIgnoreCase))
                .ToList();
            if (missing.Any())
                throw new InvalidOperationException(
                    $"CSV is missing required columns: {string.Join(", ", missing)}");

            if (lines.Length <= rowIndex)
                throw new InvalidOperationException(
                    $"CSV does not contain row {rowIndex}. File has {lines.Length - 1} data row(s).");

            var values = lines[rowIndex].Split(',');
            if (values.Length != headers.Length)
                throw new InvalidOperationException(
                    $"Column count mismatch: header has {headers.Length}, data row has {values.Length}.");

            var raw = ParseRow(headers, values)
                ?? throw new InvalidOperationException(
                    "Failed to parse the CSV row. Please check numeric fields for validity.");

            // Clean the single record (domain-knowledge bounds + logical consistency)
            return _preprocessor.CleanSingleRecord(raw);
        }

        // -----------------------------------------------------------------------
        //  SAMPLE PROFILES
        // -----------------------------------------------------------------------

        public UserBehaviour GetProfile(string profileName)
        {
            return profileName switch
            {
                "Normal Office Worker" =>
                    _normalProfile
                    ?? throw new InvalidOperationException(
                        "Normal profile unavailable. Please train the model with the full dataset first."),

                "Suspicious Printing Activity" => _preprocessor.CleanSingleRecord(new UserBehaviour
                {
                    employee_department = "Engineering Department",
                    employee_campus = "Campus A",
                    employee_position = "Design Engineer",
                    employee_origin_country = "Israel",
                    employee_seniority_years = 3,
                    is_contractor = 0,
                    employee_classification = 2,
                    has_foreign_citizenship = 0,
                    has_criminal_record = 0,
                    has_medical_history = 0,
                    total_printed_pages = 250,
                    num_printed_pages_off_hours = 200,
                    total_files_burned = 0,
                    burned_from_other = 0,
                    is_abroad = 0,
                    trip_day_number = 0,
                    hostility_country_level = 0,
                    num_entries = 12,
                    num_unique_campus = 1,
                    late_exit_flag = 0,
                    entry_during_weekend = 1
                }),

                "Excessive Facility Access" => _preprocessor.CleanSingleRecord(new UserBehaviour
                {
                    employee_department = "R&D Department",
                    employee_campus = "Campus B",
                    employee_position = "Systems Engineer",
                    employee_origin_country = "Ukraine",
                    employee_seniority_years = 1,
                    is_contractor = 1,
                    employee_classification = 1,
                    has_foreign_citizenship = 1,
                    has_criminal_record = 1,
                    has_medical_history = 0,
                    total_printed_pages = 250,
                    num_printed_pages_off_hours = 250,
                    total_files_burned = 10,
                    burned_from_other = 5,
                    is_abroad = 0,
                    trip_day_number = 0,
                    hostility_country_level = 0,
                    num_entries = 200,
                    num_unique_campus = 6,
                    late_exit_flag = 1,
                    entry_during_weekend = 1
                }),

                "Critical Insider Threat" => _preprocessor.CleanSingleRecord(new UserBehaviour
                {
                    employee_department = "Information Technology",
                    employee_campus = "Campus A",
                    employee_position = "Data Scientist",
                    employee_origin_country = "UK",
                    employee_seniority_years = 1,
                    is_contractor = 1,
                    employee_classification = 0,
                    has_foreign_citizenship = 1,
                    has_criminal_record = 1,
                    has_medical_history = 0,
                    total_printed_pages = 250,
                    num_printed_pages_off_hours = 250,
                    total_files_burned = 15,
                    burned_from_other = 10,
                    is_abroad = 1,
                    trip_day_number = 14,
                    hostility_country_level = 5,
                    num_entries = 160,
                    num_unique_campus = 8,
                    late_exit_flag = 1,
                    entry_during_weekend = 1
                }),

                _ => throw new ArgumentException($"Unknown profile: '{profileName}'")
            };
        }

        // -----------------------------------------------------------------------
        //  AUDIT LOG
        // -----------------------------------------------------------------------
        public string GetAuditLogPath() => AuditLogger.Instance.GetCurrentLogPath();
        public string GetRecentAuditEntries() => AuditLogger.Instance.GetRecentEntries(30);

        // -----------------------------------------------------------------------
        //  PRIVATE HELPERS
        // -----------------------------------------------------------------------
        private void EnsureModelReady()
        {
            if (!IsModelReady)
                throw new InvalidOperationException(
                    "The model is not ready. Please upload a dataset and train the model first.");
        }

        private UserBehaviour? FindNormalProfile(string dataPath)
        {
            try
            {
                var lines = File.ReadAllLines(dataPath);
                if (lines.Length < 2) return null;
                var headers = lines[0].Split(',');

                for (int i = 1; i < Math.Min(lines.Length, 2001); i++)
                {
                    var user = ParseRow(headers, lines[i].Split(','));
                    if (user == null || user.is_malicious != 0) continue;

                    var pred = _modelManager.Predict(user);
                    if (!pred.PredictedLabel)
                        return user;
                }
            }
            catch { /* Non-critical – profile simply won't be available */ }
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
            int idx = Array.FindIndex(headers, h => h.Trim().Equals(col, StringComparison.OrdinalIgnoreCase));
            return (idx >= 0 && idx < values.Length) ? values[idx].Trim() : string.Empty;
        }

        private static float GetFloat(string[] headers, string[] values, string col)
        {
            int idx = Array.FindIndex(headers, h => h.Trim().Equals(col, StringComparison.OrdinalIgnoreCase));
            if (idx < 0 || idx >= values.Length) return 0f;
            return float.TryParse(values[idx].Trim(), NumberStyles.Float,
                CultureInfo.InvariantCulture, out float v) ? v : 0f;
        }
    }
}