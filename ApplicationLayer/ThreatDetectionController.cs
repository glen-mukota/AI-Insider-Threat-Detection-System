using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using InsiderThreatDetection.Core.Models;
using InsiderThreatDetection.Infrastructure;

namespace InsiderThreatDetection.ApplicationLayer
{
    public class ThreatDetectionController
    {
        private readonly MLModelManager _modelManager;
        private UserBehaviour? _normalProfile;

        private static readonly string[] RequiredColumns = new[]
        {
            "employee_department", "employee_campus", "employee_position", "employee_origin_country",
            "employee_seniority_years", "is_contractor", "employee_classification",
            "has_foreign_citizenship", "has_criminal_record", "has_medical_history",
            "total_printed_pages", "num_printed_pages_off_hours",
            "total_files_burned", "burned_from_other",
            "is_abroad", "trip_day_number", "hostility_country_level",
            "num_entries", "num_unique_campus", "late_exit_flag", "entry_during_weekend",
            "is_malicious"
        };

        public ThreatDetectionController()
        {
            _modelManager = new MLModelManager();
        }

        public void TrainModel(string dataPath)
        {
            _modelManager.Train(dataPath);
            _normalProfile = FindNormalProfile(dataPath);
        }

        public float GetBaselineProbability() => _modelManager.GetNeutralBaselineProbability();

        /// <summary>
        /// Returns a fully populated ThreatPrediction with correct confidence and classification.
        /// </summary>
        public ThreatPrediction Predict(UserBehaviour input)
        {
            var raw = _modelManager.Predict(input);   // raw prediction (PredictedLabel set with optimal threshold)
            float threatProb = raw.Probability;
            string classification = threatProb >= _modelManager.OptimalThreshold ? "MALICIOUS" : "NORMAL";
            float confidence = classification == "MALICIOUS" ? threatProb : 1.0f - threatProb;

            return new ThreatPrediction
            {
                PredictedLabel = classification == "MALICIOUS",
                Probability = threatProb,
                Score = raw.Score,
                Classification = classification,
                Confidence = confidence,
                ThreatProbability = threatProb
            };
        }

        public List<(string Feature, float Contribution, float Value)> Explain(UserBehaviour input) =>
            _modelManager.Explain(input);

        public string GenerateHumanExplanation(UserBehaviour input, ThreatPrediction prediction) =>
            _modelManager.GenerateHumanExplanation(input, prediction);

        public void SaveModel(string path) => _modelManager.SaveModel(path);
        public void LoadModel(string path) => _modelManager.LoadModel(path);

        public string GetEvaluationSummary() => _modelManager.GetEvaluationSummary();
        public string GetModelComparisonTable() => _modelManager.GetModelComparisonTable();

        /// <summary>
        /// Reads a single behavioural record from a CSV file.
        /// </summary>
        public UserBehaviour LoadSingleRowFromCsv(string filePath, int rowIndex = 1)
        {
            var lines = File.ReadAllLines(filePath);
            if (lines.Length < 2)
                throw new InvalidOperationException("CSV must contain a header row and at least one data row.");

            var headers = lines[0].Split(',').Select(h => h.Trim()).ToArray();

            var missing = RequiredColumns.Where(c => !headers.Contains(c, StringComparer.OrdinalIgnoreCase)).ToList();
            if (missing.Any())
                throw new InvalidOperationException(
                    $"CSV is missing required columns: {string.Join(", ", missing)}");

            if (lines.Length <= rowIndex)
                throw new InvalidOperationException("CSV does not contain the requested row.");

            var values = lines[rowIndex].Split(',');

            if (values.Length != headers.Length)
                throw new InvalidOperationException(
                    $"Column count mismatch: header has {headers.Length} columns, data row has {values.Length}.");

            var user = ParseRow(headers, values);
            if (user == null)
                throw new InvalidOperationException("Failed to parse CSV row – check numeric values for validity.");
            return user;
        }

        /// <summary>
        /// Returns a pre‑defined employee profile.
        /// </summary>
        public UserBehaviour GetProfile(string profileName)
        {
            return profileName switch
            {
                "Normal Office Worker" => _normalProfile
                    ?? throw new InvalidOperationException("Model must be trained before using the normal profile."),

                "Suspicious Printing Activity" => new UserBehaviour
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
                },

                "Excessive Facility Access" => new UserBehaviour
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
                },

                "Critical Insider Threat" => new UserBehaviour
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
                    num_printed_pages_off_hours = 350,
                    total_files_burned = 15,
                    burned_from_other = 10,
                    is_abroad = 1,
                    trip_day_number = 14,
                    hostility_country_level = 5,
                    num_entries = 160,
                    num_unique_campus = 8,
                    late_exit_flag = 1,
                    entry_during_weekend = 1
                },

                _ => throw new ArgumentException("Unknown profile")
            };
        }

        // ---------------------------------------------------------------------
        //  PRIVATE HELPERS
        // ---------------------------------------------------------------------
        private UserBehaviour? FindNormalProfile(string dataPath)
        {
            var lines = File.ReadAllLines(dataPath);
            if (lines.Length < 2) return null;
            var headers = lines[0].Split(',');
            for (int i = 1; i < Math.Min(lines.Length, 1001); i++)
            {
                var user = ParseRow(headers, lines[i].Split(','));
                if (user != null && user.is_malicious == 0)
                {
                    var pred = _modelManager.Predict(user);
                    if (!pred.PredictedLabel)
                        return user;
                }
            }
            return null;
        }

        private UserBehaviour? ParseRow(string[] headers, string[] values)
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
                    is_malicious = 0
                };
            }
            catch
            {
                return null;
            }
        }

        private string GetString(string[] headers, string[] values, string colName)
        {
            int idx = Array.FindIndex(headers, h => h.Trim().Equals(colName, StringComparison.OrdinalIgnoreCase));
            return (idx >= 0 && idx < values.Length) ? values[idx] : string.Empty;
        }

        private float GetFloat(string[] headers, string[] values, string colName)
        {
            int idx = Array.FindIndex(headers, h => h.Trim().Equals(colName, StringComparison.OrdinalIgnoreCase));
            if (idx < 0 || idx >= values.Length) return 0f;
            return float.TryParse(values[idx], NumberStyles.Float, CultureInfo.InvariantCulture, out float v) ? v : 0f;
        }
    }
}