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

        public ThreatDetectionController()
        {
            _modelManager = new MLModelManager();
        }

        public void TrainModel(string dataPath)
        {
            _modelManager.Train(dataPath);

            _normalProfile = FindNormalProfile(dataPath);
            if (_normalProfile == null)
            {
                var means = _modelManager.BenignMeans;
                if (means != null && means.Length == 14)
                {
                    _normalProfile = new UserBehaviour
                    {
                        employee_seniority_years = means[0],
                        is_contractor = means[1],
                        employee_classification = means[2],
                        total_printed_pages = means[3],
                        num_printed_pages_off_hours = means[4],
                        total_files_burned = means[5],
                        burned_from_other = means[6],
                        is_abroad = means[7],
                        trip_day_number = means[8],
                        hostility_country_level = means[9],
                        num_entries = means[10],
                        num_unique_campus = means[11],
                        late_exit_flag = means[12],
                        entry_during_weekend = means[13],
                        is_malicious = 0
                    };
                }
            }
        }

        private UserBehaviour? FindNormalProfile(string dataPath)
        {
            try
            {
                var lines = File.ReadAllLines(dataPath);
                if (lines.Length < 2) return null;

                var headers = lines[0].Split(',');
                for (int i = 1; i < Math.Min(lines.Length, 1001); i++)
                {
                    var user = ParseRow(headers, lines[i].Split(','));
                    if (user != null && user.is_malicious == 0)
                    {
                        var prediction = _modelManager.Predict(user);
                        if (!prediction.PredictedLabel)
                            return user;
                    }
                }
            }
            catch { }
            return null;
        }

        private UserBehaviour? ParseRow(string[] headers, string[] values)
        {
            try
            {
                float Get(string colName) => TryGetFloat(headers, values, colName);
                return new UserBehaviour
                {
                    employee_seniority_years = Get("employee_seniority_years"),
                    is_contractor = Get("is_contractor"),
                    employee_classification = Get("employee_classification"),
                    total_printed_pages = Get("total_printed_pages"),
                    num_printed_pages_off_hours = Get("num_printed_pages_off_hours"),
                    total_files_burned = Get("total_files_burned"),
                    burned_from_other = Get("burned_from_other"),
                    is_abroad = Get("is_abroad"),
                    trip_day_number = Get("trip_day_number"),
                    hostility_country_level = Get("hostility_country_level"),
                    num_entries = Get("num_entries"),
                    num_unique_campus = Get("num_unique_campus"),
                    late_exit_flag = Get("late_exit_flag"),
                    entry_during_weekend = Get("entry_during_weekend"),
                    is_malicious = Get("is_malicious")
                };
            }
            catch
            {
                return null;
            }
        }

        public ThreatPrediction Predict(UserBehaviour input) => _modelManager.Predict(input);
        public List<(string Feature, float Contribution, float Value)> Explain(UserBehaviour input) => _modelManager.Explain(input);
        public string GenerateHumanExplanation(UserBehaviour input, ThreatPrediction prediction) => _modelManager.GenerateHumanExplanation(input, prediction);
        public void SaveModel(string path) => _modelManager.SaveModel(path);
        public void LoadModel(string path) => _modelManager.LoadModel(path);

        /// <summary> Returns the formatted evaluation summary (metrics + confusion matrix). </summary>
        public string GetEvaluationSummary() => _modelManager.GetEvaluationSummary();

        public UserBehaviour LoadSingleRowFromCsv(string filePath, int rowIndex = 1)
        {
            var lines = File.ReadAllLines(filePath);
            if (lines.Length <= rowIndex)
                throw new InvalidOperationException("CSV does not contain the requested row.");

            var headers = lines[0].Split(',');
            var values = lines[rowIndex].Split(',');

            float Get(string colName) => TryGetFloat(headers, values, colName);

            return new UserBehaviour
            {
                employee_seniority_years = Get("employee_seniority_years"),
                is_contractor = Get("is_contractor"),
                employee_classification = Get("employee_classification"),
                total_printed_pages = Get("total_printed_pages"),
                num_printed_pages_off_hours = Get("num_printed_pages_off_hours"),
                total_files_burned = Get("total_files_burned"),
                burned_from_other = Get("burned_from_other"),
                is_abroad = Get("is_abroad"),
                trip_day_number = Get("trip_day_number"),
                hostility_country_level = Get("hostility_country_level"),
                num_entries = Get("num_entries"),
                num_unique_campus = Get("num_unique_campus"),
                late_exit_flag = Get("late_exit_flag"),
                entry_during_weekend = Get("entry_during_weekend"),
                is_malicious = 0
            };
        }

        private float TryGetFloat(string[] headers, string[] values, string colName)
        {
            int idx = Array.FindIndex(headers, h => h.Trim().Equals(colName, StringComparison.OrdinalIgnoreCase));
            if (idx < 0 || idx >= values.Length) return 0f;
            return float.TryParse(values[idx], NumberStyles.Float, CultureInfo.InvariantCulture, out float v) ? v : 0f;
        }

        public UserBehaviour GetProfile(string profileName)
        {
            return profileName switch
            {
                "Normal Office Worker" => _normalProfile
                    ?? throw new InvalidOperationException("Model must be trained before using the normal profile."),
                "Suspicious Printing Activity" => new UserBehaviour
                {
                    employee_seniority_years = 3f,
                    is_contractor = 0,
                    employee_classification = 1,
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
                    employee_seniority_years = 8f,
                    is_contractor = 0,
                    employee_classification = 3,
                    total_printed_pages = 30,
                    num_printed_pages_off_hours = 5,
                    total_files_burned = 0,
                    burned_from_other = 0,
                    is_abroad = 0,
                    trip_day_number = 0,
                    hostility_country_level = 0,
                    num_entries = 80,
                    num_unique_campus = 3,
                    late_exit_flag = 1,
                    entry_during_weekend = 1
                },
                "Critical Insider Threat" => new UserBehaviour
                {
                    employee_seniority_years = 2f,
                    is_contractor = 1,
                    employee_classification = 0,
                    total_printed_pages = 400,
                    num_printed_pages_off_hours = 380,
                    total_files_burned = 10,
                    burned_from_other = 5,
                    is_abroad = 1,
                    trip_day_number = 7,
                    hostility_country_level = 4,
                    num_entries = 100,
                    num_unique_campus = 4,
                    late_exit_flag = 1,
                    entry_during_weekend = 1
                },
                _ => throw new ArgumentException("Unknown profile")
            };
        }
    }
}