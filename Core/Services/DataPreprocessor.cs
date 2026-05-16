// =============================================================================
//  DataPreprocessor.cs
//  Insider Threat Detection System – COS720 2026
//  Responsibility: Raw CSV cleaning, validation, and normalisation before
//  data is passed to the ML pipeline.
//
//  CIA Triad relevance:
//    Integrity  – ensures only valid, clean data enters the model so that
//                 predictions are trustworthy and not corrupted by noise.
//    Availability – robust error handling prevents crashes on bad input,
//                   keeping the system operational.
// =============================================================================

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using InsiderThreatDetection.Core.Models;

namespace InsiderThreatDetection.Core.Services
{
    /// <summary>
    /// Performs data cleaning and preprocessing on the raw Kaggle insider-threat
    /// CSV dataset before it is handed to the ML pipeline.
    ///
    /// Preprocessing steps implemented:
    ///   1. Header validation – ensures all required columns are present.
    ///   2. Missing-value imputation – numeric columns → median; categorical → mode.
    ///   3. Outlier capping (IQR method) on continuous numeric columns.
    ///   4. Categorical normalisation – trims whitespace, lower-cases for matching.
    ///   5. Type coercion – converts string numeric fields to float safely.
    ///   6. Duplicate row detection and removal.
    ///   7. Label validation – ensures is_malicious is 0 or 1.
    ///   8. Produces a preprocessing report for auditability (Integrity / CIA).
    /// </summary>
    public class DataPreprocessor
    {
        // -----------------------------------------------------------------------
        //  CONSTANTS
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

        private static readonly string[] NumericColumns =
        {
            "employee_seniority_years", "is_contractor", "employee_classification",
            "has_foreign_citizenship", "has_criminal_record", "has_medical_history",
            "total_printed_pages", "num_printed_pages_off_hours",
            "total_files_burned", "burned_from_other",
            "is_abroad", "trip_day_number", "hostility_country_level",
            "num_entries", "num_unique_campus", "late_exit_flag",
            "entry_during_weekend"
        };

        private static readonly string[] ContinuousColumns =
        {
            "employee_seniority_years",
            "total_printed_pages", "num_printed_pages_off_hours",
            "total_files_burned", "burned_from_other",
            "trip_day_number", "num_entries", "num_unique_campus"
        };

        private static readonly string[] CategoricalColumns =
        {
            "employee_department", "employee_campus",
            "employee_position", "employee_origin_country"
        };

        // -----------------------------------------------------------------------
        //  PREPROCESSING REPORT (returned to caller for display / audit log)
        // -----------------------------------------------------------------------
        public class PreprocessingReport
        {
            public int OriginalRowCount { get; set; }
            public int CleanedRowCount { get; set; }
            public int DuplicatesRemoved { get; set; }
            public int InvalidLabelRows { get; set; }
            public int MissingValuesImputed { get; set; }
            public int OutliersCapped { get; set; }
            public Dictionary<string, string> ImputationValues { get; set; } = new();
            public List<string> Warnings { get; set; } = new();

            public override string ToString()
            {
                var sb = new StringBuilder();
                sb.AppendLine("=== Data Preprocessing Report ===");
                sb.AppendLine($"Original rows        : {OriginalRowCount:N0}");
                sb.AppendLine($"Duplicates removed   : {DuplicatesRemoved:N0}");
                sb.AppendLine($"Invalid label rows   : {InvalidLabelRows:N0}");
                sb.AppendLine($"Missing values fixed : {MissingValuesImputed:N0}");
                sb.AppendLine($"Outliers capped      : {OutliersCapped:N0}");
                sb.AppendLine($"Final clean rows     : {CleanedRowCount:N0}");
                if (Warnings.Count > 0)
                {
                    sb.AppendLine("Warnings:");
                    foreach (var w in Warnings)
                        sb.AppendLine($"  ⚠ {w}");
                }
                return sb.ToString();
            }
        }

        // -----------------------------------------------------------------------
        //  PUBLIC ENTRY POINT
        // -----------------------------------------------------------------------

        /// <summary>
        /// Reads the raw CSV from <paramref name="inputPath"/>, applies all
        /// preprocessing steps, writes the cleaned CSV to a temp file and
        /// returns the temp path plus a preprocessing report.
        /// </summary>
        public (string cleanedPath, PreprocessingReport report) Preprocess(string inputPath)
        {
            var report = new PreprocessingReport();

            // ── 1. Read raw lines ──────────────────────────────────────────
            var rawLines = File.ReadAllLines(inputPath);
            if (rawLines.Length < 2)
                throw new InvalidOperationException("CSV file has no data rows.");

            var headers = rawLines[0].Split(',').Select(h => h.Trim()).ToArray();
            ValidateHeaders(headers);

            report.OriginalRowCount = rawLines.Length - 1;

            // ── 2. Parse rows into dictionaries ────────────────────────────
            var rows = new List<Dictionary<string, string>>();
            for (int i = 1; i < rawLines.Length; i++)
            {
                var values = SplitCsvLine(rawLines[i]);
                if (values.Length != headers.Length)
                {
                    report.Warnings.Add($"Row {i} skipped: column count mismatch ({values.Length} vs {headers.Length}).");
                    continue;
                }
                var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                for (int j = 0; j < headers.Length; j++)
                    dict[headers[j]] = values[j].Trim();
                rows.Add(dict);
            }

            // ── 3. Remove duplicates ───────────────────────────────────────
            var unique = rows.GroupBy(r => string.Join("|", headers.Select(h => r[h]))).Select(g => g.First()).ToList();
            report.DuplicatesRemoved = rows.Count - unique.Count;
            rows = unique;

            // ── 4. Validate & drop rows with invalid labels ────────────────
            var validRows = new List<Dictionary<string, string>>();
            foreach (var row in rows)
            {
                string labelStr = row.GetValueOrDefault("is_malicious", "");
                if (labelStr == "0" || labelStr == "1")
                {
                    validRows.Add(row);
                }
                else
                {
                    report.InvalidLabelRows++;
                }
            }
            rows = validRows;

            // ── 5. Compute column statistics for imputation ────────────────
            var medians = ComputeMedians(rows, NumericColumns);
            var modes = ComputeModes(rows, CategoricalColumns);

            // ── 6. Impute missing / non-parseable values ───────────────────
            foreach (var row in rows)
            {
                foreach (var col in NumericColumns)
                {
                    if (!TryParseFloat(row.GetValueOrDefault(col, ""), out _))
                    {
                        row[col] = medians[col].ToString(CultureInfo.InvariantCulture);
                        report.MissingValuesImputed++;
                    }
                }
                foreach (var col in CategoricalColumns)
                {
                    if (string.IsNullOrWhiteSpace(row.GetValueOrDefault(col, "")))
                    {
                        row[col] = modes[col];
                        report.MissingValuesImputed++;
                    }
                }
            }

            // Record imputation values for transparency
            foreach (var kvp in medians)
                report.ImputationValues[kvp.Key] = kvp.Value.ToString("F2", CultureInfo.InvariantCulture);
            foreach (var kvp in modes)
                report.ImputationValues[kvp.Key] = kvp.Value;

            // ── 7. Outlier capping (IQR × 1.5) for continuous columns ──────
            foreach (var col in ContinuousColumns)
            {
                var vals = rows.Select(r => ParseFloat(r[col])).OrderBy(v => v).ToArray();
                if (vals.Length < 4) continue;

                double q1 = Percentile(vals, 25);
                double q3 = Percentile(vals, 75);
                double iqr = q3 - q1;
                double lower = q1 - 1.5 * iqr;
                double upper = q3 + 1.5 * iqr;

                foreach (var row in rows)
                {
                    float v = ParseFloat(row[col]);
                    if (v < lower)
                    {
                        row[col] = lower.ToString(CultureInfo.InvariantCulture);
                        report.OutliersCapped++;
                    }
                    else if (v > upper)
                    {
                        row[col] = upper.ToString(CultureInfo.InvariantCulture);
                        report.OutliersCapped++;
                    }
                }
            }

            // ── 8. Normalise categorical text ──────────────────────────────
            foreach (var row in rows)
            {
                foreach (var col in CategoricalColumns)
                {
                    // Preserve original casing but trim whitespace
                    row[col] = row[col].Trim();
                }
            }

            report.CleanedRowCount = rows.Count;

            // ── 9. Write cleaned CSV to temp file ──────────────────────────
            string cleanedPath = Path.Combine(Path.GetTempPath(),
                $"itd_clean_{DateTime.UtcNow:yyyyMMddHHmmss}.csv");

            using (var writer = new StreamWriter(cleanedPath, false, Encoding.UTF8))
            {
                writer.WriteLine(string.Join(",", headers));
                foreach (var row in rows)
                    writer.WriteLine(string.Join(",", headers.Select(h => EscapeCsvField(row[h]))));
            }

            return (cleanedPath, report);
        }

        // -----------------------------------------------------------------------
        //  SINGLE ROW CLEANING (used by LoadSingleRowFromCsv)
        // -----------------------------------------------------------------------

        /// <summary>
        /// Cleans and validates a single parsed UserBehaviour record.
        /// Applies domain-knowledge bounds to flag obviously corrupt fields.
        /// </summary>
        public UserBehaviour CleanSingleRecord(UserBehaviour raw)
        {
            // Clamp binary flags to {0, 1}
            raw.is_contractor = Clamp01(raw.is_contractor);
            raw.has_foreign_citizenship = Clamp01(raw.has_foreign_citizenship);
            raw.has_criminal_record = Clamp01(raw.has_criminal_record);
            raw.has_medical_history = Clamp01(raw.has_medical_history);
            raw.is_abroad = Clamp01(raw.is_abroad);
            raw.late_exit_flag = Clamp01(raw.late_exit_flag);
            raw.entry_during_weekend = Clamp01(raw.entry_during_weekend);

            // Clamp classification to {0, 1, 2}
            raw.employee_classification = Math.Max(0, Math.Min(2, raw.employee_classification));

            // Non-negative guards
            raw.employee_seniority_years = Math.Max(0, raw.employee_seniority_years);
            raw.total_printed_pages = Math.Max(0, raw.total_printed_pages);
            raw.num_printed_pages_off_hours = Math.Max(0, raw.num_printed_pages_off_hours);
            raw.total_files_burned = Math.Max(0, raw.total_files_burned);
            raw.burned_from_other = Math.Max(0, raw.burned_from_other);
            raw.trip_day_number = Math.Max(0, raw.trip_day_number);
            raw.hostility_country_level = Math.Max(0, raw.hostility_country_level);
            raw.num_entries = Math.Max(0, raw.num_entries);
            raw.num_unique_campus = Math.Max(0, raw.num_unique_campus);

            // Logical consistency: off-hours pages ≤ total pages
            if (raw.num_printed_pages_off_hours > raw.total_printed_pages)
                raw.num_printed_pages_off_hours = raw.total_printed_pages;

            // burned_from_other ≤ total_files_burned
            if (raw.burned_from_other > raw.total_files_burned)
                raw.burned_from_other = raw.total_files_burned;

            // trip_day_number > 0 implies is_abroad
            if (raw.trip_day_number > 0 && raw.is_abroad == 0)
                raw.is_abroad = 1;

            // Categorical nulls
            if (string.IsNullOrWhiteSpace(raw.employee_department))
                raw.employee_department = "Unknown Department";
            if (string.IsNullOrWhiteSpace(raw.employee_campus))
                raw.employee_campus = "Unknown Campus";
            if (string.IsNullOrWhiteSpace(raw.employee_position))
                raw.employee_position = "Unknown Position";
            if (string.IsNullOrWhiteSpace(raw.employee_origin_country))
                raw.employee_origin_country = "Unknown";

            return raw;
        }

        // -----------------------------------------------------------------------
        //  PRIVATE HELPERS
        // -----------------------------------------------------------------------

        private void ValidateHeaders(string[] headers)
        {
            var missing = RequiredColumns
                .Where(c => !headers.Contains(c, StringComparer.OrdinalIgnoreCase))
                .ToList();
            if (missing.Any())
                throw new InvalidOperationException(
                    $"CSV is missing required columns: {string.Join(", ", missing)}");
        }

        private Dictionary<string, float> ComputeMedians(
            List<Dictionary<string, string>> rows, string[] cols)
        {
            var result = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
            foreach (var col in cols)
            {
                var vals = rows
                    .Select(r => r.GetValueOrDefault(col, ""))
                    .Where(s => TryParseFloat(s, out _))
                    .Select(s => ParseFloat(s))
                    .OrderBy(v => v)
                    .ToArray();

                result[col] = vals.Length == 0 ? 0f : vals[vals.Length / 2];
            }
            return result;
        }

        private Dictionary<string, string> ComputeModes(
            List<Dictionary<string, string>> rows, string[] cols)
        {
            var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var col in cols)
            {
                var mode = rows
                    .Select(r => r.GetValueOrDefault(col, "").Trim())
                    .Where(s => !string.IsNullOrEmpty(s))
                    .GroupBy(s => s)
                    .OrderByDescending(g => g.Count())
                    .Select(g => g.Key)
                    .FirstOrDefault() ?? "Unknown";
                result[col] = mode;
            }
            return result;
        }

        private static double Percentile(float[] sorted, double percentile)
        {
            if (sorted.Length == 0) return 0;
            double index = (percentile / 100.0) * (sorted.Length - 1);
            int lower = (int)Math.Floor(index);
            int upper = (int)Math.Ceiling(index);
            if (lower == upper) return sorted[lower];
            return sorted[lower] + (index - lower) * (sorted[upper] - sorted[lower]);
        }

        private static float Clamp01(float v) => Math.Max(0f, Math.Min(1f, v));

        private static bool TryParseFloat(string s, out float value) =>
            float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out value);

        private static float ParseFloat(string s) =>
            TryParseFloat(s, out float v) ? v : 0f;

        private static string[] SplitCsvLine(string line)
        {
            // Handle quoted fields (basic RFC-4180)
            var fields = new List<string>();
            bool inQuotes = false;
            var current = new StringBuilder();
            foreach (char c in line)
            {
                if (c == '"') { inQuotes = !inQuotes; }
                else if (c == ',' && !inQuotes) { fields.Add(current.ToString()); current.Clear(); }
                else { current.Append(c); }
            }
            fields.Add(current.ToString());
            return fields.ToArray();
        }

        private static string EscapeCsvField(string field)
        {
            if (field.Contains(',') || field.Contains('"') || field.Contains('\n'))
                return $"\"{field.Replace("\"", "\"\"")}\"";
            return field;
        }
    }
}