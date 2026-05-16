// =============================================================================
//  DataPreprocessor.cs
//  Insider Threat Detection System – COS720 2026
//
//  CIA Triad:
//    Integrity   – only valid, clean data enters the model.
//    Availability – robust error handling prevents crashes on bad input.
//
//  Preprocessing steps:
//    1. Header validation
//    2. Duplicate row removal
//    3. Label validation (must be 0 or 1)
//    4. Missing-value imputation (median / mode)
//    5. Security-aware outlier handling on non-behavioural continuous columns
//    6. Binary flag clamping {0,1}
//    7. Logical-consistency enforcement
//    8. Categorical whitespace normalisation
//    9. Full preprocessing report for audit / display
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
    public class DataPreprocessor
    {
        // ── Column definitions ─────────────────────────────────────────────────
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
            "num_entries", "num_unique_campus", "late_exit_flag", "entry_during_weekend"
        };

        // Only cap non-security continuous values. Behavioural spikes such as
        // off-hours printing, file burning, and multi-campus access are threat
        // signals in this dataset, so they must not be flattened by global IQR.
        private static readonly string[] ContinuousColumns =
        {
            "employee_seniority_years"
        };

        private static readonly string[] CategoricalColumns =
        {
            "employee_department", "employee_campus",
            "employee_position", "employee_origin_country"
        };

        private static readonly string[] BinaryColumns =
        {
            "is_contractor", "has_foreign_citizenship", "has_criminal_record",
            "has_medical_history", "is_abroad", "late_exit_flag",
            "entry_during_weekend"
        };

        // ── Preprocessing report ───────────────────────────────────────────────
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

            /// <summary>Rich multi-line summary for display in the UI.</summary>
            public override string ToString()
            {
                var sb = new StringBuilder();
                sb.AppendLine("╔═══════════════════════════════════════╗");
                sb.AppendLine("║     DATA PREPROCESSING REPORT         ║");
                sb.AppendLine("╚═══════════════════════════════════════╝");
                sb.AppendLine($"  Original rows        : {OriginalRowCount:N0}");
                sb.AppendLine($"  Duplicates removed   : {DuplicatesRemoved:N0}");
                sb.AppendLine($"  Invalid label rows   : {InvalidLabelRows:N0}");
                sb.AppendLine($"  Missing values fixed : {MissingValuesImputed:N0}");
                sb.AppendLine($"  Outliers capped (IQR): {OutliersCapped:N0}");
                sb.AppendLine($"  ─────────────────────────────────────");
                sb.AppendLine($"  Final clean rows     : {CleanedRowCount:N0}");
                if (Warnings.Count > 0)
                {
                    sb.AppendLine();
                    sb.AppendLine("  ⚠ Warnings:");
                    foreach (var w in Warnings)
                        sb.AppendLine($"    • {w}");
                }
                return sb.ToString();
            }
        }

        // ── Public entry point ─────────────────────────────────────────────────

        /// <summary>
        /// Reads the raw CSV, applies all preprocessing steps, writes a temp
        /// cleaned CSV, and returns the path plus a detailed report.
        /// </summary>
        public (string cleanedPath, PreprocessingReport report) Preprocess(string inputPath)
        {
            var report = new PreprocessingReport();

            // 1. Read raw lines ─────────────────────────────────────────────────
            var rawLines = File.ReadAllLines(inputPath);
            if (rawLines.Length < 2)
                throw new InvalidOperationException("CSV file has no data rows.");

            var headers = rawLines[0].Split(',').Select(h => h.Trim()).ToArray();
            ValidateHeaders(headers);

            report.OriginalRowCount = rawLines.Length - 1;

            // 2. Parse rows into dictionaries ────────────────────────────────────
            var rows = new List<Dictionary<string, string>>();
            for (int i = 1; i < rawLines.Length; i++)
            {
                var values = SplitCsvLine(rawLines[i]);
                if (values.Length != headers.Length)
                {
                    report.Warnings.Add(
                        $"Row {i} skipped: expected {headers.Length} columns, got {values.Length}.");
                    continue;
                }
                var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                for (int j = 0; j < headers.Length; j++)
                    dict[headers[j]] = values[j].Trim();
                rows.Add(dict);
            }

            // 3. Remove duplicates ───────────────────────────────────────────────
            var unique = rows
                .GroupBy(r => string.Join("|", headers.Select(h => r[h])))
                .Select(g => g.First())
                .ToList();
            report.DuplicatesRemoved = rows.Count - unique.Count;
            rows = unique;

            // 4. Validate labels ─────────────────────────────────────────────────
            var validRows = new List<Dictionary<string, string>>();
            foreach (var row in rows)
            {
                string lbl = row.GetValueOrDefault("is_malicious", "").Trim();
                if (lbl == "0" || lbl == "1")
                    validRows.Add(row);
                else
                    report.InvalidLabelRows++;
            }
            rows = validRows;

            // 5. Compute imputation statistics ───────────────────────────────────
            var medians = ComputeMedians(rows, NumericColumns);
            var modes = ComputeModes(rows, CategoricalColumns);

            // 6. Impute missing / non-parseable values ───────────────────────────
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
                report.ImputationValues[kvp.Key] =
                    kvp.Value.ToString("F2", CultureInfo.InvariantCulture);
            foreach (var kvp in modes)
                report.ImputationValues[kvp.Key] = kvp.Value;

            // 7. Domain validation and logical consistency ──────────────────────
            foreach (var row in rows)
            {
                foreach (var col in BinaryColumns)
                    row[col] = Clamp01(ParseFloat(row[col]))
                        .ToString(CultureInfo.InvariantCulture);

                row["employee_classification"] =
                    Math.Max(1f, Math.Min(4f, ParseFloat(row["employee_classification"])))
                        .ToString(CultureInfo.InvariantCulture);

                foreach (var col in NumericColumns)
                    row[col] = Math.Max(0f, ParseFloat(row[col]))
                        .ToString(CultureInfo.InvariantCulture);

                float totalPrinted = ParseFloat(row["total_printed_pages"]);
                float offHoursPrinted = ParseFloat(row["num_printed_pages_off_hours"]);
                if (offHoursPrinted > totalPrinted)
                    row["num_printed_pages_off_hours"] =
                        totalPrinted.ToString(CultureInfo.InvariantCulture);

                float totalBurned = ParseFloat(row["total_files_burned"]);
                float burnedFromOther = ParseFloat(row["burned_from_other"]);
                if (burnedFromOther > totalBurned)
                    row["burned_from_other"] =
                        totalBurned.ToString(CultureInfo.InvariantCulture);

                if (ParseFloat(row["trip_day_number"]) > 0 && ParseFloat(row["is_abroad"]) == 0)
                    row["is_abroad"] = "1";
            }

            // 8. Outlier capping (IQR × 1.5) for non-security continuous columns ─
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

            // 9. Normalise categorical text (trim whitespace only; preserve case) ─
            foreach (var row in rows)
                foreach (var col in CategoricalColumns)
                    row[col] = row[col].Trim();

            report.CleanedRowCount = rows.Count;

            // 10. Write cleaned CSV to temp file ─────────────────────────────────
            string cleanedPath = Path.Combine(Path.GetTempPath(),
                $"itd_clean_{DateTime.UtcNow:yyyyMMddHHmmss}.csv");

            using (var writer = new StreamWriter(cleanedPath, false, Encoding.UTF8))
            {
                writer.WriteLine(string.Join(",", headers));
                foreach (var row in rows)
                    writer.WriteLine(string.Join(",",
                        headers.Select(h => EscapeCsvField(row[h]))));
            }

            return (cleanedPath, report);
        }

        // ── Single-record cleaning (used for CSV upload & sample profiles) ─────

        /// <summary>
        /// Cleans and validates a single <see cref="UserBehaviour"/> record.
        /// Applies domain-knowledge bounds and logical-consistency rules.
        /// </summary>
        public UserBehaviour CleanSingleRecord(UserBehaviour raw)
        {
            // Binary flag clamping
            raw.is_contractor = Clamp01(raw.is_contractor);
            raw.has_foreign_citizenship = Clamp01(raw.has_foreign_citizenship);
            raw.has_criminal_record = Clamp01(raw.has_criminal_record);
            raw.has_medical_history = Clamp01(raw.has_medical_history);
            raw.is_abroad = Clamp01(raw.is_abroad);
            raw.late_exit_flag = Clamp01(raw.late_exit_flag);
            raw.entry_during_weekend = Clamp01(raw.entry_during_weekend);

            // Classification level in the prescribed dataset is 1..4.
            raw.employee_classification =
                Math.Max(1, Math.Min(4, raw.employee_classification));

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

            // Logical consistency
            if (raw.num_printed_pages_off_hours > raw.total_printed_pages)
                raw.num_printed_pages_off_hours = raw.total_printed_pages;

            if (raw.burned_from_other > raw.total_files_burned)
                raw.burned_from_other = raw.total_files_burned;

            if (raw.trip_day_number > 0 && raw.is_abroad == 0)
                raw.is_abroad = 1;

            // Categorical null-guards
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

        // ── Private helpers ────────────────────────────────────────────────────

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

        private static double Percentile(float[] sorted, double pct)
        {
            if (sorted.Length == 0) return 0;
            double idx = (pct / 100.0) * (sorted.Length - 1);
            int lower = (int)Math.Floor(idx);
            int upper = (int)Math.Ceiling(idx);
            return lower == upper
                ? sorted[lower]
                : sorted[lower] + (idx - lower) * (sorted[upper] - sorted[lower]);
        }

        private static float Clamp01(float v) => Math.Max(0f, Math.Min(1f, v));

        private static bool TryParseFloat(string s, out float value) =>
            float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out value);

        private static float ParseFloat(string s) =>
            TryParseFloat(s, out float v) ? v : 0f;

        private static string[] SplitCsvLine(string line)
        {
            var fields = new List<string>();
            bool inQ = false;
            var cur = new StringBuilder();
            foreach (char c in line)
            {
                if (c == '"') { inQ = !inQ; }
                else if (c == ',' && !inQ) { fields.Add(cur.ToString()); cur.Clear(); }
                else { cur.Append(c); }
            }
            fields.Add(cur.ToString());
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
