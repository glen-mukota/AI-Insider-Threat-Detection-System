// =============================================================================
//  UserBehaviour.cs
//  Insider Threat Detection System – COS720 2026
// =============================================================================

using Microsoft.ML.Data;

namespace InsiderThreatDetection.Core.Models
{
    /// <summary>
    /// Represents a single employee behavioural record used for training and
    /// inference. Column indices correspond to the Kaggle dataset column order.
    ///
    /// Behavioural indicators covered (per project specification):
    ///   • Login / facility-access patterns  (num_entries, late_exit_flag, entry_during_weekend)
    ///   • File access / data-transfer       (total_files_burned, burned_from_other)
    ///   • Privilege / classification usage  (employee_classification, is_contractor)
    ///   • Abnormal access frequency         (num_unique_campus)
    ///   • Data transfer (printing)          (total_printed_pages, num_printed_pages_off_hours)
    ///   • Travel / geopolitical risk        (is_abroad, trip_day_number, hostility_country_level)
    ///   • Background factors                (has_foreign_citizenship, has_criminal_record,
    ///                                        has_medical_history)
    /// </summary>
    public class UserBehaviour
    {
        // ── Categorical features ───────────────────────────────────────────────
        [LoadColumn(0)] public string employee_department { get; set; } = string.Empty;
        [LoadColumn(1)] public string employee_campus { get; set; } = string.Empty;
        [LoadColumn(2)] public string employee_position { get; set; } = string.Empty;
        [LoadColumn(9)] public string employee_origin_country { get; set; } = string.Empty;

        // ── Numeric features ───────────────────────────────────────────────────
        [LoadColumn(3)] public float employee_seniority_years { get; set; }
        [LoadColumn(4)] public float is_contractor { get; set; }
        [LoadColumn(5)] public float employee_classification { get; set; }
        [LoadColumn(6)] public float has_foreign_citizenship { get; set; }
        [LoadColumn(7)] public float has_criminal_record { get; set; }
        [LoadColumn(8)] public float has_medical_history { get; set; }
        [LoadColumn(10)] public float total_printed_pages { get; set; }
        [LoadColumn(11)] public float num_printed_pages_off_hours { get; set; }
        [LoadColumn(12)] public float total_files_burned { get; set; }
        [LoadColumn(13)] public float burned_from_other { get; set; }
        [LoadColumn(14)] public float is_abroad { get; set; }
        [LoadColumn(15)] public float trip_day_number { get; set; }
        [LoadColumn(16)] public float hostility_country_level { get; set; }
        [LoadColumn(17)] public float num_entries { get; set; }
        [LoadColumn(18)] public float num_unique_campus { get; set; }
        [LoadColumn(19)] public float late_exit_flag { get; set; }
        [LoadColumn(20)] public float entry_during_weekend { get; set; }

        // ── Label ──────────────────────────────────────────────────────────────
        [LoadColumn(21)] public float is_malicious { get; set; }
    }
}