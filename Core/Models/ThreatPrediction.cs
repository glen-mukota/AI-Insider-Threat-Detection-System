// =============================================================================
//  ThreatPrediction.cs
//  Insider Threat Detection System – COS720 2026
// =============================================================================

using Microsoft.ML.Data;

namespace InsiderThreatDetection.Core.Models
{
    /// <summary>
    /// Output model produced by the ML.NET prediction pipeline.
    /// Contains raw ML scores plus derived security-relevant fields.
    /// </summary>
    public class ThreatPrediction
    {
        // ── ML.NET native outputs ──────────────────────────────────────────────
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel { get; set; }

        /// <summary>Raw probability (0–1) that the record is MALICIOUS.</summary>
        public float Probability { get; set; }

        /// <summary>Raw decision score from the boosted tree learner.</summary>
        public float Score { get; set; }

        // ── Derived / UI-facing fields ─────────────────────────────────────────

        /// <summary>"MALICIOUS" or "NORMAL".</summary>
        public string Classification { get; set; } = "NORMAL";

        /// <summary>
        /// Confidence in the stated Classification (0–1).
        /// = Probability  when Classification is MALICIOUS,
        /// = 1-Probability when Classification is NORMAL.
        /// </summary>
        public float Confidence { get; set; }

        /// <summary>Threat probability expressed as 0–1 (same as Probability).</summary>
        public float ThreatProbability { get; set; }

        /// <summary>
        /// Human-readable risk band driven by ThreatProbability and Classification.
        /// e.g. "LOW RISK", "HIGH RISK", "CRITICAL RISK".
        /// </summary>
        public string RiskLevel { get; set; } = string.Empty;
    }
}