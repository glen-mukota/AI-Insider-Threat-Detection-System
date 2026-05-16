using Microsoft.ML.Data;

namespace InsiderThreatDetection.Core.Models
{
    public class ThreatPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel { get; set; }

        public float Probability { get; set; }      // raw malicious probability
        public float Score { get; set; }

        // New fields for proper confidence display and classification output
        public string Classification { get; set; }   // "MALICIOUS" or "NORMAL"
        public float Confidence { get; set; }         // 0..1 = certainty in Classification
        public float ThreatProbability { get; set; }  // 0..1 = probability of malicious
    }
}