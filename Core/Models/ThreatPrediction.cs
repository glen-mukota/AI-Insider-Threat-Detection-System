using Microsoft.ML.Data;

namespace InsiderThreatDetection.Core.Models
{
    public class ThreatPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel { get; set; }

        public float Probability { get; set; }
        public float Score { get; set; }
    }
}