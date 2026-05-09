using Microsoft.ML.Data;

namespace InsiderThreatDetection.Core.Models
{
    public class UserBehaviour
    {
        // Column indices are based on the Kaggle dataset structure
        // Index 0 : employee_id (ignored)
        // Index 1 : date (ignored)
        // Index 2 : time (ignored)
        // Index 3 : employee_seniority_years
        // Index 4 : is_contractor
        // Index 5 : employee_classification
        // Index 6-9 : other ignored columns
        // Index 10 : total_printed_pages
        // ...

        [LoadColumn(3)]
        public float employee_seniority_years { get; set; }

        [LoadColumn(4)]
        public float is_contractor { get; set; }

        [LoadColumn(5)]
        public float employee_classification { get; set; }

        [LoadColumn(10)]
        public float total_printed_pages { get; set; }

        [LoadColumn(11)]
        public float num_printed_pages_off_hours { get; set; }

        [LoadColumn(12)]
        public float total_files_burned { get; set; }

        [LoadColumn(13)]
        public float burned_from_other { get; set; }

        [LoadColumn(14)]
        public float is_abroad { get; set; }

        [LoadColumn(15)]
        public float trip_day_number { get; set; }

        [LoadColumn(16)]
        public float hostility_country_level { get; set; }

        [LoadColumn(17)]
        public float num_entries { get; set; }

        [LoadColumn(18)]
        public float num_unique_campus { get; set; }

        [LoadColumn(19)]
        public float late_exit_flag { get; set; }

        [LoadColumn(20)]
        public float entry_during_weekend { get; set; }

        [LoadColumn(21)]
        public float is_malicious { get; set; }
    }
}