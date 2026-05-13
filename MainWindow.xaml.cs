using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Win32;
using InsiderThreatDetection.ApplicationLayer;
using InsiderThreatDetection.Core.Models;

namespace InsiderThreatDetection
{
    public partial class MainWindow : Window
    {
        private readonly ThreatDetectionController _controller = new ThreatDetectionController();
        private string _lastDatasetPath = "insider_threat_clean_dataset.csv";
        private bool _modelReady = false;

        public MainWindow()
        {
            InitializeComponent();
            SampleProfileComboBox.SelectedIndex = -1;
        }

        private async void UploadCsvButton_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog { Title = "Select Training Dataset CSV", Filter = "CSV Files (*.csv)|*.csv" };
            if (dlg.ShowDialog() == true)
            {
                _lastDatasetPath = dlg.FileName;
                await TrainModelAsync();
            }
        }

        private async void TrainModelButton_Click(object sender, RoutedEventArgs e)
        {
            if (!File.Exists(_lastDatasetPath))
            {
                MessageBox.Show("No dataset loaded. Upload a dataset first.", "Training Error");
                return;
            }
            await TrainModelAsync();
        }

        private async Task TrainModelAsync()
        {
            // Clear both display panels when retraining or uploading a new dataset
            PredictionResultText.Text = "Awaiting new prediction...";
            EvaluationSummaryText.Text = "Training, please wait...";

            try
            {
                ModelStatusText.Text = "Training, please wait...";
                await Task.Run(() => _controller.TrainModel(_lastDatasetPath));
                _modelReady = true;
                ModelStatusText.Text = "Model: FastTree (selected via F1 comparison). Trained successfully ✔";

                // Show evaluation summary inline (no dialog)
                string summary = _controller.GetEvaluationSummary();
                EvaluationSummaryText.Text = summary;
            }
            catch (Exception ex)
            {
                _modelReady = false;
                ModelStatusText.Text = "Training failed.";
                EvaluationSummaryText.Text = $"Training error: {ex.Message}";
            }
        }

        private void SaveModelButton_Click(object sender, RoutedEventArgs e)
        {
            if (!_modelReady) { MessageBox.Show("Train the model first."); return; }
            var dlg = new SaveFileDialog { Filter = "ML.NET Model|*.zip" };
            if (dlg.ShowDialog() == true)
            {
                _controller.SaveModel(dlg.FileName);
                MessageBox.Show("Model saved.");
            }
        }

        private void SampleProfileComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (!_modelReady || SampleProfileComboBox.SelectedIndex <= 0) return;

            var item = (ComboBoxItem)SampleProfileComboBox.SelectedItem;
            string profile = item.Content.ToString();
            try
            {
                var input = _controller.GetProfile(profile);
                PredictAndExplain(input, profile);
            }
            catch (Exception ex) { MessageBox.Show(ex.Message); }
        }

        private void PredictFromCsvButton_Click(object sender, RoutedEventArgs e)
        {
            if (!_modelReady) { MessageBox.Show("Model not trained. Please train first."); return; }

            // Clear the prediction result area before starting (evaluation remains)
            PredictionResultText.Text = "Running prediction...";

            var dlg = new OpenFileDialog { Title = "Select CSV with a single behavioural record", Filter = "CSV Files (*.csv)|*.csv" };
            if (dlg.ShowDialog() == true)
            {
                try
                {
                    var input = _controller.LoadSingleRowFromCsv(dlg.FileName, 1);
                    string featureSummary = GetFeatureSummary(input);
                    PredictAndExplain(input, "CSV Record\n\n" + featureSummary);
                }
                catch (Exception ex) { MessageBox.Show($"Error reading CSV: {ex.Message}"); }
            }
            else
            {
                // User cancelled file selection
                PredictionResultText.Text = "No file selected.";
            }
        }

        private string GetFeatureSummary(UserBehaviour user)
        {
            return "Uploaded record key features:\n" +
                $"  employee_seniority_years = {user.employee_seniority_years}\n" +
                $"  is_contractor = {user.is_contractor}\n" +
                $"  total_printed_pages = {user.total_printed_pages}\n" +
                $"  num_printed_pages_off_hours = {user.num_printed_pages_off_hours}\n" +
                $"  num_entries = {user.num_entries}\n" +
                $"  late_exit_flag = {user.late_exit_flag}\n" +
                $"  entry_during_weekend = {user.entry_during_weekend}";
        }

        /// <summary>
        /// Updates the prediction result panel and auto‑scrolls to make it visible.
        /// </summary>
        private void PredictAndExplain(UserBehaviour input, string sourceDescription)
        {
            try
            {
                var prediction = _controller.Predict(input);
                string classification = prediction.PredictedLabel ? "MALICIOUS" : "NORMAL";

                string riskLevel;
                if (classification == "NORMAL")
                {
                    riskLevel = prediction.Probability switch
                    {
                        < 0.30f => "LOW RISK",
                        < 0.50f => "ELEVATED — MONITOR",
                        _ => "BORDERLINE — REVIEW RECOMMENDED"
                    };
                }
                else
                {
                    riskLevel = prediction.Probability switch
                    {
                        < 0.70f => "HIGH RISK",
                        < 0.90f => "VERY HIGH RISK",
                        _ => "CRITICAL RISK"
                    };
                }

                float displayConfidence = prediction.PredictedLabel
                    ? prediction.Probability
                    : 1f - prediction.Probability;

                string humanExplanation = _controller.GenerateHumanExplanation(input, prediction);

                var contributions = _controller.Explain(input);
                float baseProb = _controller.GetBaselineProbability();
                string featureRanking = "\nFeature Importance (most → least influential):\n";
                if (!float.IsNaN(baseProb))
                {
                    featureRanking += $"  [Base threat probability (neutral profile): {baseProb:P2}]\n";
                }
                foreach (var c in contributions.Take(5))
                {
                    string dir = c.Contribution >= 0 ? "increased" : "decreased";
                    featureRanking += $"  - {c.Feature}: {c.Value} ({dir} risk by {Math.Abs(c.Contribution):F3})\n";
                }

                string result = $"** Profile tested: {sourceDescription} **\n\n" +
                                $"Classification: {classification}\n" +
                                $"Threat Probability: {prediction.Probability:P2}\n" +
                                $"Confidence in prediction: {displayConfidence:P2}\n" +
                                $"Risk level: {riskLevel}\n\n" +
                                $"Explanation:\n{humanExplanation}\n" +
                                featureRanking;

                PredictionResultText.Text = result;

                // Auto‑scroll so the full result is visible
                MainScrollViewer.ScrollToEnd();
            }
            catch (Exception ex)
            {
                PredictionResultText.Text = $"Prediction failed: {ex.Message}";
            }
        }
    }
}