using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
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
            SampleProfileComboBox.SelectedIndex = -1;   // no selection at startup
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
            try
            {
                ModelStatusText.Text = "Training, please wait...";
                await Task.Run(() => _controller.TrainModel(_lastDatasetPath));
                _modelReady = true;
                ModelStatusText.Text = "Model trained successfully ✔";

                var metrics = _controller.GetMetrics();
                if (metrics != null)
                {
                    var cm = _controller.GetConfusionMatrixCounts();
                    string cmText = "";
                    if (cm != null && cm.Length == 2 && cm[0].Length == 2 && cm[1].Length == 2)
                    {
                        double tn = cm[0][0];
                        double fp = cm[0][1];
                        double fn = cm[1][0];
                        double tp = cm[1][1];
                        cmText = $"\nConfusion Matrix:\n" +
                                 $"  True Positives  : {tp}\n" +
                                 $"  True Negatives  : {tn}\n" +
                                 $"  False Positives : {fp}\n" +
                                 $"  False Negatives : {fn}";
                    }

                    string msg = $"Accuracy: {metrics.Accuracy:P2}\n" +
                                 $"Precision: {metrics.PositivePrecision:P2}\n" +
                                 $"Recall: {metrics.PositiveRecall:P2}\n" +
                                 $"F1: {metrics.F1Score:P2}" +
                                 cmText;

                    MessageBox.Show(msg, "Model Evaluation");
                }
            }
            catch (Exception ex)
            {
                _modelReady = false;
                ModelStatusText.Text = "Training failed.";
                MessageBox.Show(ex.Message, "Error");
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
            if (!_modelReady) return;
            if (SampleProfileComboBox.SelectedItem is not ComboBoxItem item || item.Content == null) return;
            string profile = item.Content.ToString()!;
            if (profile == "-- Select Profile --") return;

            try
            {
                var input = _controller.GetProfile(profile);
                PredictAndExplain(input, profile);   // pass the profile name
            }
            catch (Exception ex) { MessageBox.Show(ex.Message); }
        }

        private void PredictFromCsvButton_Click(object sender, RoutedEventArgs e)
        {
            if (!_modelReady) { MessageBox.Show("Model not trained. Please train first."); return; }
            var dlg = new OpenFileDialog { Title = "Select CSV with a single behavioural record", Filter = "CSV Files (*.csv)|*.csv" };
            if (dlg.ShowDialog() == true)
            {
                try
                {
                    var input = _controller.LoadSingleRowFromCsv(dlg.FileName, rowIndex: 1);
                    PredictAndExplain(input, "CSV Record");   // indicate CSV source
                }
                catch (Exception ex) { MessageBox.Show($"Error reading CSV: {ex.Message}"); }
            }
        }

        private void PredictAndExplain(UserBehaviour input, string sourceDescription)
        {
            try
            {
                var prediction = _controller.Predict(input);
                string riskLevel = prediction.Probability switch
                {
                    >= 0.9f => "CRITICAL RISK",
                    >= 0.75f => "HIGH RISK",
                    >= 0.5f => "MEDIUM RISK",
                    _ => "LOW RISK"
                };
                string classification = prediction.PredictedLabel ? "MALICIOUS" : "NORMAL";

                string humanExplanation = _controller.GenerateHumanExplanation(input, prediction);

                var contributions = _controller.Explain(input);
                string featureRanking = "\nFeature Importance Ranking (most → least influential):\n";
                foreach (var c in contributions.Take(5))
                {
                    string direction = c.Contribution >= 0 ? "increased" : "decreased";
                    featureRanking += $"  - {c.Feature}: {c.Value} ({direction} risk by {Math.Abs(c.Contribution):F3})\n";
                }

                // ---- Build the final message, including the source profile name ----
                string message = $"** Profile tested: {sourceDescription} **\n\n" +
                                 $"Classification: {classification}\n" +
                                 $"Confidence: {prediction.Probability:P2}\n" +
                                 $"Risk level: {riskLevel}\n\n" +
                                 $"Explanation:\n{humanExplanation}\n" +
                                 featureRanking;

                MessageBox.Show(message, "Prediction Result");
            }
            catch (Exception ex) { MessageBox.Show($"Prediction failed: {ex.Message}"); }
        }
    }
}