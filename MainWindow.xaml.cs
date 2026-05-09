using Microsoft.Win32;
using System;
using System.IO;
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

        // The index we genuinely selected (0 = "-- Select Profile --", 1 = Normal, etc.)
        private int _confirmedProfileIndex = -1;
        private bool _ignoreSelectionChange = false;

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
            try
            {
                ModelStatusText.Text = "Training, please wait...";
                await Task.Run(() => _controller.TrainModel(_lastDatasetPath));
                _modelReady = true;
                ModelStatusText.Text = "Model trained successfully ✔";

                string summary = _controller.GetEvaluationSummary();
                MessageBox.Show(summary, "Model Evaluation");
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
            if (_ignoreSelectionChange) return;                   // we are forcing a reset
            if (!_modelReady) return;

            int newIndex = SampleProfileComboBox.SelectedIndex;
            if (newIndex < 0) return;

            // Ignore the placeholder
            if (newIndex == 0)
            {
                if (_confirmedProfileIndex > 0)
                {
                    // instantly revert to the last confirmed real profile
                    _ignoreSelectionChange = true;
                    SampleProfileComboBox.SelectedIndex = _confirmedProfileIndex;
                    _ignoreSelectionChange = false;
                }
                return;
            }

            // We have a real profile.  Remember it.
            _confirmedProfileIndex = newIndex;

            ComboBoxItem item = (ComboBoxItem)SampleProfileComboBox.SelectedItem;
            string profile = item.Content.ToString();

            try
            {
                var input = _controller.GetProfile(profile);
                PredictAndExplain(input, profile);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
            finally
            {
                // After any message box, WPF WILL fire a stray SelectionChanged that
                // takes the dropdown back to the previous index.  We crush it by
                // re‑setting the correct index at a priority lower than any pending events.
                Dispatcher.BeginInvoke(new Action(() =>
                {
                    if (SampleProfileComboBox.SelectedIndex != _confirmedProfileIndex)
                    {
                        _ignoreSelectionChange = true;
                        SampleProfileComboBox.SelectedIndex = _confirmedProfileIndex;
                        _ignoreSelectionChange = false;
                    }
                }), System.Windows.Threading.DispatcherPriority.ApplicationIdle);
            }
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
                    PredictAndExplain(input, "CSV Record");
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