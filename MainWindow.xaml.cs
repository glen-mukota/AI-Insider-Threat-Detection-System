using Microsoft.Win32;
using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using InsiderThreatDetection.ApplicationLayer;
using InsiderThreatDetection.Core.Models;

namespace InsiderThreatDetection
{
    public partial class MainWindow : Window
    {
        private readonly ThreatDetectionController _controller = new ThreatDetectionController();
        private string _lastDatasetPath = "insider_threat_clean_dataset.csv";
        private bool _modelReady = false;

        // Track the last valid profile we intentionally selected.
        // We use this to prevent the ComboBox from wandering after the pop-up closes.
        private string _lastValidProfile = null;

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
            if (!_modelReady) return;
            if (e.AddedItems == null || e.AddedItems.Count == 0) return;
            if (e.AddedItems[0] is not ComboBoxItem item || item.Content == null) return;

            string profile = item.Content.ToString()!;
            if (profile == "-- Select Profile --")
            {
                // If the user somehow selects the placeholder, don’t change anything
                // and immediately restore the last valid profile if we had one.
                if (_lastValidProfile != null)
                {
                    // Push the combo-box back to the last real profile after the message pump
                    Dispatcher.BeginInvoke(DispatcherPriority.Loaded, new Action(() =>
                    {
                        // Find the ComboBoxItem that matches the last valid profile
                        foreach (ComboBoxItem i in SampleProfileComboBox.Items)
                        {
                            if (i.Content?.ToString() == _lastValidProfile)
                            {
                                SampleProfileComboBox.SelectedItem = i;
                                break;
                            }
                        }
                    }));
                }
                return;
            }

            // We have a real profile
            _lastValidProfile = profile;
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
                // After the MessageBox, WPF might have sent another SelectionChanged
                // that could derail the dropdown. We forcefully re-apply the selection
                // at a low priority to override any stray events.
                string closedProfile = profile;   // capture for lambda
                Dispatcher.BeginInvoke(DispatcherPriority.Loaded, new Action(() =>
                {
                    foreach (ComboBoxItem it in SampleProfileComboBox.Items)
                    {
                        if (it.Content?.ToString() == closedProfile)
                        {
                            SampleProfileComboBox.SelectedItem = it;
                            break;
                        }
                    }
                }));
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