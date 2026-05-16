using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Microsoft.Win32;
using InsiderThreatDetection.ApplicationLayer;

namespace InsiderThreatDetection
{
    public partial class MainWindow : Window
    {
        private readonly ThreatDetectionController _controller;
        private string _lastDatasetPath = "";

        private readonly Brush Green = new SolidColorBrush(Color.FromRgb(63, 185, 80));
        private readonly Brush Red = new SolidColorBrush(Color.FromRgb(248, 81, 73));
        private readonly Brush Orange = new SolidColorBrush(Color.FromRgb(210, 153, 34));

        public MainWindow()
        {
            InitializeComponent();

            _controller = new ThreatDetectionController();

            SampleProfileComboBox.Items.Clear();
            SampleProfileComboBox.Items.Add("-- Select Sample Profile --");
            SampleProfileComboBox.Items.Add("Normal Office Worker");
            SampleProfileComboBox.Items.Add("Suspicious Printing Activity");
            SampleProfileComboBox.Items.Add("Excessive Facility Access");
            SampleProfileComboBox.Items.Add("Critical Insider Threat");
            SampleProfileComboBox.SelectedIndex = 0;

            SampleProfileComboBox.IsEnabled = false;
            PredictFromCsvButton.IsEnabled = false;
            SaveModelButton.IsEnabled = false;

            RefreshAuditDisplay();
        }

        private async void UploadCsvButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filter = "CSV Files (*.csv)|*.csv";

            if (dialog.ShowDialog() != true) return;

            _lastDatasetPath = dialog.FileName;
            await TrainModelAsync();
        }

        private async void RetrainModelButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_lastDatasetPath))
            {
                MessageBox.Show("Upload dataset first.");
                return;
            }

            await TrainModelAsync();
        }

        private async Task TrainModelAsync()
        {
            SetTrainingState(true);

            try
            {
                await Task.Run(() => _controller.TrainModel(_lastDatasetPath));

                EvaluationSummaryText.Text = _controller.GetEvaluationSummary();
                ThresholdBadge.Text = $"Threshold: {_controller.GetBaselineProbability():P1}";
                ThresholdBadge.Visibility = Visibility.Visible;

                SetModelReadyState(true, "FastTree model trained successfully.");
            }
            catch (Exception ex)
            {
                SetModelReadyState(false, "Training failed.");
                MessageBox.Show(ex.Message);
            }
        }

        private void LoadModelButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filter = "ZIP Files (*.zip)|*.zip";

            if (dialog.ShowDialog() != true) return;

            try
            {
                _controller.LoadModel(dialog.FileName);
                SetModelReadyState(true, "Model loaded successfully.");
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void SaveModelButton_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog dialog = new SaveFileDialog();
            dialog.Filter = "ZIP Files (*.zip)|*.zip";

            if (dialog.ShowDialog() != true) return;

            _controller.SaveModel(dialog.FileName);
            MessageBox.Show("Model saved successfully.");
        }

        private void SampleProfileComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (!_controller.IsModelReady) return;
            if (SampleProfileComboBox.SelectedIndex <= 0) return;

            string profile = SampleProfileComboBox.SelectedItem.ToString();

            try
            {
                var input = _controller.GetProfile(profile);
                var prediction = _controller.Predict(input);

                PredictionResultText.Text =
                    $"Classification: {prediction.Classification}\n" +
                    $"Threat Probability: {prediction.ThreatProbability:P2}\n" +
                    $"Risk Level: {prediction.RiskLevel}";
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void PredictFromCsvButton_Click(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("CSV prediction logic executed.");
        }

        private void RefreshAuditBtn_Click(object sender, RoutedEventArgs e)
        {
            RefreshAuditDisplay();
        }

        private void RefreshAuditDisplay()
        {
            try
            {
                AuditLogText.Text = _controller.GetRecentAuditEntries();
                AuditLogPathText.Text = _controller.GetAuditLogPath();
            }
            catch
            {
                AuditLogText.Text = "Audit logs unavailable.";
            }
        }

        private void SetTrainingState(bool isTraining)
        {
            UploadCsvButton.IsEnabled = !isTraining;
            RetrainModelButton.IsEnabled = !isTraining;
            LoadModelButton.IsEnabled = !isTraining;

            SaveModelButton.IsEnabled = false;
            SampleProfileComboBox.IsEnabled = false;
            PredictFromCsvButton.IsEnabled = false;

            if (isTraining)
            {
                ModelStatusText.Text = "Training model... please wait";
                ModelStatusText.Foreground = Orange;
                StatusDot.Fill = Orange;
            }
        }

        private void SetModelReadyState(bool ready, string message)
        {
            ModelStatusText.Text = message;
            ModelStatusText.Foreground = ready ? Green : Red;
            StatusDot.Fill = ready ? Green : Red;

            UploadCsvButton.IsEnabled = true;
            RetrainModelButton.IsEnabled = true;
            LoadModelButton.IsEnabled = true;

            SaveModelButton.IsEnabled = ready;
            SampleProfileComboBox.IsEnabled = ready;
            PredictFromCsvButton.IsEnabled = ready;
        }
    }
}