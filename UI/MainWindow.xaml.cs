// =============================================================================
//  MainWindow.xaml.cs
//  Insider Threat Detection System – COS720 2026
//
//  Fully implemented code-behind for the production-ready WPF dashboard.
//  All button handlers, prediction display, feature importance bars, and
//  preprocessing report display are correctly wired.
//
//  CIA Triad:
//    Availability    – async training keeps UI fully responsive.
//    Integrity        – user sees exact preprocessing report after training.
//    Confidentiality – no data transmitted externally; results shown locally.
// =============================================================================

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Microsoft.Win32;
using InsiderThreatDetection.ApplicationLayer;
using InsiderThreatDetection.Core.Models;
using InsiderThreatDetection.Infrastructure;

namespace InsiderThreatDetection
{
    public partial class MainWindow : Window
    {
        // ── Dependencies ───────────────────────────────────────────────────────
        private readonly ThreatDetectionController _controller;
        private string _lastDatasetPath = string.Empty;

        // ── Colour palette ─────────────────────────────────────────────────────
        private static readonly Brush BrushGreen = new SolidColorBrush(Color.FromRgb(63, 185, 80));
        private static readonly Brush BrushRed = new SolidColorBrush(Color.FromRgb(248, 81, 73));
        private static readonly Brush BrushOrange = new SolidColorBrush(Color.FromRgb(210, 153, 34));
        private static readonly Brush BrushBlue = new SolidColorBrush(Color.FromRgb(31, 111, 235));
        private static readonly Brush BrushGray = new SolidColorBrush(Color.FromRgb(33, 38, 45));

        // ── Constructor ────────────────────────────────────────────────────────
        public MainWindow()
        {
            InitializeComponent();
            _controller = new ThreatDetectionController();

            // Populate sample profiles
            SampleProfileComboBox.Items.Add("-- Select Sample Profile --");
            SampleProfileComboBox.Items.Add("Normal Office Worker");
            SampleProfileComboBox.Items.Add("Suspicious Printing Activity");
            SampleProfileComboBox.Items.Add("Excessive Facility Access");
            SampleProfileComboBox.Items.Add("Critical Insider Threat");
            SampleProfileComboBox.SelectedIndex = 0;

            // Disable prediction controls until model is ready
            SampleProfileComboBox.IsEnabled = false;
            PredictFromCsvButton.IsEnabled = false;
            SaveModelButton.IsEnabled = false;

            RefreshAuditDisplay();
        }

        // ══════════════════════════════════════════════════════════════════════
        //  NAV BUTTON HANDLERS
        // ══════════════════════════════════════════════════════════════════════

        private void NavDashboard_Click(object sender, RoutedEventArgs e)
        {
            SetActiveNav(NavDashboard);
            DashboardSection.BringIntoView();
        }

        private void NavTraining_Click(object sender, RoutedEventArgs e)
        {
            SetActiveNav(NavTraining);
            TrainingSection.BringIntoView();
        }

        private void NavPrediction_Click(object sender, RoutedEventArgs e)
        {
            SetActiveNav(NavPrediction);
            PredictionSection.BringIntoView();
        }

        private void NavAudit_Click(object sender, RoutedEventArgs e)
        {
            SetActiveNav(NavAudit);
            RefreshAuditDisplay();
            AuditSection.BringIntoView();
        }

        private void SetActiveNav(Button active)
        {
            var navButtons = new[] { NavDashboard, NavTraining, NavPrediction, NavAudit };
            foreach (var btn in navButtons)
            {
                btn.Background = Brushes.Transparent;
                btn.Foreground = new SolidColorBrush(Color.FromRgb(139, 148, 158));
            }
            active.Background = new SolidColorBrush(Color.FromRgb(33, 38, 45));
            active.Foreground = new SolidColorBrush(Color.FromRgb(230, 237, 243));
        }

        // ══════════════════════════════════════════════════════════════════════
        //  TRAINING HANDLERS
        // ══════════════════════════════════════════════════════════════════════

        private async void UploadCsvButton_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog
            {
                Title = "Select Insider Threat Dataset (CSV)",
                Filter = "CSV Files (*.csv)|*.csv|All Files (*.*)|*.*"
            };

            if (dialog.ShowDialog() != true) return;

            _lastDatasetPath = dialog.FileName;
            await TrainModelAsync();
        }

        private async void RetrainModelButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_lastDatasetPath))
            {
                MessageBox.Show(
                    "No dataset has been uploaded yet.\n\nPlease click 'Upload Dataset & Train' first.",
                    "No Dataset", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }
            await TrainModelAsync();
        }

        private async Task TrainModelAsync()
        {
            SetTrainingState(true);

            try
            {
                Core.Services.DataPreprocessor.PreprocessingReport report = null!;

                await Task.Run(() =>
                {
                    report = _controller.TrainModel(_lastDatasetPath);
                });

                // ── Show preprocessing report ───────────────────────────────
                PreprocessingReportText.Text = report.ToString();
                PreprocessingCard.Visibility = Visibility.Visible;

                // ── Brief preprocessing summary in Model Management panel ───
                PreprocessSummary.Text =
                    $"✅ Preprocessing complete\n" +
                    $"   {report.OriginalRowCount:N0} rows → {report.CleanedRowCount:N0} clean\n" +
                    $"   {report.DuplicatesRemoved:N0} duplicates removed\n" +
                    $"   {report.OutliersCapped:N0} non-security outliers capped";
                PreprocessPanel.Visibility = Visibility.Visible;

                // ── Show evaluation summary ────────────────────────────────
                EvaluationSummaryText.Text = _controller.GetEvaluationSummary();

                // ── Show threshold info ────────────────────────────────────
                float threshold = _controller.GetCalibratedThreshold();
                float baseline = _controller.GetBaselineProbability();

                ThresholdBadge.Text =
                    $"🎯 Decision Threshold: {threshold:P0}  " +
                    $"(records ≥ {threshold:P0} → MALICIOUS)";
                BaselineBadge.Text =
                    float.IsNaN(baseline)
                    ? "Baseline probability: N/A"
                    : $"Benign baseline probability: {baseline:P1}";
                ThresholdPanel.Visibility = Visibility.Visible;

                SetModelReadyState(true,
                    $"✅ FastTree model trained successfully — " +
                    $"threshold: {threshold:P0}");
            }
            catch (Exception ex)
            {
                SetModelReadyState(false, "❌ Training failed.");
                MessageBox.Show(
                    $"Training error:\n\n{ex.Message}",
                    "Training Failed", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        // ══════════════════════════════════════════════════════════════════════
        //  MODEL MANAGEMENT HANDLERS
        // ══════════════════════════════════════════════════════════════════════

        private void LoadModelButton_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog
            {
                Title = "Load Trained ML.NET Model",
                Filter = "ML.NET Model (*.zip)|*.zip|All Files (*.*)|*.*"
            };

            if (dialog.ShowDialog() != true) return;

            try
            {
                _controller.LoadModel(dialog.FileName);
                EvaluationSummaryText.Text =
                    "ℹ  Model loaded from file.\n\n" +
                    _controller.GetEvaluationSummary();

                float threshold = _controller.GetCalibratedThreshold();
                float baseline = _controller.GetBaselineProbability();

                ThresholdBadge.Text =
                    $"🎯 Decision Threshold: {threshold:P0}  " +
                    $"(records ≥ {threshold:P0} → MALICIOUS)";
                BaselineBadge.Text =
                    float.IsNaN(baseline)
                    ? "Baseline probability: N/A — load the matching .metadata.json file for calibrated baselines."
                    : $"Benign baseline probability: {baseline:P1}";
                ThresholdPanel.Visibility = Visibility.Visible;

                SetModelReadyState(true,
                    $"✅ Model loaded: {Path.GetFileName(dialog.FileName)}");
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"Failed to load model:\n\n{ex.Message}",
                    "Load Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void SaveModelButton_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new SaveFileDialog
            {
                Title = "Save Trained Model",
                Filter = "ML.NET Model (*.zip)|*.zip",
                FileName = $"insider_threat_model_{DateTime.Now:yyyyMMdd_HHmm}.zip"
            };

            if (dialog.ShowDialog() != true) return;

            try
            {
                _controller.SaveModel(dialog.FileName);
                MessageBox.Show(
                    $"Model saved successfully to:\n{dialog.FileName}",
                    "Model Saved", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"Failed to save model:\n\n{ex.Message}",
                    "Save Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        // ══════════════════════════════════════════════════════════════════════
        //  PREDICTION HANDLERS
        // ══════════════════════════════════════════════════════════════════════

        private void SampleProfileComboBox_SelectionChanged(object sender,
                                                             SelectionChangedEventArgs e)
        {
            if (!_controller.IsModelReady) return;
            if (SampleProfileComboBox.SelectedIndex <= 0) return;

            string profileName = SampleProfileComboBox.SelectedItem?.ToString() ?? "";

            try
            {
                var input = _controller.GetProfile(profileName);
                var prediction = _controller.Predict(input);
                var explains = _controller.Explain(input);
                var humanExp = _controller.GenerateHumanExplanation(input, prediction);

                DisplayPredictionResult(prediction, explains, humanExp,
                    $"Profile: {profileName}");
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"Prediction error:\n\n{ex.Message}",
                    "Prediction Error", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }

        private void PredictFromCsvButton_Click(object sender, RoutedEventArgs e)
        {
            if (!_controller.IsModelReady)
            {
                MessageBox.Show("Please train the model first.",
                    "Model Not Ready", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var dialog = new OpenFileDialog
            {
                Title = "Select CSV File for Prediction",
                Filter = "CSV Files (*.csv)|*.csv|All Files (*.*)|*.*"
            };

            if (dialog.ShowDialog() != true) return;

            try
            {
                var results = _controller.PredictAllRowsFromCsv(dialog.FileName, maxRows: 50);
                if (results.Count == 0)
                {
                    MessageBox.Show(
                        "No valid records were found in the selected CSV.",
                        "No Predictions", MessageBoxButton.OK, MessageBoxImage.Information);
                    return;
                }

                var displayRows = results
                    .Select(r => new CsvPredictionRow
                    {
                        Row = r.Row,
                        Classification = r.Prediction.Classification,
                        Probability = $"{r.Prediction.ThreatProbability:P1}",
                        Confidence = $"{r.Prediction.Confidence:P1}",
                        RiskLevel = r.Prediction.RiskLevel
                    })
                    .ToList();

                CsvResultsGrid.ItemsSource = displayRows;

                int maliciousCount = results.Count(r => r.Prediction.PredictedLabel);
                var highestRisk = results
                    .OrderByDescending(r => r.Prediction.ThreatProbability)
                    .First();

                CsvResultsSummaryText.Text =
                    $"Analysed {results.Count:N0} row(s) from {Path.GetFileName(dialog.FileName)}. " +
                    $"{maliciousCount:N0} row(s) crossed the calibrated malicious threshold. " +
                    $"Highest risk: row {highestRisk.Row} at {highestRisk.Prediction.ThreatProbability:P1}.";
                CsvResultsCard.Visibility = Visibility.Visible;

                var explains = _controller.Explain(highestRisk.Input);
                var humanExp = _controller.GenerateHumanExplanation(
                    highestRisk.Input, highestRisk.Prediction);

                DisplayPredictionResult(highestRisk.Prediction, explains, humanExp,
                    $"CSV batch: {Path.GetFileName(dialog.FileName)} — row {highestRisk.Row}");
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"CSV prediction error:\n\n{ex.Message}",
                    "Prediction Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        // ══════════════════════════════════════════════════════════════════════
        //  DISPLAY HELPERS
        // ══════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Updates all result panels (probability bars, classification badge,
        /// risk badge, explanation text, feature importance bars) from a
        /// completed prediction.
        /// </summary>
        private void DisplayPredictionResult(
            ThreatPrediction prediction,
            List<(string Feature, float Contribution, float Value, float BenignMean)> explains,
            string humanExplanation,
            string sourceLabel)
        {
            bool isMal = prediction.Classification == "MALICIOUS";
            PredictionSourceText.Text = sourceLabel;

            // ── Colour palette for this result ─────────────────────────────
            Brush primaryColour = isMal
                ? new SolidColorBrush(Color.FromRgb(248, 81, 73))   // red
                : new SolidColorBrush(Color.FromRgb(63, 185, 80));  // green

            // Resolve parent widths after layout so the visual bars scale correctly.
            Dispatcher.InvokeAsync(() =>
            {
                double parentWidth = ((Border)ProbabilityFill.Parent).ActualWidth;
                ProbabilityFill.Width = Math.Max(2, parentWidth * prediction.ThreatProbability);
                ProbabilityFill.Background = primaryColour;

                double confParent = ((Border)ConfidenceFill.Parent).ActualWidth;
                ConfidenceFill.Width = Math.Max(2, confParent * prediction.Confidence);
            }, System.Windows.Threading.DispatcherPriority.Loaded);

            ProbabilityPct.Text = $"{prediction.ThreatProbability:P0}";
            ConfidencePct.Text = $"{prediction.Confidence:P0}";

            // ── Classification badge ────────────────────────────────────────
            ClassificationText.Text = isMal ? "⚠  MALICIOUS" : "✔  NORMAL";
            ClassificationText.Foreground = primaryColour;
            ClassificationBadge.Background =
                isMal
                ? new SolidColorBrush(Color.FromArgb(60, 248, 81, 73))
                : new SolidColorBrush(Color.FromArgb(60, 63, 185, 80));

            // ── Risk level badge ────────────────────────────────────────────
            RiskLevelText.Text = prediction.RiskLevel;
            RiskLevelText.Foreground = GetRiskBrush(prediction.RiskLevel);

            // ── Risk badge in Model Management panel ────────────────────────
            RiskBadge.Background = isMal
                ? new SolidColorBrush(Color.FromRgb(88, 28, 28))
                : new SolidColorBrush(Color.FromRgb(20, 60, 35));
            RiskBadge.Visibility = Visibility.Visible;
            RiskBadgeText.Text = isMal ? "⚠  MALICIOUS" : "✔  NORMAL";
            RiskBadgeProb.Text =
                $"P={prediction.ThreatProbability:P1}  |  {prediction.RiskLevel}";

            // ── Explanation text ────────────────────────────────────────────
            PredictionExplanationText.Text = humanExplanation;
            PredictionExplanationText.Foreground =
                new SolidColorBrush(Color.FromRgb(201, 209, 217));

            // ── Feature importance bars ─────────────────────────────────────
            BuildFeatureBars(explains);
            FeatureImportanceCard.Visibility = Visibility.Visible;
            PredictionSection.BringIntoView();
        }

        /// <summary>Builds the feature importance bar items for the ItemsControl.</summary>
        private void BuildFeatureBars(
            List<(string Feature, float Contribution, float Value, float BenignMean)> explains)
        {
            var items = new List<FeatureBarItem>();
            double maxAbs = 0;
            foreach (var e in explains)
                if (Math.Abs(e.Contribution) > maxAbs)
                    maxAbs = Math.Abs(e.Contribution);

            if (maxAbs == 0) maxAbs = 1;
            // Keep bars inside the explanation card at the minimum supported window width.
            const double barMax = 150.0;

            foreach (var e in explains)
            {
                double width = Math.Max(2, (Math.Abs(e.Contribution) / maxAbs) * barMax);
                bool raisesRisk = e.Contribution > 0;

                items.Add(new FeatureBarItem
                {
                    Label = MLModelManager.HumanReadableName(e.Feature),
                    BarWidth = width,
                    BarColour = raisesRisk
                                  ? new SolidColorBrush(Color.FromRgb(248, 81, 73))
                                  : new SolidColorBrush(Color.FromRgb(63, 185, 80)),
                    TextColour = raisesRisk
                                  ? new SolidColorBrush(Color.FromRgb(248, 81, 73))
                                  : new SolidColorBrush(Color.FromRgb(63, 185, 80)),
                    ContribText = (raisesRisk ? "▲ +" : "▼ ") +
                                  $"{e.Contribution:P1}",
                    ValueText = $"{e.Value:F1} [{e.BenignMean:F1}]"
                });
            }

            FeatureBars.ItemsSource = items;
        }

        private static Brush GetRiskBrush(string riskLevel) => riskLevel switch
        {
            "CRITICAL RISK" => new SolidColorBrush(Color.FromRgb(248, 81, 73)),
            "VERY HIGH RISK" => new SolidColorBrush(Color.FromRgb(210, 100, 50)),
            "HIGH RISK" => new SolidColorBrush(Color.FromRgb(210, 153, 34)),
            "BORDERLINE – REVIEW RECOMMENDED"
                                          => new SolidColorBrush(Color.FromRgb(139, 148, 158)),
            "ELEVATED – MONITOR" => new SolidColorBrush(Color.FromRgb(88, 166, 255)),
            "LOW RISK" => new SolidColorBrush(Color.FromRgb(63, 185, 80)),
            _ => new SolidColorBrush(Color.FromRgb(139, 148, 158))
        };

        // ══════════════════════════════════════════════════════════════════════
        //  STATE HELPERS
        // ══════════════════════════════════════════════════════════════════════

        private void SetTrainingState(bool isTraining)
        {
            UploadCsvButton.IsEnabled = !isTraining;
            RetrainModelButton.IsEnabled = !isTraining;
            LoadModelButton.IsEnabled = !isTraining;
            SaveModelButton.IsEnabled = false;
            SampleProfileComboBox.IsEnabled = false;
            PredictFromCsvButton.IsEnabled = false;

            TrainingProgressPanel.Visibility =
                isTraining ? Visibility.Visible : Visibility.Collapsed;

            if (isTraining)
            {
                TrainingProgressText.Text =
                    "⏳ Preprocessing dataset and training FastTree model…\n" +
                    "This may take 30–90 seconds for large datasets. Please wait.";
                ModelStatusText.Text = "Training model… please wait";
                ModelStatusText.Foreground = BrushOrange;
                StatusDot.Fill = BrushOrange;
            }
        }

        private void SetModelReadyState(bool ready, string message)
        {
            ModelStatusText.Text = message;
            ModelStatusText.Foreground = ready ? BrushGreen : BrushRed;
            StatusDot.Fill = ready ? BrushGreen : BrushRed;

            UploadCsvButton.IsEnabled = true;
            RetrainModelButton.IsEnabled = true;
            LoadModelButton.IsEnabled = true;
            SaveModelButton.IsEnabled = ready;
            SampleProfileComboBox.IsEnabled = ready;
            PredictFromCsvButton.IsEnabled = ready;

            TrainingProgressPanel.Visibility = Visibility.Collapsed;

            RefreshAuditDisplay();
        }

        // ══════════════════════════════════════════════════════════════════════
        //  AUDIT LOG
        // ══════════════════════════════════════════════════════════════════════

        private void RefreshAuditBtn_Click(object sender, RoutedEventArgs e)
        {
            RefreshAuditDisplay();
        }

        private void RefreshAuditDisplay()
        {
            try
            {
                AuditLogText.Text = _controller.GetRecentAuditEntries();
                AuditLogPathText.Text = $"Log file: {_controller.GetAuditLogPath()}";

                // Scroll to bottom
                AuditLogText.ScrollToEnd();
            }
            catch
            {
                AuditLogText.Text = "Audit logs unavailable.";
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  DATA BINDING HELPER
    // ══════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// View model for a single row in the feature importance bar chart.
    /// </summary>
    public class FeatureBarItem
    {
        public string Label { get; set; } = string.Empty;
        public double BarWidth { get; set; }
        public Brush BarColour { get; set; } = Brushes.Gray;
        public Brush TextColour { get; set; } = Brushes.Gray;
        public string ContribText { get; set; } = string.Empty;
        public string ValueText { get; set; } = string.Empty;
    }

    public class CsvPredictionRow
    {
        public int Row { get; set; }
        public string Classification { get; set; } = string.Empty;
        public string Probability { get; set; } = string.Empty;
        public string Confidence { get; set; } = string.Empty;
        public string RiskLevel { get; set; } = string.Empty;
    }
}
