// =============================================================================
//  MainWindow.xaml.cs
//  Insider Threat Detection System – COS720 2026
//
//  Code-behind for the WPF main window.
//  Follows the MVC/layered pattern: UI calls ThreatDetectionController,
//  never MLModelManager or DataPreprocessor directly.
//
//  CIA Triad responsibilities in the UI layer:
//    Confidentiality – only displays non-sensitive aggregated results.
//    Integrity        – all inputs validated before being passed to the model.
//    Availability     – every async operation is guarded; UI never freezes.
// =============================================================================

using System;
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
        // -----------------------------------------------------------------------
        //  FIELDS
        // -----------------------------------------------------------------------
        private readonly ThreatDetectionController _controller;
        private string _lastDatasetPath = string.Empty;

        // Colour constants (hex strings → Brushes)
        private static readonly SolidColorBrush BrushGreen = new SolidColorBrush(Color.FromRgb(0x3F, 0xB9, 0x50));
        private static readonly SolidColorBrush BrushRed = new SolidColorBrush(Color.FromRgb(0xF8, 0x51, 0x49));
        private static readonly SolidColorBrush BrushOrange = new SolidColorBrush(Color.FromRgb(0xD2, 0x99, 0x22));
        private static readonly SolidColorBrush BrushBlue = new SolidColorBrush(Color.FromRgb(0x1F, 0x6F, 0xEB));
        private static readonly SolidColorBrush BrushGray = new SolidColorBrush(Color.FromRgb(0x30, 0x36, 0x3D));
        private static readonly SolidColorBrush BrushText = new SolidColorBrush(Color.FromRgb(0xE6, 0xED, 0xF3));
        private static readonly SolidColorBrush BrushSecondary = new SolidColorBrush(Color.FromRgb(0x8B, 0x94, 0x9E));

        // -----------------------------------------------------------------------
        //  CONSTRUCTOR
        // -----------------------------------------------------------------------
        public MainWindow()
        {
            InitializeComponent();
            _controller = new ThreatDetectionController();
            SampleProfileComboBox.SelectedIndex = 0;
            RefreshAuditDisplay();
        }

        // -----------------------------------------------------------------------
        //  BUTTON HANDLERS – MODEL MANAGEMENT
        // -----------------------------------------------------------------------

        private async void UploadCsvButton_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                Title = "Select Training Dataset CSV",
                Filter = "CSV Files (*.csv)|*.csv|All Files (*.*)|*.*"
            };

            if (dlg.ShowDialog() != true) return;
            _lastDatasetPath = dlg.FileName;
            await TrainModelAsync();
        }

        private async void RetrainModelButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_lastDatasetPath) || !File.Exists(_lastDatasetPath))
            {
                ShowError("No dataset available. Please upload a training dataset first.");
                return;
            }
            await TrainModelAsync();
        }

        private void SaveModelButton_Click(object sender, RoutedEventArgs e)
        {
            if (!_controller.IsModelReady)
            {
                ShowError("Please train the model before saving.");
                return;
            }

            var dlg = new SaveFileDialog
            {
                Title = "Save Trained Model",
                Filter = "ML.NET Model (*.zip)|*.zip",
                DefaultExt = "zip",
                FileName = $"insider_threat_model_{DateTime.Now:yyyyMMdd_HHmm}"
            };

            if (dlg.ShowDialog() != true) return;

            try
            {
                _controller.SaveModel(dlg.FileName);
                ShowSuccess($"Model saved successfully to:\n{dlg.FileName}");
                RefreshAuditDisplay();
            }
            catch (Exception ex)
            {
                ShowError($"Failed to save model:\n{ex.Message}");
            }
        }

        private void LoadModelButton_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                Title = "Load Trained Model",
                Filter = "ML.NET Model (*.zip)|*.zip"
            };

            if (dlg.ShowDialog() != true) return;

            try
            {
                _controller.LoadModel(dlg.FileName);
                SetModelReadyState(true,
                    $"Model loaded from file: {Path.GetFileName(dlg.FileName)}" +
                    " (note: evaluation metrics require retraining)");
                EvaluationSummaryText.Text =
                    "Model loaded from file.\n" +
                    "Full evaluation metrics are only available after training with a dataset.\n" +
                    "You can still make predictions using the loaded model.";
                RefreshAuditDisplay();
            }
            catch (Exception ex)
            {
                ShowError($"Failed to load model:\n{ex.Message}");
            }
        }

        // -----------------------------------------------------------------------
        //  TRAINING (async – keeps UI responsive)
        // -----------------------------------------------------------------------
        private async Task TrainModelAsync()
        {
            SetTrainingState(true);
            PredictionResultText.Text = "Awaiting prediction after training...";
            PredictionResultText.Foreground = BrushSecondary;
            EvaluationSummaryText.Text = "Training model, please wait...";
            EvaluationSummaryText.Foreground = BrushSecondary;
            HideResultVisuals();

            Core.Services.DataPreprocessor.PreprocessingReport? report = null;
            Exception? trainingError = null;

            try
            {
                report = await Task.Run(() => _controller.TrainModel(_lastDatasetPath));
            }
            catch (Exception ex)
            {
                trainingError = ex;
            }
            finally
            {
                SetTrainingState(false);
            }

            if (trainingError != null)
            {
                SetModelReadyState(false, $"Training failed: {trainingError.Message}");
                EvaluationSummaryText.Text = $"Training error:\n{trainingError.Message}";
                EvaluationSummaryText.Foreground = BrushRed;
                return;
            }

            // Show preprocessing summary
            if (report != null)
            {
                PreprocessingPanel.Visibility = Visibility.Visible;
                PreprocessingText.Text = BuildPreprocessingSummary(report);
            }

            // Show evaluation
            string evalSummary = _controller.GetEvaluationSummary();
            EvaluationSummaryText.Text = evalSummary;
            EvaluationSummaryText.Foreground = BrushText;

            // Update threshold badge
            ThresholdBadge.Text = $"Threshold: {_controller.GetBaselineProbability():P1}";
            ThresholdBadge.Visibility = Visibility.Visible;

            SetModelReadyState(true,
                "FastTree Boosted Decision Tree trained successfully ✓  " +
                "|  Multi-model comparison completed");

            RefreshAuditDisplay();
        }

        // -----------------------------------------------------------------------
        //  PREDICTION FROM SAMPLE PROFILE
        // -----------------------------------------------------------------------
        private void SampleProfileComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (!_controller.IsModelReady) return;
            if (SampleProfileComboBox.SelectedIndex <= 0) return;

            var item = (ComboBoxItem)SampleProfileComboBox.SelectedItem;
            string name = item.Content.ToString() ?? string.Empty;

            try
            {
                var input = _controller.GetProfile(name);
                RunPredictionAndDisplay(input, name);
            }
            catch (Exception ex)
            {
                ShowError($"Profile error: {ex.Message}");
            }
        }

        // -----------------------------------------------------------------------
        //  PREDICTION FROM CSV
        // -----------------------------------------------------------------------
        private void PredictFromCsvButton_Click(object sender, RoutedEventArgs e)
        {
            if (!_controller.IsModelReady)
            {
                ShowError("Please train or load a model before making predictions.");
                return;
            }

            var dlg = new OpenFileDialog
            {
                Title = "Select CSV Record for Prediction",
                Filter = "CSV Files (*.csv)|*.csv"
            };

            if (dlg.ShowDialog() != true) return;

            try
            {
                var input = _controller.LoadSingleRowFromCsv(dlg.FileName, 1);
                string summary = BuildInputSummary(input);
                RunPredictionAndDisplay(input, $"CSV Upload\n\n{summary}");
            }
            catch (Exception ex)
            {
                ShowError($"Error reading CSV:\n{ex.Message}");
            }
        }

        // -----------------------------------------------------------------------
        //  AUDIT LOG REFRESH
        // -----------------------------------------------------------------------
        private void RefreshAuditBtn_Click(object sender, RoutedEventArgs e)
        {
            RefreshAuditDisplay();
        }

        // -----------------------------------------------------------------------
        //  CORE PREDICTION & DISPLAY LOGIC
        // -----------------------------------------------------------------------
        private void RunPredictionAndDisplay(UserBehaviour input, string sourceDescription)
        {
            try
            {
                var prediction = _controller.Predict(input);
                var contributions = _controller.Explain(input);
                float baseProb = _controller.GetBaselineProbability();
                string humanExp = _controller.GenerateHumanExplanation(input, prediction);

                // ── Update classification bar visual ──
                UpdateResultVisuals(prediction);

                // ── Build result text ──
                var result = BuildResultText(sourceDescription, prediction, humanExp,
                    contributions, baseProb);

                PredictionResultText.Text = result;
                PredictionResultText.Foreground = BrushText;

                // Scroll to result
                Dispatcher.BeginInvoke(new Action(() =>
                {
                    if (VisualTreeHelper.GetParent(this) is ScrollViewer sv)
                        sv.ScrollToEnd();
                }), System.Windows.Threading.DispatcherPriority.Loaded);

                RefreshAuditDisplay();
            }
            catch (Exception ex)
            {
                PredictionResultText.Text = $"Prediction failed:\n{ex.Message}";
                PredictionResultText.Foreground = BrushRed;
            }
        }

        private string BuildResultText(
            string source,
            ThreatPrediction prediction,
            string humanExplanation,
            System.Collections.Generic.List<(string Feature, float Contribution, float Value, float BenignMean)> contributions,
            float baseProb)
        {
            var sb = new System.Text.StringBuilder();

            // Header
            sb.AppendLine($"══ Analysed: {source.Split('\n')[0]} ══");
            sb.AppendLine();

            // Core metrics
            sb.AppendLine($"  Classification    : {prediction.Classification}");
            sb.AppendLine($"  Threat Probability: {prediction.ThreatProbability:P2}");
            sb.AppendLine($"  Confidence        : {prediction.Confidence:P2}");
            sb.AppendLine($"  Risk Level        : {prediction.RiskLevel}");
            sb.AppendLine();

            // Explanation
            sb.AppendLine("── AI Explanation ──────────────────────────────");
            sb.AppendLine(humanExplanation);

            // Feature importance
            sb.AppendLine("── Feature Importance (Top 5, Perturbation-Based) ──");
            if (!float.IsNaN(baseProb))
                sb.AppendLine($"   Baseline threat probability (neutral profile): {baseProb:P2}");
            sb.AppendLine();

            foreach (var c in contributions.Take(5))
            {
                string direction = c.Contribution > 0 ? "▲ raises" : "▼ lowers";
                string flag = Math.Abs(c.Contribution) > 0.05f ? " ⚠" : "";
                sb.AppendLine(
                    $"   {MLModelManager.HumanReadableName(c.Feature),-36}: " +
                    $"Value={c.Value:F1}  " +
                    $"[Baseline≈{c.BenignMean:F1}]  " +
                    $"{direction} risk by {Math.Abs(c.Contribution):F3}{flag}");
            }

            if (source.Contains('\n'))
            {
                // CSV record – show full input
                sb.AppendLine();
                sb.AppendLine("── Uploaded Record Details ──────────────────────");
                sb.AppendLine(string.Join("\n", source.Split('\n').Skip(1)));
            }

            return sb.ToString();
        }

        // -----------------------------------------------------------------------
        //  UI HELPER METHODS
        // -----------------------------------------------------------------------
        private void UpdateResultVisuals(ThreatPrediction prediction)
        {
            ClassificationBar.Visibility = Visibility.Visible;
            RiskBadge.Visibility = Visibility.Visible;

            float pct = prediction.ThreatProbability;

            // Fill the bar proportionally to threat probability
            ClassificationFill.Width = double.NaN; // auto
            ClassificationFill.HorizontalAlignment = HorizontalAlignment.Left;

            // Colour coding
            SolidColorBrush barColor;
            SolidColorBrush badgeColor;

            switch (prediction.RiskLevel)
            {
                case "LOW RISK":
                    barColor = BrushGreen;
                    badgeColor = BrushGreen;
                    break;
                case "ELEVATED – MONITOR":
                    barColor = BrushOrange;
                    badgeColor = BrushOrange;
                    break;
                case "BORDERLINE – REVIEW RECOMMENDED":
                    barColor = BrushOrange;
                    badgeColor = BrushOrange;
                    break;
                case "HIGH RISK":
                    barColor = BrushRed;
                    badgeColor = BrushRed;
                    break;
                default: // VERY HIGH RISK, CRITICAL RISK
                    barColor = new SolidColorBrush(Color.FromRgb(0xFF, 0x00, 0x00));
                    badgeColor = barColor;
                    break;
            }

            ClassificationFill.Background = barColor;
            RiskBadge.Background = badgeColor;
            RiskBadgeText.Text = prediction.RiskLevel;

            // Set fill width dynamically via proportion
            ClassificationBar.Loaded -= ClassificationBar_Loaded;
            ClassificationBar.Loaded += ClassificationBar_Loaded;
            _lastProbability = pct;

            void ClassificationBar_Loaded(object s, RoutedEventArgs ev)
            {
                double available = ClassificationBar.ActualWidth;
                ClassificationFill.Width = available * _lastProbability;
            }

            // Also set immediately if bar is already rendered
            if (ClassificationBar.ActualWidth > 0)
                ClassificationFill.Width = ClassificationBar.ActualWidth * pct;
        }

        private float _lastProbability;

        private void HideResultVisuals()
        {
            ClassificationBar.Visibility = Visibility.Collapsed;
            RiskBadge.Visibility = Visibility.Collapsed;
        }

        private void SetTrainingState(bool isTraining)
        {
            UploadCsvButton.IsEnabled = !isTraining;
            RetrainModelButton.IsEnabled = !isTraining;
            SaveModelButton.IsEnabled = false;
            LoadModelButton.IsEnabled = !isTraining;
            SampleProfileComboBox.IsEnabled = !isTraining;
            PredictFromCsvButton.IsEnabled = !isTraining;

            ModelStatusText.Text = isTraining
                ? "⏳ Training in progress – preprocessing data and training ML model..."
                : ModelStatusText.Text;
            ModelStatusText.Foreground = isTraining ? BrushOrange : BrushSecondary;
            StatusDot.Fill = isTraining ? BrushOrange : BrushRed;
        }

        private void SetModelReadyState(bool ready, string message)
        {
            IsEnabled = true;
            ModelStatusText.Text = message;
            ModelStatusText.Foreground = ready ? BrushGreen : BrushRed;
            StatusDot.Fill = ready ? BrushGreen : BrushRed;
            SaveModelButton.IsEnabled = ready;

            // Re-enable all controls
            UploadCsvButton.IsEnabled = true;
            RetrainModelButton.IsEnabled = true;
            LoadModelButton.IsEnabled = true;
            SampleProfileComboBox.IsEnabled = true;
            PredictFromCsvButton.IsEnabled = true;
        }

        private void RefreshAuditDisplay()
        {
            AuditLogText.Text = _controller.GetRecentAuditEntries();
            AuditLogPathText.Text = $"📁 Log location: {_controller.GetAuditLogPath()}";
        }

        private static string BuildInputSummary(UserBehaviour u) =>
            $"Uploaded record key features:\n" +
            $"  Department           : {u.employee_department}\n" +
            $"  Position             : {u.employee_position}\n" +
            $"  Seniority (years)    : {u.employee_seniority_years}\n" +
            $"  Contractor           : {(u.is_contractor == 1 ? "Yes" : "No")}\n" +
            $"  Total Pages Printed  : {u.total_printed_pages}\n" +
            $"  Off-Hours Printing   : {u.num_printed_pages_off_hours}\n" +
            $"  Files Burned         : {u.total_files_burned}\n" +
            $"  Building Entries     : {u.num_entries}\n" +
            $"  Late Exits           : {(u.late_exit_flag == 1 ? "Yes" : "No")}\n" +
            $"  Weekend Entry        : {(u.entry_during_weekend == 1 ? "Yes" : "No")}\n" +
            $"  Is Abroad            : {(u.is_abroad == 1 ? "Yes" : "No")}";

        private static string BuildPreprocessingSummary(
            Core.Services.DataPreprocessor.PreprocessingReport r) =>
            $"Original rows: {r.OriginalRowCount:N0}  |  " +
            $"Duplicates removed: {r.DuplicatesRemoved:N0}  |  " +
            $"Invalid labels: {r.InvalidLabelRows}  |  " +
            $"Missing values imputed: {r.MissingValuesImputed:N0}  |  " +
            $"Outliers capped: {r.OutliersCapped:N0}  |  " +
            $"Final clean rows: {r.CleanedRowCount:N0}";

        private static void ShowError(string message) =>
            MessageBox.Show(message, "Error", MessageBoxButton.OK, MessageBoxImage.Warning);

        private static void ShowSuccess(string message) =>
            MessageBox.Show(message, "Success", MessageBoxButton.OK, MessageBoxImage.Information);
    }
}