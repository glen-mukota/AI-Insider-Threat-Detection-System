// =============================================================================
//  AuditLogger.cs
//  Insider Threat Detection System – COS720 2026
//
//  CIA Triad:
//    Confidentiality – logs are written locally; no PII transmitted externally.
//    Integrity        – every log entry is time-stamped and append-only.
//    Availability     – failures in logging never crash the detection system.
//
//  Software Engineering: Single Responsibility Principle – this class ONLY
//  handles audit logging and nothing else.
// =============================================================================

using System;
using System.IO;
using System.Text;

namespace InsiderThreatDetection.Core.Services
{
    /// <summary>
    /// Thread-safe append-only audit logger.
    /// Records all security-relevant events (model training, predictions,
    /// model saves/loads, errors) to a rotating daily log file.
    ///
    /// In a real enterprise deployment this would write to a SIEM (Security
    /// Information and Event Management) system. For this prototype, entries
    /// are written to the local %TEMP%\InsiderThreatDetection\ directory.
    /// </summary>
    public sealed class AuditLogger
    {
        // -----------------------------------------------------------------------
        //  SINGLETON
        // -----------------------------------------------------------------------
        private static readonly Lazy<AuditLogger> _instance =
            new Lazy<AuditLogger>(() => new AuditLogger());

        public static AuditLogger Instance => _instance.Value;

        // -----------------------------------------------------------------------
        //  STATE
        // -----------------------------------------------------------------------
        private readonly string _logDirectory;
        private readonly object _lock = new object();

        private AuditLogger()
        {
            _logDirectory = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "InsiderThreatDetection", "Logs");

            try { Directory.CreateDirectory(_logDirectory); }
            catch { /* If we can't create the dir, logging will silently fail */ }
        }

        // -----------------------------------------------------------------------
        //  PUBLIC API
        // -----------------------------------------------------------------------

        public void LogModelTrained(string dataPath, int rowCount, double accuracy, double f1)
        {
            Write("MODEL_TRAINED",
                $"Dataset=\"{dataPath}\" Rows={rowCount} Accuracy={accuracy:P2} F1={f1:P2}");
        }

        public void LogPrediction(string profileSource, string classification,
                                   float probability, string riskLevel)
        {
            Write("PREDICTION",
                $"Source=\"{profileSource}\" Result={classification} " +
                $"Probability={probability:P2} Risk={riskLevel}");
        }

        public void LogModelSaved(string path)
        {
            Write("MODEL_SAVED", $"Path=\"{path}\"");
        }

        public void LogModelLoaded(string path)
        {
            Write("MODEL_LOADED", $"Path=\"{path}\"");
        }

        public void LogError(string context, string message)
        {
            Write("ERROR", $"Context=\"{context}\" Message=\"{message}\"");
        }

        public void LogSecurityEvent(string eventType, string details)
        {
            Write("SECURITY_EVENT", $"Type={eventType} Details=\"{details}\"");
        }

        public void LogPreprocessing(string report)
        {
            Write("PREPROCESSING", report.Replace(Environment.NewLine, " | "));
        }

        // -----------------------------------------------------------------------
        //  PRIVATE HELPERS
        // -----------------------------------------------------------------------

        private void Write(string eventType, string details)
        {
            try
            {
                string timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss.fff");
                string logFile = Path.Combine(_logDirectory,
                    $"audit_{DateTime.UtcNow:yyyyMMdd}.log");
                string line = $"[{timestamp}] [{eventType}] {details}";

                lock (_lock)
                {
                    File.AppendAllText(logFile, line + Environment.NewLine, Encoding.UTF8);
                }
            }
            catch
            {
                // Logging must never crash the application (Availability principle)
            }
        }

        /// <summary>
        /// Returns the current audit log file path so the UI can show the user
        /// where logs are stored.
        /// </summary>
        public string GetCurrentLogPath()
        {
            return Path.Combine(_logDirectory, $"audit_{DateTime.UtcNow:yyyyMMdd}.log");
        }

        /// <summary>
        /// Returns the last N lines of the current audit log for in-app display.
        /// </summary>
        public string GetRecentEntries(int count = 20)
        {
            try
            {
                string path = GetCurrentLogPath();
                if (!File.Exists(path)) return "No audit entries yet.";
                var lines = File.ReadAllLines(path);
                int skip = Math.Max(0, lines.Length - count);
                return string.Join(Environment.NewLine, lines, skip, lines.Length - skip);
            }
            catch
            {
                return "Unable to read audit log.";
            }
        }
    }
}