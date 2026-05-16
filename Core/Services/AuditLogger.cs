// =============================================================================
//  AuditLogger.cs
//  Insider Threat Detection System – COS720 2026
//
//  CIA Triad:
//    Confidentiality – logs are written locally; no PII transmitted externally.
//    Integrity        – every log entry is time-stamped and append-only.
//    Availability     – failures in logging never crash the detection system.
//
//  Software Engineering: Singleton + Single Responsibility Principle.
// =============================================================================

using System;
using System.IO;
using System.Text;

namespace InsiderThreatDetection.Core.Services
{
    /// <summary>
    /// Thread-safe append-only audit logger (Singleton).
    /// Records all security-relevant events: model training, predictions,
    /// model saves/loads, preprocessing, and errors.
    ///
    /// CIA Integrity: every entry is timestamped UTC and append-only.
    /// In a real enterprise this would integrate with a SIEM; here it writes
    /// to %LOCALAPPDATA%\InsiderThreatDetection\Logs\.
    /// </summary>
    public sealed class AuditLogger
    {
        // ── Singleton ──────────────────────────────────────────────────────────
        private static readonly Lazy<AuditLogger> _instance =
            new Lazy<AuditLogger>(() => new AuditLogger());

        public static AuditLogger Instance => _instance.Value;

        // ── State ──────────────────────────────────────────────────────────────
        private readonly string _logDirectory;
        private readonly object _lock = new object();

        private AuditLogger()
        {
            _logDirectory = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "InsiderThreatDetection", "Logs");

            try { Directory.CreateDirectory(_logDirectory); }
            catch { /* Logging must never prevent startup */ }
        }

        // ── Public API ─────────────────────────────────────────────────────────

        public void LogModelTrained(string dataPath, int rowCount, double accuracy, double f1)
        {
            Write("MODEL_TRAINED",
                $"Dataset=\"{Path.GetFileName(dataPath)}\" Rows={rowCount:N0} " +
                $"Accuracy={accuracy:P2} F1={f1:P2}");
        }

        public void LogPrediction(string source, string classification,
                                   float probability, string riskLevel)
        {
            Write("PREDICTION",
                $"Source=\"{source}\" Result={classification} " +
                $"Probability={probability:P2} Risk={riskLevel}");
        }

        public void LogModelSaved(string path)
            => Write("MODEL_SAVED", $"Path=\"{path}\"");

        public void LogModelLoaded(string path)
            => Write("MODEL_LOADED", $"Path=\"{path}\"");

        public void LogError(string context, string message)
            => Write("ERROR", $"Context=\"{context}\" Message=\"{message}\"");

        public void LogSecurityEvent(string eventType, string details)
            => Write("SECURITY_EVENT", $"Type={eventType} Details=\"{details}\"");

        public void LogPreprocessing(string report)
            => Write("PREPROCESSING", report.Replace(Environment.NewLine, " | "));

        public void LogCsvPrediction(string fileName, int rowIndex, string classification,
                                      float probability, string riskLevel)
        {
            Write("CSV_PREDICTION",
                $"File=\"{fileName}\" Row={rowIndex} Result={classification} " +
                $"Probability={probability:P2} Risk={riskLevel}");
        }

        // ── Path helpers ───────────────────────────────────────────────────────

        public string GetCurrentLogPath()
            => Path.Combine(_logDirectory, $"audit_{DateTime.UtcNow:yyyyMMdd}.log");

        /// <summary>Returns the last <paramref name="count"/> lines for in-app display.</summary>
        public string GetRecentEntries(int count = 30)
        {
            try
            {
                string path = GetCurrentLogPath();
                if (!File.Exists(path)) return "No audit entries for today yet.";
                var lines = File.ReadAllLines(path);
                int skip = Math.Max(0, lines.Length - count);
                return string.Join(Environment.NewLine, lines, skip, lines.Length - skip);
            }
            catch
            {
                return "Unable to read audit log.";
            }
        }

        // ── Private helpers ────────────────────────────────────────────────────

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
                // CIA Availability: logging failures must never crash the system
            }
        }
    }
}