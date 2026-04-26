"""
GuardNet – Alert & Logging System
Handles console alerts, sound alerts, simulated email alerts, and CSV logging.
"""

import os
import sys
import csv
import time
import smtplib
import threading
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    LOG_FILE, LOG_DIR,
    ENABLE_SOUND_ALERT, ENABLE_EMAIL_ALERT,
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_RECIPIENT,
    ALERT_COOLDOWN_SECONDS, VIOLENCE_THRESHOLD
)


# ─── CSV Logger ───────────────────────────────────────────────────────────────

class EventLogger:
    """Thread-safe CSV logger for violence detection events."""

    FIELDNAMES = ["timestamp", "confidence", "source", "notes"]

    def __init__(self, log_path: str = LOG_FILE):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def log(self, confidence: float, source: str = "camera", notes: str = ""):
        row = {
            "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "confidence": f"{confidence:.4f}",
            "source"    : source,
            "notes"     : notes,
        }
        with self._lock:
            with open(self.log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow(row)
        return row


# ─── Alert Manager ────────────────────────────────────────────────────────────

class AlertManager:
    """
    Fires alerts (console / sound / email) with a cooldown to avoid spam.
    Thread-safe.
    """

    def __init__(self):
        self._last_alert_time = 0.0
        self._lock = threading.Lock()
        self.logger = EventLogger()

    # ── Public ────────────────────────────────────────────────────────────────

    def trigger(self, confidence: float, source: str = "camera"):
        """Call this every time violence is detected."""
        now = time.time()
        with self._lock:
            if now - self._last_alert_time < ALERT_COOLDOWN_SECONDS:
                return
            self._last_alert_time = now

        # Fire all alert types (non-blocking where possible)
        self._console_alert(confidence, source)
        row = self.logger.log(confidence, source)

        if ENABLE_SOUND_ALERT:
            threading.Thread(target=self._sound_alert, daemon=True).start()

        if ENABLE_EMAIL_ALERT:
            threading.Thread(
                target=self._email_alert, args=(confidence, source, row["timestamp"]),
                daemon=True
            ).start()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _console_alert(self, confidence: float, source: str):
        ts = datetime.now().strftime("%H:%M:%S")
        bar = "█" * int(confidence * 20)
        print(
            f"\n{'='*60}\n"
            f"  ⚠️  VIOLENCE DETECTED  [{ts}]\n"
            f"  Source     : {source}\n"
            f"  Confidence : {bar} {confidence*100:.1f}%\n"
            f"  Threshold  : {VIOLENCE_THRESHOLD*100:.0f}%\n"
            f"{'='*60}\n"
        )

    def _sound_alert(self):
        """Cross-platform terminal bell; replace with pygame/playsound if desired."""
        try:
            # Try system bell
            sys.stdout.write("\a")
            sys.stdout.flush()
            # On Linux with 'beep' installed:
            os.system("beep -f 880 -l 200 2>/dev/null || true")
        except Exception:
            pass

    def _email_alert(self, confidence: float, source: str, timestamp: str):
        """Simulated SMTP email alert (set ENABLE_EMAIL_ALERT=True in config)."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[GuardNet] ⚠️ Violence Detected – {timestamp}"
            msg["From"]    = SMTP_USER
            msg["To"]      = ALERT_RECIPIENT

            body = f"""
            <html><body>
            <h2 style="color:red;">⚠️ GuardNet – Violence Alert</h2>
            <table>
              <tr><td><b>Timestamp</b></td><td>{timestamp}</td></tr>
              <tr><td><b>Source</b></td><td>{source}</td></tr>
              <tr><td><b>Confidence</b></td><td>{confidence*100:.1f}%</td></tr>
              <tr><td><b>Threshold</b></td><td>{VIOLENCE_THRESHOLD*100:.0f}%</td></tr>
            </table>
            <p>Please review the footage immediately.</p>
            </body></html>
            """
            msg.attach(MIMEText(body, "html"))

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.sendmail(SMTP_USER, ALERT_RECIPIENT, msg.as_string())
            print("[INFO] Email alert sent.")
        except Exception as e:
            print(f"[WARN] Email alert failed: {e}")


# ─── Singleton ────────────────────────────────────────────────────────────────

_alert_manager: AlertManager | None = None

def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
