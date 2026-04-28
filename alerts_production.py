"""
GuardNet – Alert & Logging System (Production)
Supports dynamic email input (no hardcoded addresses).
Attaches clip to email when available.
"""
import os, sys, csv, time, smtplib, threading
from datetime import datetime
from email.mime.text        import MIMEText
from email.mime.multipart   import MIMEMultipart
from email.mime.base        import MIMEBase
from email                  import encoders

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LOG_FILE, LOG_DIR, ALERT_COOLDOWN_SECONDS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")


class EventLogger:
    FIELDNAMES = ["timestamp", "camera_id", "confidence", "clip_path"]

    def __init__(self, path=LOG_FILE):
        self.path  = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def log(self, confidence: float, camera_id: str = "cam0",
            clip_path: str = "") -> dict:
        row = {
            "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id" : camera_id,
            "confidence": f"{confidence:.4f}",
            "clip_path" : clip_path,
        }
        with self._lock:
            with open(self.path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)
        return row

    def read_all(self):
        try:
            with open(self.path, "r") as f:
                return list(csv.DictReader(f))
        except Exception:
            return []


class AlertManager:
    def __init__(self):
        self._last    = {}          # camera_id → last alert time
        self._lock    = threading.Lock()
        self.logger   = EventLogger()

    def trigger(self, confidence: float, camera_id: str = "cam0",
                alert_email: str = "", clip_path: str = ""):
        now = time.time()
        with self._lock:
            if now - self._last.get(camera_id, 0) < ALERT_COOLDOWN_SECONDS:
                return
            self._last[camera_id] = now

        self._console(confidence, camera_id)
        row = self.logger.log(confidence, camera_id, clip_path)

        if alert_email and SMTP_USER and SMTP_PASS:
            threading.Thread(
                target=self._email,
                args=(confidence, camera_id, row["timestamp"],
                      alert_email, clip_path),
                daemon=True,
            ).start()

    def _console(self, conf, cam):
        ts  = datetime.now().strftime("%H:%M:%S")
        bar = "█" * int(conf * 20)
        print(f"\n{'='*55}\n  ⚠️  VIOLENCE DETECTED [{ts}]  Camera: {cam}\n"
              f"  Confidence: {bar} {conf*100:.1f}%\n{'='*55}\n")

    def _email(self, conf, cam, ts, recipient, clip_path):
        try:
            msg              = MIMEMultipart()
            msg["Subject"]   = f"[GuardNet] ⚠️ Violence Detected – {ts}"
            msg["From"]      = SMTP_USER
            msg["To"]        = recipient
            body = f"""
            <html><body style="font-family:Arial;background:#0d1117;color:#e6edf3;padding:20px">
            <h2 style="color:#f85149">⚠️ GuardNet – Violence Alert</h2>
            <table style="border-collapse:collapse;width:100%">
              <tr><td style="padding:8px;color:#8892a4"><b>Timestamp</b></td><td>{ts}</td></tr>
              <tr><td style="padding:8px;color:#8892a4"><b>Camera</b></td><td>{cam}</td></tr>
              <tr><td style="padding:8px;color:#8892a4"><b>Confidence</b></td><td>{conf*100:.1f}%</td></tr>
            </table>
            <p style="color:#f85149">Please review the footage immediately.</p>
            </body></html>"""
            msg.attach(MIMEText(body, "html"))

            # Attach clip if it exists
            if clip_path and os.path.exists(clip_path):
                with open(clip_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition",
                                    f'attachment; filename="{os.path.basename(clip_path)}"')
                    msg.attach(part)

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                s.starttls()
                s.login(SMTP_USER, SMTP_PASS)
                s.sendmail(SMTP_USER, recipient, msg.as_string())
            print(f"[INFO] Email alert sent to {recipient}")
        except Exception as e:
            print(f"[WARN] Email failed: {e}")


_alert_mgr = None
def get_alert_manager() -> AlertManager:
    global _alert_mgr
    if _alert_mgr is None:
        _alert_mgr = AlertManager()
    return _alert_mgr
