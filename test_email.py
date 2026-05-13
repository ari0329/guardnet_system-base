"""
GuardNet – Full Alert System Test
==================================
Tests all 4 alert channels independently and then together:
  1. Email (SMTP / Gmail app password)
  2. Twilio SMS
  3. Twilio Voice Call
  4. Local alert sound

Usage:
    python test_email.py

Fill in your credentials in the CONFIG block below, or put them in a .env file.

REAL KOLKATA POLICE NUMBERS (replace POLICE_NUMBER with the nearest one):
─────────────────────────────────────────────────────────────────────────
  Emergency (national)          : 100
  Lalbazar Control Room (24×7)  : +913322143230
  Gariahat Police Station        : +913324863702
  Park Street Police Station     : +913322268321
  Ballygunge Police Station      : +913322872100
  Tollygunge Police Station      : +913324642765
  Kasba Police Station           : +913324420164
  Kalighat Police Station        : +913324540177
  Bhowanipur Police Station      : (033) 2454-0177
  Bowbazar Police Station        : (033) 2236-8100

  Use E.164 format for Twilio: +91 then the number without leading 0.
  On a Twilio TRIAL account the number must be verified first at:
  console.twilio.com → Phone Numbers → Verified Caller IDs
"""

import os
import sys
import time
import smtplib
import threading
from datetime import datetime
from email.mime.text      import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — fill these in (or set as environment variables in .env)
# ══════════════════════════════════════════════════════════════════════════════

# ── Email ─────────────────────────────────────────────────────────────────────
SMTP_USER       = os.getenv("SMTP_USER",       "palarindam422@gmail.com")
SMTP_PASS       = os.getenv("SMTP_PASS",       "opzu msaz yaoj rpll")   # Gmail app password
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "palarindam433@gmail.com")

# ── Twilio ────────────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN",  "your_auth_token_here")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "+1XXXXXXXXXX")   # Your Twilio number

# ── Police / recipient phone ───────────────────────────────────────────────────
# Real Kolkata Police — Lalbazar Control Room (24×7 operational dispatch)
# Change this to the nearest police station for your deployment location.
#
#   Lalbazar Control Room (HQ, 24×7) : +913322143230
#   Gariahat PS                       : +913324863702
#   Park Street PS                    : +913322268321
#   Ballygunge PS                     : +913322872100
#   Tollygunge PS                     : +913324642765
#   Kasba PS                          : +913324420164
#
POLICE_NUMBER = os.getenv("POLICE_NUMBER", "+913322143230")   # ← Lalbazar Control Room

# ── Location label ─────────────────────────────────────────────────────────────
# This appears in the SMS body and is spoken in the voice call.
# Change this to your actual deployment location.
#
# Examples:
#   "Gate 2, South City Mall, Prince Anwar Shah Road, Kolkata"
#   "Platform 4, Howrah Junction, Kolkata"
#   "Crossing of Park Street and AJC Bose Road, Kolkata"
#   "Lobby, Infinity IT Park, New Town, Kolkata"
#
LOCATION_LABEL = os.getenv(
    "LOCATION_LABEL",
    "Lalbazar Police HQ Area, Kolkata"   # ← real location, change as needed
)

# ── Mock incident data ────────────────────────────────────────────────────────
MOCK_CONFIDENCE = 0.87          # 87 % — well above the 20 % threshold
MOCK_CAMERA_ID  = "CAM-1"
MOCK_TIMESTAMP  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ══════════════════════════════════════════════════════════════════════════════

SEPARATOR = "=" * 60


def header(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def ok(msg: str):
    print(f"  ✅  {msg}")


def fail(msg: str):
    print(f"  ❌  {msg}")


def info(msg: str):
    print(f"  ℹ️   {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Email
# ══════════════════════════════════════════════════════════════════════════════

def test_email() -> bool:
    header("TEST 1 / 4 — Email Alert (Gmail SMTP)")

    if not SMTP_USER or not SMTP_PASS:
        fail("SMTP_USER or SMTP_PASS is empty — skipping email test.")
        return False

    try:
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = f"[GuardNet TEST] ⚠️ Violence Alert – {MOCK_TIMESTAMP}"
        msg["From"]    = SMTP_USER
        msg["To"]      = EMAIL_RECIPIENT

        html_body = f"""
        <html>
        <body style="font-family:Arial;background:#0d1117;color:#e6edf3;padding:24px">
          <h2 style="color:#f85149">⚠️ GuardNet – Violence Detected (TEST)</h2>
          <table style="border-collapse:collapse;width:100%;max-width:480px">
            <tr>
              <td style="padding:8px 12px;color:#8892a4;width:140px"><b>Timestamp</b></td>
              <td style="padding:8px 12px">{MOCK_TIMESTAMP}</td>
            </tr>
            <tr style="background:#161b22">
              <td style="padding:8px 12px;color:#8892a4"><b>Camera</b></td>
              <td style="padding:8px 12px">{MOCK_CAMERA_ID}</td>
            </tr>
            <tr>
              <td style="padding:8px 12px;color:#8892a4"><b>Confidence</b></td>
              <td style="padding:8px 12px;color:#f85149;font-weight:bold">
                {MOCK_CONFIDENCE * 100:.1f}%
              </td>
            </tr>
            <tr style="background:#161b22">
              <td style="padding:8px 12px;color:#8892a4"><b>Location</b></td>
              <td style="padding:8px 12px">{LOCATION_LABEL}</td>
            </tr>
            <tr>
              <td style="padding:8px 12px;color:#8892a4"><b>Nearest Police</b></td>
              <td style="padding:8px 12px">{POLICE_NUMBER}</td>
            </tr>
            <tr style="background:#161b22">
              <td style="padding:8px 12px;color:#8892a4"><b>Threshold</b></td>
              <td style="padding:8px 12px">20% (triggered at ≥ 20%)</td>
            </tr>
          </table>
          <p style="margin-top:20px;color:#f85149">
            ⚠️ This is a TEST alert. No real incident has occurred.
          </p>
          <p style="margin-top:8px;color:#8892a4;font-size:12px">
            Real incident: Kolkata Police Lalbazar Control Room — 033-2214-3230 | Emergency: 100
          </p>
        </body>
        </html>
        """

        msg.attach(MIMEText(html_body, "html"))

        info(f"Connecting to smtp.gmail.com:587 …")
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, EMAIL_RECIPIENT, msg.as_string())

        ok(f"Email sent → {EMAIL_RECIPIENT}")
        ok("Check your inbox (and spam folder).")
        return True

    except smtplib.SMTPAuthenticationError:
        fail("Authentication failed.")
        info("Make sure you are using a Gmail App Password, not your normal password.")
        info("Generate at: myaccount.google.com → Security → App Passwords")
        return False
    except Exception as exc:
        fail(f"Email error: {exc}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Twilio SMS
# ══════════════════════════════════════════════════════════════════════════════

def test_twilio_sms() -> bool:
    header("TEST 2 / 4 — Twilio SMS to Police Number")

    if "ACxxxx" in TWILIO_ACCOUNT_SID or "your_auth" in TWILIO_AUTH_TOKEN:
        fail("Twilio credentials still contain placeholder values — skipping.")
        info("Replace TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER.")
        return False

    if not POLICE_NUMBER or POLICE_NUMBER.startswith("+91XXXX"):
        fail("POLICE_NUMBER not set — skipping SMS test.")
        info("Set POLICE_NUMBER to a verified E.164 number, e.g. +913322143230")
        return False

    try:
        from twilio.rest import Client
    except ImportError:
        fail("'twilio' package not installed.")
        info("Run:  pip install twilio")
        return False

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        body = (
            f"[GUARDNET ALERT] VIOLENCE DETECTED\n"
            f"Time      : {MOCK_TIMESTAMP}\n"
            f"Camera    : {MOCK_CAMERA_ID}\n"
            f"Confidence: {MOCK_CONFIDENCE * 100:.1f}%\n"
            f"Location  : {LOCATION_LABEL}\n"
            f"Threshold : 20% (minimal violence detection enabled)\n"
            f"Action    : Immediate response required.\n"
            f"Kolkata Police: 100 | Lalbazar: 033-2214-3230\n"
            f"(This is a TEST message from GuardNet)"
        )

        info(f"Sending SMS: {TWILIO_FROM_NUMBER} → {POLICE_NUMBER} …")
        message = client.messages.create(
            body=body,
            from_=TWILIO_FROM_NUMBER,
            to=POLICE_NUMBER,
        )
        ok(f"SMS sent! SID: {message.sid}")
        ok(f"Status: {message.status}")
        return True

    except Exception as exc:
        fail(f"Twilio SMS error: {exc}")
        _twilio_hint(str(exc))
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Twilio Voice Call
# ══════════════════════════════════════════════════════════════════════════════

def test_twilio_voice() -> bool:
    header("TEST 3 / 4 — Twilio Automated Voice Call")

    if "ACxxxx" in TWILIO_ACCOUNT_SID or "your_auth" in TWILIO_AUTH_TOKEN:
        fail("Twilio credentials still contain placeholder values — skipping.")
        return False

    if not POLICE_NUMBER or POLICE_NUMBER.startswith("+91XXXX"):
        fail("POLICE_NUMBER not set — skipping voice call test.")
        info("Set POLICE_NUMBER to a Twilio-verified E.164 number.")
        return False

    try:
        from twilio.rest import Client
    except ImportError:
        fail("'twilio' package not installed.")
        info("Run:  pip install twilio")
        return False

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        spoken_message = (
            f"Attention. This is an automated alert from GuardNet, "
            f"a C C T V intelligence system. "
            f"Violence has been detected at {LOCATION_LABEL}. "
            f"Camera {MOCK_CAMERA_ID} recorded an incident at {MOCK_TIMESTAMP} "
            f"with a confidence level of {MOCK_CONFIDENCE * 100:.0f} percent. "
            f"The violence detection threshold is set to 20 percent "
            f"to catch minimal and early stage violence. "
            f"Immediate police response is required. "
            f"Repeating — Violence detected at {LOCATION_LABEL}. "
            f"Camera {MOCK_CAMERA_ID}. "
            f"Confidence {MOCK_CONFIDENCE * 100:.0f} percent. "
            f"Please respond immediately. "
            f"For backup contact Lalbazar Control Room at 033 2214 3230. "
            f"This was a test alert. Thank you."
        )

        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f'  <Say voice="alice" language="en-IN">{spoken_message}</Say>'
            "  <Pause length=\"2\"/>"
            f'  <Say voice="alice" language="en-IN">'
            f"    This concludes the GuardNet alert. Lalbazar Control Room: 033 2214 3230."
            f"  </Say>"
            "</Response>"
        )

        info(f"Initiating voice call: {TWILIO_FROM_NUMBER} → {POLICE_NUMBER} …")
        call = client.calls.create(
            to=POLICE_NUMBER,
            from_=TWILIO_FROM_NUMBER,
            twiml=twiml,
        )
        ok(f"Voice call initiated! SID: {call.sid}")
        ok(f"Status: {call.status}")
        info("The phone should ring within ~5 seconds.")
        return True

    except Exception as exc:
        fail(f"Twilio Voice error: {exc}")
        _twilio_hint(str(exc))
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Local alert sound
# ══════════════════════════════════════════════════════════════════════════════

def test_alert_sound() -> bool:
    header("TEST 4 / 4 — Local Alert Sound")

    ALERT_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "alert.wav"
    )

    # ── Try playsound ──────────────────────────────────────────────────────
    try:
        from playsound import playsound
        if os.path.exists(ALERT_FILE):
            info(f"Playing {ALERT_FILE} via playsound …")
            playsound(ALERT_FILE, block=True)
            ok("Sound played via playsound.")
            return True
        else:
            info("assets/alert.wav not found — will try pygame synthesised beep.")
    except ImportError:
        info("playsound not installed — trying pygame.")
    except Exception as exc:
        info(f"playsound failed ({exc}) — trying pygame.")

    # ── Try pygame ─────────────────────────────────────────────────────────
    try:
        import pygame
        import numpy as np

        pygame.mixer.init(frequency=44100, size=-16, channels=2)

        if os.path.exists(ALERT_FILE):
            info(f"Playing {ALERT_FILE} via pygame …")
            pygame.mixer.music.load(ALERT_FILE)
            pygame.mixer.music.play()
            time.sleep(2)
            ok("Sound played via pygame.")
        else:
            info("Synthesising 440 Hz beep via pygame (no alert.wav found) …")
            sr   = 44100
            t    = np.linspace(0, 1.5, int(sr * 1.5), False)
            wave = (np.sin(2 * 3.14159 * 440 * t) * 32767).astype("int16")
            stereo = np.column_stack([wave, wave])
            sound  = pygame.sndarray.make_sound(stereo)
            sound.play()
            time.sleep(1.8)
            ok("Synthesised beep played via pygame.")
        return True

    except ImportError:
        info("pygame not installed — falling back to terminal bell.")
    except Exception as exc:
        info(f"pygame failed ({exc}) — falling back to terminal bell.")

    # ── Terminal bell fallback ─────────────────────────────────────────────
    print("\a\a\a", end="", flush=True)
    ok("Terminal bell fired (install playsound or pygame for audio).")
    info("pip install playsound    OR    pip install pygame numpy")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Twilio error hints
# ══════════════════════════════════════════════════════════════════════════════

def _twilio_hint(error_str: str):
    error_lower = error_str.lower()
    if "21219" in error_str or "not verified" in error_lower:
        info("Trial account: the TO number must be verified.")
        info("Verify at: console.twilio.com → Phone Numbers → Verified Caller IDs")
    elif "21608" in error_str or "unverified" in error_lower:
        info("The FROM number is not a valid Twilio number.")
        info("Buy a number at: console.twilio.com → Phone Numbers → Buy a Number")
    elif "20003" in error_str or "authenticate" in error_lower:
        info("Account SID or Auth Token is wrong.")
        info("Copy them from: console.twilio.com → Dashboard → Account Info")
    elif "geo" in error_lower or "region" in error_lower:
        info("International calling/SMS to India may be disabled.")
        info("Enable at: console.twilio.com → Voice/SMS → Geo Permissions → India")
    elif "not a valid phone" in error_lower:
        info("Phone number must be in E.164 format: +913322143230")


# ══════════════════════════════════════════════════════════════════════════════
# FULL SYSTEM TEST — simulates a real GuardNet violence trigger
# ══════════════════════════════════════════════════════════════════════════════

def test_full_system():
    header("FULL SYSTEM TEST — Simulated Violence Detection Event")
    info(f"Camera     : {MOCK_CAMERA_ID}")
    info(f"Confidence : {MOCK_CONFIDENCE * 100:.1f}%  (threshold: 20%)")
    info(f"Timestamp  : {MOCK_TIMESTAMP}")
    info(f"Location   : {LOCATION_LABEL}")
    info(f"Police No  : {POLICE_NUMBER}  (Lalbazar Control Room, Kolkata)")
    print()

    results = {}

    results["sound"] = test_alert_sound()
    time.sleep(1)

    results["email"] = test_email()
    time.sleep(1)

    results["sms"]   = test_twilio_sms()
    time.sleep(2)

    results["voice"] = test_twilio_voice()

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  RESULTS SUMMARY")
    print(SEPARATOR)
    labels = {
        "sound": "Alert Sound   ",
        "email": "Email Alert   ",
        "sms":   "Twilio SMS    ",
        "voice": "Twilio Voice  ",
    }
    all_passed = True
    for key, label in labels.items():
        status = "✅ PASS" if results[key] else "❌ FAIL"
        if not results[key]:
            all_passed = False
        print(f"  {label}: {status}")
    print(SEPARATOR)
    if all_passed:
        print("  🎉  ALL CHANNELS WORKING — GuardNet alert system is ready!")
    else:
        print("  ⚠️   Some channels failed — see details above.")
    print(SEPARATOR)
    print()
    print("  Real Kolkata Police Contacts:")
    print("  ─────────────────────────────")
    print("  Emergency (national)       : 100")
    print("  Lalbazar Control Room (HQ) : 033-2214-3230")
    print("  Gariahat PS                : 033-2486-3702")
    print("  Park Street PS             : 033-2226-8321")
    print("  Ballygunge PS              : 033-2287-2100")
    print("  Tollygunge PS              : 033-2464-2765")
    print("  Kasba PS                   : 033-2442-0164")
    print(SEPARATOR)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  GuardNet Alert System — Full Test Suite")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Run individual tests or the full system test
    # Comment/uncomment as needed:

    test_full_system()

    # Or test channels individually:
    # test_email()
    # test_twilio_sms()
    # test_twilio_voice()
    # test_alert_sound()
