import smtplib
from dotenv import load_dotenv
import os
load_dotenv()

try:
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
    server.sendmail(
        os.getenv("SMTP_USER"),
        os.getenv("ALERT_RECIPIENT"),
        "Subject: GuardNet Test\n\nEmail alert is working!"
    )
    server.quit()
    print("SUCCESS! Check your inbox.")
except Exception as e:
    print(f"FAILED: {e}")