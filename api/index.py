"""Vercel entry point: a lite online demo of Samvya.

A Vercel function may be at most 500 MB, and the camera and stress models need PyTorch
(several GB), so this deployment installs api/requirements.txt without it. The same app.py
then runs in lite mode: chat, spoken replies, safety replies and reports work; the camera
check-in, the USB wearable and voice typing need the full app on a computer.

In the Vercel project, set GROQ_API_KEY (chat), SARVAM_API_KEY (voice) and SECRET_KEY under
Settings > Environment Variables. Never commit them.
"""

import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("SAMVYA_WEARABLE_PORT", "off")
# /tmp is the only writable folder. It is emptied when Vercel starts a new instance, so
# profiles and reports in the online demo are temporary.
os.environ.setdefault("DATABASE_PATH", "/tmp/samvya-demo.db")

# Locally, app.py answers only requests for 127.0.0.1 or localhost, so a web page on another
# domain cannot reach the unauthenticated local APIs (DNS rebinding). This public deployment is
# reached under several Vercel addresses and custom domains, so it accepts any host name.
os.environ["SAMVYA_ALLOWED_HOSTS"] = "*"

if not os.environ.get("SECRET_KEY"):
    # Voice tickets are signed by one function instance and may be read by another, so the key
    # must be the same everywhere. Derive one from the provider keys if SECRET_KEY is not set.
    material = "".join(os.environ.get(name, "") for name in ("GROQ_API_KEY", "SARVAM_API_KEY"))
    if material:
        os.environ["SECRET_KEY"] = hashlib.sha256(("samvya-voice-ticket:" + material).encode()).hexdigest()

from app import app  # noqa: E402  (Vercel serves this WSGI app)
