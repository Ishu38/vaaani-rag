"""SMS sender abstraction.

Implementations:
  - ConsoleSmsSender: writes OTPs to the log (dev / unconfigured prod).
  - Msg91SmsSender:   posts to MSG91 OTP flow (requires DLT registration in India).
  - TwilioSmsSender:  REST API call to Twilio Messages.

Pick via SMS_PROVIDER env var. Defaults to console if unset.
"""
from __future__ import annotations

from typing import Protocol

import httpx

from config import (
    MSG91_AUTH_KEY,
    MSG91_SENDER_ID,
    MSG91_TEMPLATE_ID,
    SMS_PROVIDER,
    TWILIO_FROM,
    TWILIO_SID,
    TWILIO_TOKEN,
)


class SmsSender(Protocol):
    """Anything with a `.send_otp(phone, code)` method."""
    def send_otp(self, phone: str, code: str) -> None: ...


class ConsoleSmsSender:
    """Dev sender: prints the OTP to the log so signup works locally."""

    def send_otp(self, phone: str, code: str) -> None:
        print("=" * 70)
        print(f"[sms:console] phone={phone}  OTP={code}")
        print("(Configure SMS_PROVIDER + provider creds to send real texts.)")
        print("=" * 70, flush=True)


class Msg91SmsSender:
    """MSG91 OTP API. Requires DLT-registered template in India."""

    def send_otp(self, phone: str, code: str) -> None:
        if not MSG91_AUTH_KEY:
            raise RuntimeError("MSG91_AUTH_KEY not set")
        # MSG91 expects phone with country code, no leading +
        normalized = phone.lstrip("+")
        url = "https://control.msg91.com/api/v5/otp"
        params = {
            "template_id": MSG91_TEMPLATE_ID,
            "mobile": normalized,
            "authkey": MSG91_AUTH_KEY,
            "otp": code,
            "sender": MSG91_SENDER_ID,
        }
        with httpx.Client(timeout=15) as c:
            r = c.get(url, params=params)
            r.raise_for_status()


class TwilioSmsSender:
    """Twilio Messages API. Good fallback when DLT registration isn't done."""

    def send_otp(self, phone: str, code: str) -> None:
        if not (TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM):
            raise RuntimeError("Twilio creds not set")
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json"
        data = {
            "From": TWILIO_FROM,
            "To": phone if phone.startswith("+") else f"+{phone}",
            "Body": f"Your Vaaani verification code is {code}. It expires in 10 minutes.",
        }
        with httpx.Client(timeout=15, auth=(TWILIO_SID, TWILIO_TOKEN)) as c:
            r = c.post(url, data=data)
            r.raise_for_status()


_sender: SmsSender | None = None


def get_sender() -> SmsSender:
    """Return the cached singleton SMS sender based on SMS_PROVIDER."""
    global _sender
    if _sender is None:
        if SMS_PROVIDER == "msg91" and MSG91_AUTH_KEY:
            _sender = Msg91SmsSender()
        elif SMS_PROVIDER == "twilio" and TWILIO_SID:
            _sender = TwilioSmsSender()
        else:
            _sender = ConsoleSmsSender()
    return _sender
