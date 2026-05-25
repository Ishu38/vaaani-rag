"""Email sender abstraction.

Two implementations:
  - ConsoleEmailSender: prints to stdout/log; used in dev when no SMTP creds.
  - SmtpEmailSender:    real STARTTLS SMTP; activated when SMTP_HOST is set.

`get_sender()` picks the right one at startup based on env vars.
"""
from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from typing import Protocol

from config import (
    SMTP_FROM,
    SMTP_HOST,
    SMTP_PASS,
    SMTP_PORT,
    SMTP_USE_TLS,
    SMTP_USER,
)


class EmailSender(Protocol):
    """Anything with a `.send(to, subject, body_text, body_html)` method."""
    def send(self, to: str, subject: str, body_text: str, body_html: str | None = None) -> None: ...


class ConsoleEmailSender:
    """Dev sender: writes the message to the log so verification links are visible."""

    def send(self, to: str, subject: str, body_text: str, body_html: str | None = None) -> None:
        print("=" * 70)
        print(f"[email:console] to={to}  subject={subject}")
        print("-" * 70)
        print(body_text)
        print("=" * 70, flush=True)


class SmtpEmailSender:
    """Real SMTP sender. Uses STARTTLS by default; set SMTP_USE_TLS=0 for plain."""

    def send(self, to: str, subject: str, body_text: str, body_html: str | None = None) -> None:
        msg = EmailMessage()
        msg["From"] = SMTP_FROM
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body_text)
        if body_html:
            msg.add_alternative(body_html, subtype="html")
        ctx = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            if SMTP_USE_TLS:
                s.starttls(context=ctx)
            if SMTP_USER:
                s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)


_sender: EmailSender | None = None


def get_sender() -> EmailSender:
    """Return the cached singleton sender (console in dev, SMTP if configured)."""
    global _sender
    if _sender is None:
        _sender = SmtpEmailSender() if SMTP_HOST else ConsoleEmailSender()
    return _sender
