"""Messenger bots: Telegram first, WhatsApp dropping into the same dispatcher.

The transport modules (telegram.py / whatsapp.py) call into `dispatch.handle`
with a normalized `IncomingMessage`. Dispatch resolves the chat to a Vaaani
user, runs the command/state-machine logic, and returns one or more
`OutgoingReply` objects the transport then renders in its own way (text,
audio, document).
"""
from .dispatch import IncomingMessage, OutgoingReply, handle_incoming

__all__ = ["IncomingMessage", "OutgoingReply", "handle_incoming"]
