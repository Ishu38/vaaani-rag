"""Transport-agnostic message dispatcher.

Receives a normalized `IncomingMessage`, returns a list of `OutgoingReply`
objects. Transports (telegram.py, whatsapp.py) handle the wire format on
each side. The same dispatcher must work for both — never reference
Telegram-specific or WhatsApp-specific concepts here.

State model:
  - First contact (chat_id not in messenger_links): only /start, /help,
    and plain link-code messages are accepted. Everything else gets the
    onboarding nudge.
  - Linked + mode='chat': plain text → RAG. /commands switch behaviour.
  - mode='review': digit replies grade the current card; everything else
    cancels review and falls back to chat.
  - mode='awaiting_explanation': next plain text is treated as Feynman
    explanation; commands cancel the wait.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from auth import service as auth_service

from . import store

# Public URL for audio files. Telegram (and WA) fetch attachment URLs
# server-to-server, so they need a public hostname — using the production
# domain works in both dev (where Telegram polls the prod webhook) and
# prod. Override via env if you ever serve audio from a separate host.
import os
PUBLIC_BASE_URL = os.environ.get("VAAANI_PUBLIC_URL", "https://brain.vaaani.in")


# =========================================================================
#  Normalized data types (transport-independent)
# =========================================================================

@dataclass
class IncomingMessage:
    kind: str               # 'telegram' | 'whatsapp'
    chat_id: str
    text: str               # may be empty when other fields populated
    username: str | None = None
    # Transport pre-downloads attachments to a temp path before calling
    # dispatch. The dispatcher stays transport-agnostic — it never knows
    # how to talk to Telegram's getFile or WhatsApp's media endpoint.
    attachment_path: str | None = None
    attachment_kind: str | None = None   # 'photo' | 'pdf' | 'document'
    attachment_mime: str | None = None
    attachment_name: str | None = None   # original filename if any


@dataclass
class OutgoingReply:
    kind: str               # echoes IncomingMessage.kind
    chat_id: str
    text: str = ""
    audio_url: str | None = None
    parse_mode: str = "Markdown"   # most transports support this or HTML
    silent: bool = False           # disable notification for noisy replies


# =========================================================================
#  Command grammar
# =========================================================================

HELP_TEXT = (
    "*Vaaani study assistant*\n\n"
    "Just type a question and I'll answer from your material.\n"
    "Or snap a photo of a textbook page / forward a PDF — I'll OCR + ingest it.\n\n"
    "Commands:\n"
    "/socratic on — assistant teaches by asking instead of telling\n"
    "/socratic off — back to direct answers\n"
    "/listen <doc> — narrate an ingested document\n"
    "/podcast <doc> — turn a document into a 2-host podcast\n"
    "/docs — list ingested documents\n"
    "/review — start spaced review (then reply 1/2/3/4 to grade)\n"
    "/explain <topic> — write an explanation, get a graph diff back\n"
    "/cancel — leave the current mode\n"
    "/unlink — disconnect this chat from your Vaaani account\n"
    "/help — show this message\n"
)

START_PROMPT = (
    "Welcome to *Vaaani*. To use this bot, link it to your account:\n\n"
    "1. Open https://brain.vaaani.in/app\n"
    "2. Sign in\n"
    "3. Open the Telegram link panel (Account → Link Telegram)\n"
    "4. Paste the code it gives you here, e.g. `vaaani-7a3f29b1`\n\n"
    "Once linked, just type a question. /help for the full command list."
)


# =========================================================================
#  Entry point
# =========================================================================

def handle_incoming(msg: IncomingMessage) -> list[OutgoingReply]:
    text = (msg.text or "").strip()

    # First touch from this chat — either link-code or onboarding.
    user_id = store.resolve_user(msg.kind, msg.chat_id)
    if user_id is None:
        return _handle_unlinked(msg, text)

    # Attachments are handled regardless of conversation mode — a photo
    # is always an ingest, never a review answer.
    if msg.attachment_path:
        return _handle_attachment(msg, user_id)

    # Already linked: route by current mode.
    state = store.get_state(msg.kind, msg.chat_id)
    mode = state.get("mode", "chat")

    # Universal commands work in any mode.
    if text.lower() in ("/start", "start"):
        return _reply(msg, f"Already linked. {HELP_TEXT}")
    if text.lower() == "/help":
        return _reply(msg, HELP_TEXT)
    if text.lower() == "/cancel":
        store.reset_state(msg.kind, msg.chat_id)
        return _reply(msg, "OK, back to chat mode.")
    if text.lower() == "/unlink":
        store.unlink_chat(msg.kind, msg.chat_id)
        return _reply(msg, "Unlinked. This chat is no longer connected to a Vaaani account.")

    if mode == "review":
        return _handle_review_turn(msg, user_id, text, state)
    if mode == "awaiting_explanation":
        return _handle_explanation_turn(msg, user_id, text, state)

    # mode == "chat" (or unrecognized): parse commands first.
    return _handle_chat_turn(msg, user_id, text)


# =========================================================================
#  Unlinked path
# =========================================================================

_CODE_RE = "vaaani-"


def _handle_unlinked(msg: IncomingMessage, text: str) -> list[OutgoingReply]:
    # Telegram's /start can carry a deep-link payload after a space:
    # "/start vaaani-7a3f29b1". Strip the command prefix.
    candidate = text
    low = candidate.lower()
    if low.startswith("/start "):
        candidate = candidate[len("/start "):].strip()
        low = candidate.lower()

    if low.startswith(_CODE_RE):
        user_id = store.consume_link_code(low, msg.kind)
        if user_id is None:
            return _reply(msg, "That code is invalid, expired, or already used. Mint a fresh one on the web and try again.")
        user = auth_service.get_user_by_id(user_id) or {}
        display = user.get("name") or user.get("email") or "your account"
        store.link_chat(msg.kind, msg.chat_id, user_id, username=msg.username)
        return _reply(
            msg,
            f"Linked to *{display}*. You can now type a question and I'll answer from your ingested material. /help for the full command list.",
        )

    # Anything else from an unlinked chat → onboarding.
    return _reply(msg, START_PROMPT)


# =========================================================================
#  Chat-mode commands
# =========================================================================

def _handle_chat_turn(msg: IncomingMessage, user_id: int, text: str) -> list[OutgoingReply]:
    low = text.lower()

    if low.startswith("/socratic"):
        rest = text[len("/socratic"):].strip().lower()
        if rest in ("on", "yes", "1"):
            store.set_socratic(msg.kind, msg.chat_id, True)
            return _reply(msg, "Socratic mode *on*. I'll teach by asking instead of telling.")
        if rest in ("off", "no", "0"):
            store.set_socratic(msg.kind, msg.chat_id, False)
            return _reply(msg, "Socratic mode *off*. Direct answers from now on.")
        current = "on" if store.get_socratic(msg.kind, msg.chat_id) else "off"
        return _reply(msg, f"Socratic mode is currently *{current}*. Use `/socratic on` or `/socratic off`.")

    if low.startswith("/docs"):
        return _list_docs(msg, user_id)

    if low.startswith("/listen"):
        return _make_audio(msg, text[len("/listen"):].strip(), mode="narration", user_id=user_id)
    if low.startswith("/podcast"):
        return _make_audio(msg, text[len("/podcast"):].strip(), mode="podcast", user_id=user_id)

    if low.startswith("/review"):
        return _start_review(msg, user_id)

    if low.startswith("/explain"):
        topic_hint = text[len("/explain"):].strip()
        return _start_explain(msg, topic_hint)

    if not text:
        return _reply(msg, "Send a question, or /help for commands.")

    # Plain text → RAG. Resolve socratic preference + call the chat
    # endpoint via the same internal function the web uses.
    return _rag_answer(msg, user_id, text)


def _rag_answer(msg: IncomingMessage, user_id: int, query: str) -> list[OutgoingReply]:
    """Run a single chat turn. Imports lazily to avoid pulling main's
    heavy globals (sentence-transformers, etc.) at module load."""
    try:
        from main import _run_intent, retriever  # type: ignore
        from llm import collect_stream
    except Exception as e:
        return _reply(msg, f"Backend not ready: {e}")

    socratic = store.get_socratic(msg.kind, msg.chat_id)
    try:
        answer = _run_chat_for_messenger(query, socratic=socratic, user_id=user_id)
    except Exception as e:
        return _reply(msg, f"Sorry — I hit an error answering: {e}")
    return _reply(msg, answer)


def _run_chat_for_messenger(query: str, *, socratic: bool, user_id: int) -> str:
    """One chat turn for the bot, via the SAME pipeline the web uses
    (_run_intent): intent routing, graph-RAG, Hermes, the relevance gate,
    and per-user privacy scoping all apply. Messengers just get the final
    string — no streaming, no structured output.

    (This used to hand-roll retrieval with signatures that had drifted —
    every call in it crashed on any bot question. Delegating keeps it from
    rotting again.)
    """
    from main import _allowed_paths, _run_intent  # lazy: avoid cycle at module load
    from auth import service as auth_service
    from memory import record_query

    user = None
    try:
        user = auth_service.get_user_by_id(user_id) if user_id else None
    except Exception:
        user = None

    result, _retrieval = _run_intent(
        query, False,
        socratic=socratic,
        user=user,
        allowed_paths=_allowed_paths(user),
    )
    answer = (result.answer or "").strip()

    if not answer:
        answer = "I couldn't draw a clear answer from your material this turn. Try rephrasing or /docs to see what's loaded."

    # Persist a query trace so memory + dashboards still work for bot turns.
    try:
        record_query(query, user_id=user_id)
    except Exception:
        pass

    return answer


# =========================================================================
#  /docs
# =========================================================================

def _allowed_names_for(user_id: int | None) -> set[str] | None:
    """Doc display-names the linked account may read (privacy scope)."""
    try:
        from main import _allowed_doc_names
        from auth import service as auth_service
        user = auth_service.get_user_by_id(user_id) if user_id else None
        return _allowed_doc_names(user)
    except Exception:
        return set()  # fail closed


def _list_docs(msg: IncomingMessage, user_id: int | None = None) -> list[OutgoingReply]:
    try:
        from audio import list_narratable_docs
        docs = list_narratable_docs()
        names = _allowed_names_for(user_id)
        if names is not None:
            docs = [d for d in docs if d.get("doc_name") in names]
    except Exception:
        docs = []
    if not docs:
        return _reply(msg, "No documents ingested yet. Upload one on https://brain.vaaani.in/app or via the Notion connector.")
    lines = ["*Ingested documents:*"]
    for d in docs[:20]:
        lines.append(f"• {d['doc_name']}  _{d['estimated_minutes']} min · {d['chunks']} chunks_")
    if len(docs) > 20:
        lines.append(f"… and {len(docs) - 20} more")
    return _reply(msg, "\n".join(lines))


# =========================================================================
#  /listen + /podcast
# =========================================================================

def _make_audio(msg: IncomingMessage, name: str, *, mode: str, user_id: int | None = None) -> list[OutgoingReply]:
    name = name.strip()
    if not name:
        return _reply(msg, f"Usage: `/{mode if mode == 'podcast' else 'listen'} <doc name>`. Use /docs to see options.")
    allowed = _allowed_names_for(user_id)
    if allowed is not None and name not in allowed:
        return _reply(msg, f"I don't have a document called *{name}*. Use /docs to see what's ingested.")
    try:
        from audio import narrate_doc, podcast_doc
        result = podcast_doc(name) if mode == "podcast" else narrate_doc(name)
    except KeyError:
        return _reply(msg, f"I don't have a document called *{name}*. Use /docs to see what's ingested.")
    except FileNotFoundError as e:
        return _reply(msg, f"Voice setup issue: {e}")
    except (ValueError, RuntimeError) as e:
        return _reply(msg, f"Audio generation failed: {e}")
    url = f"{PUBLIC_BASE_URL}/audio/file/{result.cache_hash}.mp3"
    minutes = int(result.duration_s // 60)
    seconds = int(result.duration_s % 60)
    label = "Podcast" if mode == "podcast" else "Narration"
    caption = f"{label} of *{result.doc_name}* — {minutes}:{seconds:02d}{' (cached)' if result.cached else ''}"
    return [OutgoingReply(kind=msg.kind, chat_id=msg.chat_id, text=caption, audio_url=url)]


# =========================================================================
#  /review
# =========================================================================

def _start_review(msg: IncomingMessage, user_id: int) -> list[OutgoingReply]:
    try:
        from adaptive import spaced
        item = spaced.next_review(user_id)
    except Exception as e:
        return _reply(msg, f"Review unavailable: {e}")
    if item is None:
        return _reply(msg, "Nothing due right now. Come back later — I'll surface concepts as their review window opens.")
    store.set_state(msg.kind, msg.chat_id, "review", item["node_id"])
    return _reply(msg, _format_review_card(item))


def _format_review_card(item: dict) -> str:
    head = f"*{item['display']}*"
    if item.get("type"):
        head += f" _({item['type']})_"
    body = item.get("prompt", "")
    if item["mode"] == "cloze":
        body = body.replace("_____", "*_____*")
    source = f"\n_From {item['source']}_" if item.get("source") else ""
    grade_line = "\nGrade by replying: *1*=again  *2*=hard  *3*=good  *4*=easy  ·  /reveal to show the answer  ·  /cancel to stop"
    return f"{head}\n\n{body}{source}{grade_line}"


def _handle_review_turn(msg: IncomingMessage, user_id: int, text: str, state: dict) -> list[OutgoingReply]:
    node_id = state.get("payload", "")
    low = text.lower().strip()

    # Show the canonical answer + the source sentence without grading.
    if low in ("/reveal", "reveal", "show", "answer"):
        try:
            from adaptive import spaced
            item = spaced.build_review_item(node_id, "")
        except Exception:
            item = None
        if not item:
            store.reset_state(msg.kind, msg.chat_id)
            return _reply(msg, "I lost the card — start over with /review.")
        body = item.get("answer") or item.get("description") or "(no canonical answer; grade based on your own recall)"
        return _reply(msg, f"*Answer:* {body}\n\nGrade: *1* again  *2* hard  *3* good  *4* easy")

    grade_map = {"1": "again", "2": "hard", "3": "good", "4": "easy",
                 "again": "again", "hard": "hard", "good": "good", "easy": "easy"}
    grade = grade_map.get(low)
    if not grade:
        # Anything else cancels review and routes back to chat.
        store.reset_state(msg.kind, msg.chat_id)
        return _handle_chat_turn(msg, user_id, text)

    try:
        from adaptive import spaced
        graded = spaced.grade_node(user_id, node_id, "", grade)
        nxt = spaced.next_review(user_id)
    except Exception as e:
        store.reset_state(msg.kind, msg.chat_id)
        return _reply(msg, f"Could not record grade: {e}")

    confirmation = f"Graded *{grade}*. Next review in ~{graded['interval_days']}d."
    if not nxt:
        store.reset_state(msg.kind, msg.chat_id)
        return _reply(msg, f"{confirmation}\n\nQueue empty for now.")
    store.set_state(msg.kind, msg.chat_id, "review", nxt["node_id"])
    return _reply(msg, f"{confirmation}\n\n{_format_review_card(nxt)}")


# =========================================================================
#  /explain (Feynman)
# =========================================================================

def _start_explain(msg: IncomingMessage, topic_hint: str) -> list[OutgoingReply]:
    try:
        from feynman import list_topics
        topics = list_topics()
    except Exception as e:
        return _reply(msg, f"Feynman unavailable: {e}")
    if not topics:
        return _reply(msg, "No graph yet — ingest a document first.")

    chosen = None
    hint = topic_hint.strip().lower()
    if hint:
        for t in topics:
            if t["id"].lower() == hint or t["display"].lower() == hint:
                chosen = t
                break
        if not chosen:
            for t in topics:
                if hint in t["display"].lower() or hint in t["id"].lower():
                    chosen = t
                    break
    if not chosen:
        # No hint or no match — list top topics and ask.
        sample = "\n".join(f"• {t['display']}" for t in topics[:8])
        return _reply(msg, f"Pick a topic — try `/explain {topics[0]['display']}`. Top topics:\n{sample}")

    store.set_state(msg.kind, msg.chat_id, "awaiting_explanation", chosen["id"])
    return _reply(msg, f"OK — explain *{chosen['display']}* in your own words. Two to five sentences usually surfaces the most gaps. /cancel to abandon.")


def _handle_explanation_turn(msg: IncomingMessage, user_id: int, text: str, state: dict) -> list[OutgoingReply]:
    if text.startswith("/"):
        # Command interrupts the wait — drop back to chat and re-dispatch.
        store.reset_state(msg.kind, msg.chat_id)
        return _handle_chat_turn(msg, user_id, text)
    if len(text) < 20:
        return _reply(msg, "Write at least a few sentences first — minimum 20 characters.")
    topic_id = state.get("payload", "")
    try:
        from feynman import diff_explanation
        result = diff_explanation(text, topic_id, k=2)
    except KeyError:
        store.reset_state(msg.kind, msg.chat_id)
        return _reply(msg, "I lost the topic — start again with /explain.")
    except (ValueError, RuntimeError) as e:
        store.reset_state(msg.kind, msg.chat_id)
        return _reply(msg, f"Diff failed: {e}")

    store.reset_state(msg.kind, msg.chat_id)
    return _reply(msg, _format_feynman_result(result.to_json()))


def _format_feynman_result(d: dict) -> str:
    head = f"*{d['topic_display']}* — coverage *{d['coverage_pct']}%*"
    summary = d.get("summary", "")
    covered = ", ".join(n["display"] for n in d.get("nodes_covered", [])[:8]) or "_(none)_"
    missed_list = d.get("nodes_missed", [])
    missed = ", ".join(n["display"] for n in missed_list[:8]) or "_(none)_"
    extras = ", ".join(d.get("student_extras", [])[:6])
    extras_line = f"\n\n_Mentions outside this 2-hop neighbourhood:_ {extras}" if extras else ""
    more_missed = f" _… and {len(missed_list) - 8} more_" if len(missed_list) > 8 else ""
    return (
        f"{head}\n\n{summary}\n\n"
        f"*Covered:* {covered}\n"
        f"*Missed:* {missed}{more_missed}{extras_line}\n\n"
        f"Tip: open the graph overlay on the web for the visual view."
    )


# =========================================================================
#  Attachment handling (photo OCR + PDF ingest)
# =========================================================================

def _handle_attachment(msg: IncomingMessage, user_id: int) -> list[OutgoingReply]:
    from pathlib import Path
    from . import attachments

    path = Path(msg.attachment_path or "")
    if not path.exists():
        return _reply(msg, "I lost the file before I could read it. Try sending again.")

    label = msg.username or f"chat-{msg.chat_id}"
    kind = (msg.attachment_kind or "").lower()

    try:
        if kind == "pdf" or attachments.is_pdf(msg.attachment_mime, msg.attachment_name):
            result = attachments.ingest_pdf(path, source_label=label, owner_user_id=user_id)
            return _reply(
                msg,
                f"Got the PDF *{msg.attachment_name or result['doc_filename']}*. "
                f"Added {result['chunks_added']} chunk{'s' if result['chunks_added'] != 1 else ''} "
                f"({result['total_chunks']} total in your corpus). "
                f"Ask me anything about it.",
            )
        if kind == "photo":
            result = attachments.ingest_photo(
                path, source_label=label, caption=msg.text or "", owner_user_id=user_id
            )
            if not result.get("ok"):
                if result.get("reason") == "too_little_text":
                    return _reply(
                        msg,
                        f"I could only read {result.get('chars', 0)} characters from that photo — likely too blurry, dim, or skewed. Try again with a flatter, brighter shot.",
                    )
                return _reply(msg, "I couldn't read that photo. Try a clearer shot.")
            return _reply(
                msg,
                f"Read *{result['words']} words* from your photo. "
                f"Added {result['chunks_added']} chunk{'s' if result['chunks_added'] != 1 else ''} "
                f"({result['total_chunks']} total in your corpus). "
                f"Ask me anything about it — or /review to fold it into spaced practice.",
            )
        # Unsupported document type — be specific so the student can retry.
        return _reply(
            msg,
            "I can read photos (snap a textbook page) and PDFs. That file type isn't supported yet — convert it or paste the text.",
        )
    except RuntimeError as e:
        return _reply(msg, f"Couldn't process that file: {e}")
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


# =========================================================================
#  Helpers
# =========================================================================

def _reply(msg: IncomingMessage, text: str, *, audio_url: str | None = None) -> list[OutgoingReply]:
    return [OutgoingReply(
        kind=msg.kind,
        chat_id=msg.chat_id,
        text=text,
        audio_url=audio_url,
    )]
