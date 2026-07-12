---
title: Vaaani
emoji: 🌉
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Vaaani

Vaaani RAG — graph-RAG study assistant with narration, Feynman-diff, spaced review and
the Root-Bridge tutor. Runs CPU-only; the LLM is served by the separate Vaaani engine
Space and reached over `VAAANI_LLM_BASE_URL`.

**Space Variables to set (Settings → Variables and secrets):**
- `VAAANI_LLM_BASE_URL` = `https://shaankar39-vaaani-flagship.hf.space`
- `VAAANI_LLM_MODEL` = `vaaani-base`
- `VAAANI_FLAGSHIP_MODEL` = `vaaani-flagship`
- `JWT_SECRET` = (a long random string — secret)

Storage is ephemeral (Option A): `data/` is seeded into the image; signups/ingests reset
on restart. Switch to HF persistent storage for production.
