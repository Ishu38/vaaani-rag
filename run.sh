#!/usr/bin/env bash
# Launch the RAG assistant.
# - Portable BLAS preload: probes the laptop's miniconda env and common
#   Linux system paths so turbovec's _turbovec.abi3.so can resolve cblas_sgemm
#   the same way on dev + VPS.
# - USE_TF=0 keeps the broken miniconda tensorflow 2.21 out of the import path.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
for candidate in \
  /home/ishu/miniconda3/envs/aligner/lib/libopenblas.so.0 \
  /usr/lib/x86_64-linux-gnu/libopenblas.so.0 \
  /usr/lib/libopenblas.so.0 \
  ; do
  if [ -f "$candidate" ]; then
    export LD_PRELOAD="${LD_PRELOAD:-}:$candidate"
    break
  fi
done
export USE_TF=0
export USE_FLAX=0
# Prefer the project venv if present (prod: /opt/vaaani-rag/.venv).
if [ -d "$HERE/.venv/bin" ]; then
  export PATH="$HERE/.venv/bin:$PATH"
fi
# JWT secret (random per-restart in dev; set a stable value in production env)
if [ -z "${JWT_SECRET:-}" ]; then
  export JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(64))')"
fi

# ---- GitHub OAuth -----------------------------------------------------
# Production: values come from systemd EnvironmentFile (/etc/vaaani-rag/env).
# Local dev: set them in a gitignored .env you source manually, or export
# them in your shell before launching this script. DO NOT hardcode the
# client secret in this file — it lives in the repo.
#   - Homepage URL:           http://127.0.0.1:8765
#   - Authorization callback: http://127.0.0.1:8765/auth/github/callback
# -----------------------------------------------------------------------

# ---- SMTP (real email delivery) ---------------------------------------
# Without SMTP_HOST set, get_sender() falls back to ConsoleEmailSender —
# verification links print to this log instead of being mailed. To send
# real mail, uncomment + fill in below (Gmail example; create an App
# Password at myaccount.google.com → Security → App passwords):
#
# export SMTP_HOST=smtp.gmail.com
# export SMTP_PORT=587
# export SMTP_USER=roychinu45@gmail.com
# export SMTP_PASS=xxxx-xxxx-xxxx-xxxx          # 16-char Google App Password
# export SMTP_FROM='Vaaani <roychinu45@gmail.com>'
# export SMTP_USE_TLS=1
# -----------------------------------------------------------------------
cd "$HERE/backend"
PORT="${PORT:-8765}"
exec uvicorn main:app --host 127.0.0.1 --port "$PORT" "$@"
