# Deploy Vaaani RAG alongside Vaani CAVP on the same VPS

This is the **shared-VPS** path. Both products run on the same Hostinger
machine, Cloudflare Tunnel multiplexes hostnames to localhost ports, no
nginx and no Let's Encrypt (Cloudflare handles HTTPS at the edge).

> Different from `DEPLOY.md` in the project root — that one is for a fresh
> standalone VPS with nginx + Let's Encrypt at the apex. Use this file when
> you're co-tenanting on the Vaani CAVP machine.

## Topology you'll end up with

```
Internet
   │
   ▼
Cloudflare edge (HTTPS)
   │
   ▼  (Cloudflare Tunnel)
VPS
   ├── 127.0.0.1:3001  → contrastive-voice-profiling-server-1   (Vaani CAVP gateway)
   ├── 127.0.0.1:8766  → vaaani-rag systemd unit                (Vaaani RAG — NEW)
   └── docker: mongo, engine, etc. (CAVP-internal, not exposed)

Hostnames at the edge:
   api.vaaani.in   → :3001   (existing)
   app.vaaani.in   → Cloudflare Pages SPA (existing)
   vaaani.in       → Cloudflare Pages agency site (existing)
   brain.vaaani.in → :8766   (NEW — Vaaani RAG)
```

## Prereqs to confirm on the VPS

```bash
# 1. Free RAM ≥ 2 GB (sentence-transformers + Python + headroom)
free -h
# 2. Free disk ≥ 5 GB
df -h /
# 3. Cloudflared running and managing api.vaaani.in already
sudo systemctl status cloudflared
cat /etc/cloudflared/config.yml | head -20
# 4. Python 3.10+ available
python3 --version
```

If RAM is tight on the existing box (≤4 GB total), consider upgrading the
VPS tier first — running both products on 2 GB will swap and feel awful.

## 1. System packages (one-time)

```bash
sudo apt update && sudo apt install -y \
  python3 python3-venv python3-pip \
  libopenblas-dev \
  build-essential pkg-config
```

`libopenblas-dev` exports `cblas_sgemm` at `/usr/lib/x86_64-linux-gnu/libopenblas.so.0`,
which the portable BLAS probe in `run.sh` picks up automatically.

## 2. Service user + directories

Keep `vaaani` separate from whatever user runs Vaani CAVP, so a misbehaving
process in one product can't write into the other.

```bash
sudo adduser --system --group --home /opt/vaaani-rag vaaani
sudo mkdir -p /opt/vaaani-rag /etc/vaaani-rag
sudo chown -R vaaani:vaaani /opt/vaaani-rag
sudo chmod 700 /etc/vaaani-rag        # secrets dir, root-readable only
```

## 3. Push code from laptop to VPS

From your laptop (`/home/ishu/Desktop/rag-assistant`):

```bash
rsync -av --delete \
  --exclude='__pycache__' \
  --exclude='node_modules' \
  --exclude='data/index.tq' \
  --exclude='data/users.db' \
  --exclude='data/memory.json' \
  --exclude='data/raw' \
  --exclude='.venv' \
  --exclude='*.pyc' \
  /home/ishu/Desktop/rag-assistant/ \
  user@VPS_IP:/tmp/rag-assistant/

ssh user@VPS_IP "sudo rsync -av --chown=vaaani:vaaani /tmp/rag-assistant/ /opt/vaaani-rag/ && rm -rf /tmp/rag-assistant"
```

The excludes keep your laptop's local SQLite + vector index out of prod —
prod gets a fresh DB the first time it boots.

## 4. Python venv (as the vaaani user, on the VPS)

```bash
sudo -u vaaani bash <<'EOF'
cd /opt/vaaani-rag
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
EOF
```

If the dev `run.sh` calls `uvicorn` directly (not `.venv/bin/uvicorn`),
update the unit file's `ExecStart` to point at the venv binary — see
section 6 below.

## 5. Secrets file

```bash
sudo cp /opt/vaaani-rag/deploy/vaaani-rag.env.example /etc/vaaani-rag/env
sudo chmod 600 /etc/vaaani-rag/env
sudo nano /etc/vaaani-rag/env          # fill in the real values
```

Required at minimum: `DEEPSEEK_API_KEY`, `JWT_SECRET`, and the GitHub OAuth
trio (already created on your account — see section 8 for the callback
URL update).

## 6. systemd unit

```bash
# Edit deploy/vaaani-rag.service if your venv path differs from /opt/vaaani-rag.
# The default ExecStart=/opt/vaaani-rag/run.sh works if run.sh can find a
# system-wide uvicorn — otherwise swap to:
#   ExecStart=/opt/vaaani-rag/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8766

sudo cp /opt/vaaani-rag/deploy/vaaani-rag.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vaaani-rag
sudo systemctl status vaaani-rag        # should be 'active (running)'
curl -sS http://127.0.0.1:8766/status   # should return JSON
```

Tail logs while it warms up (sentence-transformers downloads the model on
first run, ~80 MB):
```bash
journalctl -u vaaani-rag -f
```

## 7. Cloudflare Tunnel — add the hostname

```bash
sudo nano /etc/cloudflared/config.yml
# Insert the ingress rule from deploy/cloudflared-ingress.snippet.yml
# BEFORE the catch-all service: http_status:404 line.

# Route the DNS record (one-time, idempotent):
sudo cloudflared tunnel route dns <tunnel-name-or-uuid> brain.vaaani.in

# Reload the tunnel so it picks up the new ingress rule:
sudo systemctl reload cloudflared
sudo systemctl status cloudflared
```

After ~30 seconds, `https://brain.vaaani.in` resolves and serves the
landing page. `dig brain.vaaani.in` should show a Cloudflare IP.

## 8. Update GitHub OAuth callback

GitHub OAuth Apps allow multiple callback URLs only if you list them on the
"Authorization callback URL" field — but the standard OAuth App only takes
one. So either:

**Option A (cleanest):** Create a second OAuth App on GitHub purely for
production, with callback `https://brain.vaaani.in/auth/github/callback`.
Put its Client ID + Secret in `/etc/vaaani-rag/env` on the VPS. Keep the
dev OAuth App's creds in your laptop `run.sh` unchanged.

**Option B (one app, two callbacks via wildcard isn't supported):**
Change the single OAuth App's callback to the prod URL, accept that local
dev sign-in breaks until you swap it back. Not recommended.

Take **option A** — two apps, two sets of creds, never collide.

## 9. Smoke-test from outside the VPS

```bash
# From your laptop:
curl -sS https://brain.vaaani.in/status
curl -sS https://brain.vaaani.in/auth/github/configured   # {"configured": true}
```

Open `https://brain.vaaani.in/login` in a browser, click "Continue with
GitHub", and confirm the full round-trip works. Errors land in
`journalctl -u vaaani-rag -f`.

## 10. Updates later

```bash
# From laptop:
rsync -av --delete \
  --exclude='__pycache__' --exclude='.venv' --exclude='data/' \
  /home/ishu/Desktop/rag-assistant/ \
  user@VPS_IP:/tmp/rag-assistant/

ssh user@VPS_IP <<'EOF'
sudo rsync -av --chown=vaaani:vaaani --exclude='data/' /tmp/rag-assistant/ /opt/vaaani-rag/
sudo -u vaaani /opt/vaaani-rag/.venv/bin/pip install -r /opt/vaaani-rag/requirements.txt
sudo systemctl restart vaaani-rag
journalctl -u vaaani-rag -n 50
EOF
```

## Ops cheat-sheet

| Action                  | Command                                       |
|-------------------------|-----------------------------------------------|
| Tail app logs           | `journalctl -u vaaani-rag -f`                 |
| Restart                 | `sudo systemctl restart vaaani-rag`           |
| Tunnel status           | `sudo systemctl status cloudflared`           |
| Tunnel logs             | `journalctl -u cloudflared -f`                |
| Confirm port binding    | `sudo ss -tlnp \| grep 8766`                  |
| Disk used by index      | `sudo du -sh /opt/vaaani-rag/data/`           |
| Edit secrets            | `sudo nano /etc/vaaani-rag/env && sudo systemctl restart vaaani-rag` |
| Rotate JWT (logs all users out) | `sudo nano /etc/vaaani-rag/env` → change `JWT_SECRET` → restart |
