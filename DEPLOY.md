# Deploying Vaaani Graph-RAG on a Hostinger VPS

This guide assumes a fresh Hostinger KVM VPS running **Ubuntu 22.04 / 24.04**.
You'll end with: a systemd-managed uvicorn on `127.0.0.1:8765`, nginx in front
on `:80/:443`, and HTTPS via Let's Encrypt.

## 0. What's deployed

| URL                | Serves                                       |
|--------------------|----------------------------------------------|
| `/`                | Animated landing page (`site/index.html`)    |
| `/app`             | Chat assistant (`frontend/index.html`)       |
| `/graph-view`      | Knowledge constellation                      |
| `/site/*`          | Static landing assets (css, js)              |
| `/chat`, `/ingest`, `/status`, `/graph` | JSON API                  |
| `/docs`            | FastAPI Swagger                              |

## 1. Provision

In Hostinger control panel:
- VPS plan ≥ **KVM 2** (2 vCPU, 8 GB RAM). Embedding model + DeepSeek client
  comfortably fit; smaller plans will swap.
- OS: Ubuntu 24.04 LTS.
- Set up SSH key access from your machine.

```bash
ssh root@YOUR_VPS_IP
adduser --disabled-password --gecos "" vaaani
usermod -aG sudo vaaani
rsync -a ~/.ssh /home/vaaani/ && chown -R vaaani:vaaani /home/vaaani/.ssh
su - vaaani
```

## 2. System dependencies

```bash
sudo apt update && sudo apt install -y \
  python3 python3-venv python3-pip \
  libopenblas-dev \
  build-essential pkg-config \
  nginx certbot python3-certbot-nginx \
  ufw git
```

`libopenblas-dev` is what made `turbovec` import on the dev box; the system
package provides the same `cblas_sgemm` symbol without LD_PRELOAD hacks.

## 3. Get the code

```bash
mkdir -p /home/vaaani/apps && cd /home/vaaani/apps
git clone <your-repo-or-rsync-the-folder> rag-assistant
# or: scp -r /home/ishu/Desktop/rag-assistant vaaani@VPS:/home/vaaani/apps/
cd rag-assistant
```

## 4. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Environment variables

Create `/home/vaaani/apps/rag-assistant/.env` (NOT checked into git):

```bash
# --- core LLM ---
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com

# --- auth ---
# Generate once with: python3 -c "import secrets; print(secrets.token_urlsafe(64))"
JWT_SECRET=long-random-string
COOKIE_SECURE=1                       # set to 1 because nginx terminates TLS
APP_BASE_URL=https://vaaani.in

# --- Google sign-in (optional; button stays hidden until both set) ---
# 1) Console: https://console.cloud.google.com/apis/credentials
# 2) Create an OAuth client ID (Web application)
# 3) Authorized redirect URI: https://vaaani.in/auth/google/callback
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=https://vaaani.in/auth/google/callback

# --- SMTP for verification emails (optional; falls back to log-only in dev) ---
# Hostinger provides per-domain SMTP; or use Resend/SendGrid/Postmark.
SMTP_HOST=smtp.hostinger.com
SMTP_PORT=587
SMTP_USER=no-reply@vaaani.in
SMTP_PASS=
SMTP_FROM=no-reply@vaaani.in
SMTP_USE_TLS=1

# --- SMS / OTP (optional; falls back to log-only) ---
# India: MSG91 (needs DLT-registered template, fastest local delivery)
SMS_PROVIDER=msg91
MSG91_AUTH_KEY=
MSG91_SENDER_ID=VAAANI
MSG91_TEMPLATE_ID=

# International: Twilio (no DLT hassle)
# SMS_PROVIDER=twilio
# TWILIO_SID=
# TWILIO_TOKEN=
# TWILIO_FROM=+1xxxxxxxxxx
```

Lock it down: `chmod 600 .env`.

### Provider notes

| Capability   | Without config                              | With config                                 |
|--------------|---------------------------------------------|---------------------------------------------|
| Email verify | Link printed to `journalctl -u vaaani`      | Real email via SMTP                         |
| Phone OTP    | OTP printed to `journalctl -u vaaani`       | SMS via MSG91 (India) or Twilio (global)    |
| Google SSO   | Button hidden on signup/login pages         | Button visible, full OAuth code flow        |

The site never shows a "fake" button — features that lack credentials are hidden, not stubbed.

## 6. systemd service

Create `/etc/systemd/system/vaaani.service`:

```ini
[Unit]
Description=Vaaani Graph-RAG Assistant
After=network.target

[Service]
Type=exec
User=vaaani
Group=vaaani
WorkingDirectory=/home/vaaani/apps/rag-assistant/backend
EnvironmentFile=/home/vaaani/apps/rag-assistant/.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/vaaani/apps/rag-assistant/.venv/bin/uvicorn main:app \
  --host 127.0.0.1 --port 8765 \
  --workers 1 --timeout-keep-alive 30
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vaaani
sudo systemctl status vaaani
curl -sS http://127.0.0.1:8765/status
```

## 7. nginx reverse proxy

Create `/etc/nginx/sites-available/vaaani`:

```nginx
server {
    listen 80;
    server_name vaaani.in www.vaaani.in;  # adjust to your domain

    client_max_body_size 50M;  # allow PDF uploads

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 180s;   # long enough for DeepSeek calls during ingest
        proxy_connect_timeout 30s;
        proxy_buffering off;       # avoid stalling streamed responses
    }
}
```

Enable and reload:
```bash
sudo ln -s /etc/nginx/sites-available/vaaani /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

## 8. HTTPS with Let's Encrypt

Point DNS A records for `vaaani.in` and `www.vaaani.in` at the VPS IP first,
then:
```bash
sudo certbot --nginx -d vaaani.in -d www.vaaani.in
```
Certbot will rewrite the nginx config to add a `443` server block and a
permanent redirect from `80`.

## 9. Firewall

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw --force enable
```

## 10. Updates

```bash
cd /home/vaaani/apps/rag-assistant
git pull            # or rsync the new files
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart vaaani
```

## 11. Logs and ops

| Action                  | Command                                  |
|-------------------------|------------------------------------------|
| App logs (tail, follow) | `journalctl -u vaaani -f`                |
| nginx access            | `sudo tail -f /var/log/nginx/access.log` |
| nginx error             | `sudo tail -f /var/log/nginx/error.log`  |
| Restart app             | `sudo systemctl restart vaaani`          |
| Stop app                | `sudo systemctl stop vaaani`             |
| Disk used by index      | `du -sh data/`                           |

## 12. Common pitfalls

- **Long ingest stalls nginx**: solved by `proxy_read_timeout 180s` above.
  If you ingest very large PDFs and still hit timeouts, bump it further or
  move ingest behind a background worker.
- **413 Request Entity Too Large on PDF upload**: `client_max_body_size` in
  the nginx server block (already set to 50M above).
- **DeepSeek 401 / empty replies**: env var not loaded — check
  `EnvironmentFile=` in the systemd unit and `cat /proc/$(pgrep -f uvicorn)/environ | tr '\0' '\n' | grep DEEPSEEK`.
- **turbovec ImportError on `cblas_sgemm`**: install `libopenblas-dev`
  (step 2). On Hostinger's stock Ubuntu image that's all you need —
  no LD_PRELOAD required.
