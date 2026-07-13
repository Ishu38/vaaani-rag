# Deploy — Frontend on Vercel, Backend on Google Cloud (free during the 90-day trial)

**Architecture:** `app.vaaani.in` (Vercel, static frontend) talks to `api.vaaani.in`
(one Google Cloud VM running backend + model + all state). Both end in
`.vaaani.in`, so the login cookie is shared and auth "just works" — no cross-site
cookie problems, no proxying the slow AI responses through Vercel.

**Cost:** Vercel Hobby = free. The GCP VM is paid by your $300 / 90-day trial
credit (a small VM costs well under $300 for 90 days). After the trial the VM
starts billing (~₹3–4k/mo) — by then a school should be paying, or pause the VM.

**Trial note:** trial accounts can't get a GPU, so AI answers stay ~1 min (fine
for a demo). Faster answers need a later paid upgrade (credit carries over).

Commands marked **🔑** touch YOUR accounts — run them yourself (type
`! <command>` in the session, or in your own terminal). Everything else I already
prepared in the repo.

---

## Phase 0 — New GCP project + a domain you control

```bash
🔑 gcloud auth login
🔑 gcloud projects create vaaani-app-<unique> --name="Vaaani App"
🔑 gcloud config set project vaaani-app-<unique>
🔑 gcloud billing accounts list
🔑 gcloud billing projects link vaaani-app-<unique> --billing-account=XXXXXX-XXXXXX-XXXXXX
gcloud services enable compute.googleapis.com
```
You already own `vaaani.in`. You'll point two records at the end (Phase 5).

## Phase 1 — Create the one VM

```bash
🔑 gcloud compute instances create vaaani-vm \
    --zone=asia-south1-a \
    --machine-type=e2-standard-2 \
    --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
    --boot-disk-size=30GB \
    --tags=http-server,https-server
🔑 gcloud compute firewall-rules create allow-web \
    --allow=tcp:80,tcp:443 --target-tags=http-server,https-server
🔑 gcloud compute ssh vaaani-vm --zone=asia-south1-a
```
`e2-standard-2` = 2 vCPU / 8 GB — enough for the 4B model (q8 ~4 GB) + backend +
state. Ubuntu 24.04 ships Python 3.12 (matches your dev env).

## Phase 2 — Get the code + model onto the VM

Easiest: push the repo to a private GitHub repo, then on the VM:
```bash
sudo apt-get update && sudo apt-get install -y python3.12-venv build-essential \
    libopenblas0 git tesseract-ocr ffmpeg
git clone https://github.com/<you>/rag-assistant.git ~/rag-assistant     # or scp it
git clone https://github.com/<you>/vaaani-model.git ~/vaaani-model       # engine + config
```
Copy the model GGUFs (from your laptop, ~4 GB) — via a GCS bucket is simplest:
```bash
🔑 gsutil mb gs://vaaani-models-<unique>                          # on laptop
🔑 gsutil cp ~/vaaani-model/gguf/vaaani-base4-q8_0.gguf gs://vaaani-models-<unique>/
# on the VM:
gsutil cp gs://vaaani-models-<unique>/vaaani-base4-q8_0.gguf ~/vaaani-model/gguf/
```

Build the two Python envs on the VM:
```bash
cd ~/vaaani-model && python3 -m venv .venv && .venv/bin/pip install -U pip \
    && .venv/bin/pip install "llama-cpp-python>=0.3.9[server]"
cd ~/rag-assistant && python3 -m venv .venv && .venv/bin/pip install -U pip \
    && .venv/bin/pip install -r requirements.txt
```

## Phase 3 — Run engine + backend as services (survive reboots)

`/etc/systemd/system/vaaani-engine.service`:
```ini
[Unit]
Description=Vaaani model engine
After=network.target
[Service]
User=%i
WorkingDirectory=/home/<you>/vaaani-model
ExecStart=/home/<you>/vaaani-model/.venv/bin/python -m llama_cpp.server --config_file config_cpu.json
Restart=always
[Install]
WantedBy=multi-user.target
```
`/etc/systemd/system/vaaani-backend.service`:
```ini
[Unit]
Description=Vaaani backend
After=vaaani-engine.service
[Service]
User=<you>
WorkingDirectory=/home/<you>/rag-assistant/backend
Environment=JWT_SECRET=<run: openssl rand -hex 48>
Environment=COOKIE_SECURE=1
Environment=COOKIE_DOMAIN=.vaaani.in
Environment=COOKIE_SAMESITE=lax
Environment=CORS_ORIGINS=https://app.vaaani.in
Environment=VAAANI_LLM_BASE_URL=http://127.0.0.1:8011
ExecStart=/home/<you>/rag-assistant/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8765
Restart=always
[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vaaani-engine vaaani-backend
```
Then seed the Core Library once (so the graph isn't empty):
```bash
cd ~/rag-assistant/backend && ../.venv/bin/python library/build_library.py
../.venv/bin/python library/build_library_graph.py
```

## Phase 4 — Free HTTPS on api.vaaani.in (Caddy auto-cert)

```bash
sudo apt-get install -y caddy
echo 'api.vaaani.in {
    reverse_proxy 127.0.0.1:8765
}' | sudo tee /etc/caddy/Caddyfile
sudo systemctl restart caddy
```
Caddy fetches a free Let's Encrypt certificate automatically once DNS points here.

## Phase 5 — DNS (in your domain registrar / Cloudflare)

```
🔑 A     api.vaaani.in   → <the VM's external IP>   (gcloud compute instances describe vaaani-vm --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
🔑 CNAME app.vaaani.in   → cname.vercel-dns.com     (Vercel shows the exact target)
```

## Phase 6 — Deploy the frontend to Vercel

I already built the static bundle. On your laptop:
```bash
cd ~/Desktop/rag-assistant
VAAANI_API_BASE=https://api.vaaani.in python deploy/build_vercel.py
cd deploy/vercel_build
🔑 npx vercel --prod          # first run: log in, accept defaults
```
Then in the Vercel dashboard, add the domain **app.vaaani.in** to the project
(it prints the CNAME for Phase 5).

## Phase 7 — Smoke test

- `https://api.vaaani.in/` → 200 (backend up, HTTPS green).
- `https://app.vaaani.in/login` → sign in → refresh → still logged in (shared cookie works).
- On a **phone**, `https://app.vaaani.in/explore` and `/feel` → camera + mic now work (HTTPS).
- Ask a question in Chat → grounded answer (~1 min on CPU — expected).

## Rollback / pause (stop the meter)

```bash
🔑 gcloud compute instances stop vaaani-vm --zone=asia-south1-a   # stops billing for compute
```
Frontend on Vercel stays free and up regardless.

---

### What I changed in the app to make this work (already done, verified locally)
- `config.py` + `auth/security.py`: `COOKIE_DOMAIN` / `COOKIE_SAMESITE` so the
  session is shared across `app.` and `api.` subdomains.
- `main.py`: CORS now allows the exact `CORS_ORIGINS` **with credentials** in prod
  (permissive locally). Defaults unchanged → local dev still works.
- `deploy/build_vercel.py`: assembles the Vercel bundle + injects the API shim
  so every relative call goes to `api.vaaani.in` with the shared cookie.
