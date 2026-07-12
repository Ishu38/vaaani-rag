#!/usr/bin/env python3
"""Assemble the static frontend for Vercel (app.vaaani.in).

Every API call stays same-origin: Vercel rewrites proxy all backend paths
server-side to the GCP backend (api.vaaani.in), so the session cookie is
always first-party and can never be blocked. No client-side fetch shim needed.

    VAAANI_API_BASE=https://api.vaaani.in python deploy/build_vercel.py
    cd deploy/vercel_build && vercel --prod      # (Neil runs this)
"""
import os
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "deploy" / "vercel_build"
API = os.environ.get("VAAANI_API_BASE", "https://api.vaaani.in").rstrip("/")

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

# 1) all of site/ (assets + pages) at the root
shutil.copytree(ROOT / "site", OUT, dirs_exist_ok=True)
# 2) the two frontend/ pages, renamed to their clean routes
shutil.copy(ROOT / "frontend" / "index.html", OUT / "app.html")
shutil.copy(ROOT / "frontend" / "graph.html", OUT / "graph-view.html")

# 3) No _api.js shim — Vercel rewrites proxy all API calls same-origin,
#    so fetch('/chat', ...) stays relative and the cookie is first-party.

# Pages that MUST stay reachable without signing in: the marketing funnel, the
# auth pages themselves, the parental-consent + email-verify landing pages, and
# legal/PWA fallbacks. Everything else is a product surface and is gated.
PUBLIC_PAGES = {
    "index.html", "about.html", "contact.html", "integrations.html",
    "pricing.html", "login.html", "signup.html", "verify.html",
    "parental-consent.html", "privacy.html", "offline.html", "404.html",
    "forgot-password.html", "reset-password.html",
}

# Login-guard: hide the page until /auth/me confirms a session, then either
# reveal it or bounce anon visitors to /login (preserving where they headed).
# Only a DEFINITIVE signal logs someone out: a 200 saying user==null, or a
# 401/403. Anything else — a 5xx (backend restarting), a 429, a network error,
# or /auth/me taking >7s — is treated as a transient blip and REVEALS the page
# so a signed-in learner is never kicked to /login by a hiccup (data endpoints
# still 401 on their own, so revealing an empty shell to a true anon is safe).
GUARD = (
    "<script>(function(){"
    "var s=document.createElement('style');s.id='__vg';"
    "s.textContent='html{visibility:hidden!important}';"
    "document.documentElement.appendChild(s);"
    "var done=false;"
    "function show(){if(done)return;done=true;var e=document.getElementById('__vg');if(e)e.remove();}"
    "function go(){if(done)return;done=true;location.replace('/login?next='+encodeURIComponent(location.pathname+location.search));}"
    # safety net: never leave the learner staring at a blank (hidden) page if
    # /auth/me stalls — reveal after 7s (fail-open), don't redirect.
    "var t=setTimeout(show,7000);"
    "fetch('/auth/me')"
    ".then(function(r){"
    "if(r.ok){return r.json().then(function(d){clearTimeout(t);(d&&d.user)?show():go();});}"
    "clearTimeout(t);"
    "if(r.status===401||r.status===403){go();}else{show();}"
    "})"
    ".catch(function(){clearTimeout(t);show();});})();</script>"
)

n = 0
gated = 0
for f in OUT.rglob("*.html"):
    if f.name in PUBLIC_PAGES:
        continue
    t = f.read_text(encoding="utf-8", errors="ignore")
    if "</head>" in t and "__vg" not in t:
        f.write_text(t.replace("</head>", GUARD + "</head>", 1), encoding="utf-8")
        gated += 1
        n += 1

# 4) Vercel config — every API path is proxied server-side to the backend so
# all fetch() calls stay same-origin and the session cookie is always first-party.
# Prefixes that share a name with a static clean-URL page (/explore, /cognitive,
# /simulation) MUST only match sub-paths (/(.*)) so the bare path still serves
# the .html file. All other prefixes get both a root-level AND a sub-path rule.
_API_PREFIXES = [
    "auth", "audio", "chat", "cognitive", "explore", "feynman", "figures",
    "graph", "hermes", "ingest", "learning", "loop", "messenger",
    "simulation", "status", "youtube",
]
_SUBPATH_ONLY = {"cognitive", "explore", "simulation"}
_rewrites = []
for p in _API_PREFIXES:
    if p in _SUBPATH_ONLY:
        _rewrites.append(f'    {{ "source": "/{p}/(.*)", "destination": "{API}/{p}/$1" }}')
    else:
        _rewrites.append(f'    {{ "source": "/{p}", "destination": "{API}/{p}" }}')
        _rewrites.append(f'    {{ "source": "/{p}/(.*)", "destination": "{API}/{p}/$1" }}')
_rewrites_str = ",\n".join(_rewrites)
(OUT / "vercel.json").write_text(
    '{\n'
    '  "cleanUrls": true,\n'
    '  "trailingSlash": false,\n'
    '  "rewrites": [\n'
    f'{_rewrites_str}\n'
    '  ]\n'
    '}\n', encoding="utf-8"
)

pages = sum(1 for _ in OUT.rglob("*.html"))
print(f"assembled {pages} pages → {OUT}")
print(f"  login-guard on {gated} product pages ({len(PUBLIC_PAGES)} left public)")
print(f"  {len(_API_PREFIXES)} API prefixes proxied → {API}")
print(f"  deploy with:  cd {OUT}  &&  vercel --prod")
