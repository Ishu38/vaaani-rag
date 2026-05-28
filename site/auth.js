/* =========================================================
   Shared helpers for auth + dashboard pages
   ========================================================= */

const Auth = {
  /** Read JSON or return null on any error. */
  async json(url, opts = {}) {
    const r = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
      credentials: 'same-origin',
      ...opts,
    });
    const text = await r.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { /* keep null */ }
    return { ok: r.ok, status: r.status, data };
  },

  async me() {
    const r = await this.json('/auth/me');
    return r.data && r.data.user ? r.data.user : null;
  },

  async googleConfigured() {
    const r = await this.json('/auth/google/configured');
    return !!(r.data && r.data.configured);
  },

  async githubConfigured() {
    const r = await this.json('/auth/github/configured');
    return !!(r.data && r.data.configured);
  },

  /** Wire a Google Sign-in button: hide unless configured. */
  async wireGoogleButton(btn) {
    if (!btn) return;
    const on = await this.googleConfigured();
    if (!on) { btn.style.display = 'none'; return; }
    btn.addEventListener('click', () => { window.location = '/auth/google/start'; });
  },

  /** Wire a GitHub Sign-in button: hide unless configured. */
  async wireGithubButton(btn) {
    if (!btn) return;
    const on = await this.githubConfigured();
    if (!on) { btn.style.display = 'none'; return; }
    btn.addEventListener('click', () => { window.location = '/auth/github/start'; });
  },

  setStatus(el, kind, message) {
    if (!el) return;
    el.className = `status-msg ${kind}`;
    el.textContent = message;
    el.style.display = message ? 'block' : 'none';
  },
};

/** Show user name / signed-in state in any nav with #navAuth. Falls back to "Sign in". */
async function paintNavAuth() {
  const slot = document.getElementById('navAuth');
  const user = await Auth.me();
  if (slot) {
    if (user) {
      slot.innerHTML = `<a class="btn btn-primary" href="/account">${escapeHtml(user.name || user.email)} <span class="arr">→</span></a>`;
    } else {
      slot.innerHTML = `<a class="btn btn-ghost" href="/login">Sign in</a> <a class="btn btn-primary" href="/signup">Get started <span class="arr">→</span></a>`;
    }
  }
  paintAuthGatedSurfaces(!!user);
}

/**
 * Hide app-only surfaces from signed-out visitors:
 *   - footer "Product" column links (Chat / Knowledge map / Status) that
 *     otherwise advertise gated routes to anonymous users
 *   - the "Launch the assistant" CTA, relabelled to "Open your chat" for
 *     signed-in users (no point inviting them to launch what they're in)
 * Any element tagged [data-auth="in"] is shown only when signed in;
 * [data-auth="out"] is shown only when signed out.
 */
function paintAuthGatedSurfaces(signedIn) {
  document.querySelectorAll('[data-auth="in"]').forEach((el) => {
    el.style.display = signedIn ? '' : 'none';
  });
  document.querySelectorAll('[data-auth="out"]').forEach((el) => {
    el.style.display = signedIn ? 'none' : '';
  });
  document.querySelectorAll('[data-auth-label-in]').forEach((el) => {
    if (signedIn) {
      const label = el.getAttribute('data-auth-label-in');
      if (label) el.textContent = label;
    }
  });
}

function escapeHtml(s) {
  return String(s || '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

document.addEventListener('DOMContentLoaded', () => paintNavAuth());
