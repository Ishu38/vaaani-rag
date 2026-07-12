/* Vaaani service worker — DELIBERATELY NON-CACHING (2026-07-07).
 *
 * History: the previous caching worker used a cache-first strategy for
 * cross-origin GETs. Because the _api.js shim sends /auth/me to the
 * cross-origin backend (api.vaaani.in), that response got cached and served
 * stale — pinning a logged-out {user:null} and repeatedly breaking sign-in
 * across app.vaaani.in <-> api.vaaani.in.
 *
 * This worker caches NOTHING. On activate it purges every cache this origin
 * ever created (clearing any poisoned entry on existing clients automatically,
 * with no manual "clear site data" needed), then lets every request go
 * straight to the network. A trivial fetch handler is kept only so the app
 * stays an installable PWA — it never returns a cached response.
 *
 * The app is backed by a live LLM API and was never usefully offline, so there
 * is no meaningful loss, and this eliminates the stale-auth failure mode for good.
 */
const SW_VERSION = 'vaaani-nocache-20260707';

self.addEventListener('install', () => {
  // Take over as soon as possible.
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    // Delete ALL caches (any version) — including the poisoned /auth/me entry.
    const keys = await caches.keys();
    await Promise.all(keys.map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

// Pass-through: no respondWith(), so the browser performs its normal network
// fetch for every request. Present solely to satisfy PWA installability.
self.addEventListener('fetch', () => { /* intentionally no-op */ });

self.addEventListener('message', (event) => {
  if (event.data === 'SKIP_WAITING') self.skipWaiting();
});
