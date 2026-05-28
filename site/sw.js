/* Vaaani PWA service worker.
   - Static shell: stale-while-revalidate.
   - Navigations: network-first with /offline.html fallback.
   - API calls (/chat, /ingest, /auth/*, /status, /admin/*, /audio/narrate,
     /feynman/*, /messenger/*, /youtube/*, /hermes/*): never intercepted —
     they must hit the network so auth cookies, streaming, and POSTs work.
   Bump CACHE_VERSION whenever shell assets change so old clients refresh. */
const CACHE_VERSION = 'vaaani-v1-20260527';
const SHELL_CACHE = `${CACHE_VERSION}-shell`;
const RUNTIME_CACHE = `${CACHE_VERSION}-runtime`;

const SHELL_ASSETS = [
  '/',
  '/app',
  '/offline.html',
  '/style.css',
  '/main.js',
  '/page.css',
  '/page.js',
  '/auth.css',
  '/auth.js',
  '/manifest.webmanifest',
  '/icon-192.png',
  '/icon-512.png',
  '/apple-touch-icon.png',
  '/favicon-32.png',
];

/* Path prefixes that must always go to the network — never cached, never
   intercepted with a cached response. Streaming and auth depend on this. */
const NETWORK_ONLY_PREFIXES = [
  '/chat', '/ingest', '/auth', '/status', '/admin',
  '/audio/narrate', '/audio/podcast', '/feynman',
  '/messenger', '/youtube', '/hermes', '/learning',
  '/figures', '/docs', '/openapi.json',
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL_CACHE);
    /* addAll is atomic — if any fetch fails, none are cached. Use individual
       puts so a single missing asset doesn't abort the whole install. */
    await Promise.all(SHELL_ASSETS.map(async (url) => {
      try {
        const resp = await fetch(url, { credentials: 'same-origin' });
        if (resp.ok) await cache.put(url, resp.clone());
      } catch (_) { /* swallow — shell asset will retry on next navigation */ }
    }));
    self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter(k => !k.startsWith(CACHE_VERSION)).map(k => caches.delete(k)));
    await self.clients.claim();
  })());
});

function isNetworkOnly(url) {
  if (url.origin !== self.location.origin) return false;
  return NETWORK_ONLY_PREFIXES.some(p => url.pathname === p || url.pathname.startsWith(p + '/'));
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (isNetworkOnly(url)) return;

  /* Navigation requests → network-first, fall back to cached /app, then /offline.html. */
  if (req.mode === 'navigate') {
    event.respondWith((async () => {
      try {
        const fresh = await fetch(req);
        const cache = await caches.open(RUNTIME_CACHE);
        cache.put(req, fresh.clone()).catch(() => {});
        return fresh;
      } catch (_) {
        const cached = await caches.match(req) || await caches.match('/app') || await caches.match('/');
        return cached || caches.match('/offline.html');
      }
    })());
    return;
  }

  /* Same-origin static assets → stale-while-revalidate. */
  if (url.origin === self.location.origin) {
    event.respondWith((async () => {
      const cache = await caches.open(RUNTIME_CACHE);
      const cached = await cache.match(req);
      const network = fetch(req).then(resp => {
        if (resp && resp.ok) cache.put(req, resp.clone()).catch(() => {});
        return resp;
      }).catch(() => cached);
      return cached || network;
    })());
    return;
  }

  /* Cross-origin (fonts, CDN katex/gsap) → cache opaque responses too. */
  event.respondWith((async () => {
    const cache = await caches.open(RUNTIME_CACHE);
    const cached = await cache.match(req);
    const network = fetch(req).then(resp => {
      if (resp) cache.put(req, resp.clone()).catch(() => {});
      return resp;
    }).catch(() => cached);
    return cached || network;
  })());
});

self.addEventListener('message', (event) => {
  if (event.data === 'SKIP_WAITING') self.skipWaiting();
});
