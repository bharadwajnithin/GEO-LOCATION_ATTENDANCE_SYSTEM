/* Basic PWA Service Worker for Django app */
const CACHE_NAME = 'attendance-pwa-v2';
const OFFLINE_URL = '/static/offline.html';
const PRECACHE_URLS = [
  '/',
  '/static/manifest.json',
  OFFLINE_URL
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(PRECACHE_URLS)).then(self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
    )).then(() => self.clients.claim())
  );
});

// Network-first for navigation, cache-first for others
self.addEventListener('fetch', event => {
  const req = event.request;
  const url = new URL(req.url);

  // Always bypass cross-origin requests (avoid breaking Google Maps, CDNs, etc.)
  if (url.origin !== self.location.origin) {
    // Special case: explicitly allow Google Maps domains without caching
    if (url.hostname.endsWith('googleapis.com') || url.hostname.endsWith('gstatic.com')) {
      event.respondWith(fetch(req));
      return;
    }
    // For other cross-origin requests, just pass-through
    event.respondWith(fetch(req));
    return;
  }

  // Navigation requests: network-first, fallback to cache/offline
  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req).then(resp => {
        const copy = resp.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(req, copy)).catch(() => {});
        return resp;
      }).catch(async () => {
        const cached = await caches.match(req);
        return cached || caches.match(OFFLINE_URL);
      })
    );
    return;
  }

  // Only cache GET same-origin requests
  if (req.method !== 'GET') {
    event.respondWith(fetch(req));
    return;
  }

  // Static assets: cache-first
  event.respondWith(
    caches.match(req).then(cached => {
      if (cached) return cached;
      return fetch(req).then(resp => {
        try {
          const copy = resp.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(req, copy)).catch(() => {});
        } catch (e) { /* ignore */ }
        return resp;
      }).catch(async () => {
        // As a last resort, try offline page for same-origin document requests
        const accept = req.headers.get('accept') || '';
        if (accept.includes('text/html')) {
          const offline = await caches.match(OFFLINE_URL);
          if (offline) return offline;
        }
        // Ensure a Response is always returned
        return Response.error();
      });
    })
  );
});
