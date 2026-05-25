/* =========================================================
   Vaaani — secondary pages (About, Contact)
   Smaller animation set than the landing — same vocabulary.
   ========================================================= */

gsap.registerPlugin(ScrollTrigger);

const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

if (reduced) {
  document.querySelectorAll('.reveal, .page-title .line > span, .page-hero .lede, .back-link')
    .forEach(el => gsap.set(el, { opacity: 1, y: 0 }));
} else {
  // Pre-state
  gsap.set('.back-link', { y: 12, opacity: 0 });
  gsap.set('.page-hero .eyebrow', { y: 18, opacity: 0 });
  gsap.set('.page-title .line > span', { yPercent: 110, opacity: 0 });
  gsap.set('.page-hero .lede', { y: 18, opacity: 0 });
  gsap.set('.reveal', { y: 28, opacity: 0 });

  // Page entry timeline
  const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });
  tl
    .to('.back-link', { y: 0, opacity: 1, duration: 0.5 }, 0)
    .to('.page-hero .eyebrow', { y: 0, opacity: 1, duration: 0.55 }, 0.1)
    .to('.page-title .line > span', { yPercent: 0, opacity: 1, duration: 0.85, stagger: 0.1, ease: 'power4.out' }, 0.18)
    .to('.page-hero .lede', { y: 0, opacity: 1, duration: 0.6, stagger: 0.1 }, 0.6);

  // Scroll reveals
  gsap.utils.toArray('.reveal').forEach(el => {
    gsap.to(el, {
      y: 0,
      opacity: 1,
      duration: 0.8,
      ease: 'power3.out',
      scrollTrigger: { trigger: el, start: 'top 88%', toggleActions: 'play none none reverse' },
    });
  });

  // Magnetic primary buttons (shared interaction language with the landing page)
  document.querySelectorAll('.btn-primary').forEach(btn => {
    const onMove = (e) => {
      const r = btn.getBoundingClientRect();
      const dx = (e.clientX - (r.left + r.width / 2)) * 0.2;
      const dy = (e.clientY - (r.top + r.height / 2)) * 0.25;
      gsap.to(btn, { x: dx, y: dy, duration: 0.3, ease: 'power2.out' });
    };
    const onLeave = () => gsap.to(btn, { x: 0, y: 0, duration: 0.45, ease: 'elastic.out(1, 0.4)' });
    btn.addEventListener('mousemove', onMove);
    btn.addEventListener('mouseleave', onLeave);
  });

  // Refresh once Google Maps iframe loads to recalc layout
  document.querySelectorAll('iframe').forEach(f => {
    f.addEventListener('load', () => ScrollTrigger.refresh());
  });
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => ScrollTrigger.refresh());
  }
}
