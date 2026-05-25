/* =========================================================
   Vaaani landing site — professional-grade GSAP storytelling
   - Intro curtain lifts on first paint
   - Hero: masked word reveals + constellation breathe
   - Chapter 1: pinned crossfade with dot progress strip
   - Chapter 2: dialogs slide in from opposite sides
   - Chapter 3: pinned constellation build (strokes, clusters, labels, lockstep bullets)
   - Chapter 4: horizontal subjects reel with per-panel scale/lift + scroll-snap
   - Parallax layers, brand hover, magnetic CTAs, 3D tilt
   - Custom cursor, scroll progress bar, Lenis smooth scroll
   - Reduced-motion + mobile fallbacks throughout
   ========================================================= */

gsap.registerPlugin(ScrollTrigger, ScrollToPlugin);

const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const isTouch  = window.matchMedia('(pointer: coarse)').matches;
const desktop  = window.matchMedia('(min-width: 980px)').matches;
const EASE_OUT   = 'power3.out';
const EASE_INOUT = 'power3.inOut';
const EASE_EXPO  = 'power4.out';

/* =========================================================
   TEXT SPLITTING — masked words
   ========================================================= */
function splitWords(el) {
  if (!el || el.dataset.split === '1') return [];
  const out = [];
  function walk(node) {
    if (node.nodeType === 3) {
      const frag = document.createDocumentFragment();
      const parts = node.textContent.split(/(\s+)/);
      parts.forEach(part => {
        if (!part) return;
        if (/^\s+$/.test(part)) {
          frag.appendChild(document.createTextNode(part));
        } else {
          const wrap = document.createElement('span');
          wrap.className = 'split-word';
          const inner = document.createElement('span');
          inner.className = 'split-word-i';
          inner.textContent = part;
          wrap.appendChild(inner);
          frag.appendChild(wrap);
          out.push(inner);
        }
      });
      node.parentNode.replaceChild(frag, node);
    } else if (node.nodeType === 1 && !node.classList.contains('split-word')) {
      Array.from(node.childNodes).forEach(walk);
    }
  }
  walk(el);
  el.dataset.split = '1';
  return out;
}

/* =========================================================
   LENIS — smooth scroll
   ========================================================= */
let lenis = null;
if (!reduced && !isTouch && typeof Lenis !== 'undefined') {
  lenis = new Lenis({
    duration: 1.15,
    easing: t => 1 - Math.pow(1 - t, 4),
    smoothWheel: true,
    smoothTouch: false,
    wheelMultiplier: 1.0,
    lerp: 0.085,
  });
  lenis.on('scroll', ScrollTrigger.update);
  gsap.ticker.add(t => lenis.raf(t * 1000));
  gsap.ticker.lagSmoothing(0);
}

/* =========================================================
   ANCHOR SCROLLS — route through Lenis when available
   ========================================================= */
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const id = a.getAttribute('href');
    const target = id && id.length > 1 ? document.querySelector(id) : null;
    if (!target) return;
    e.preventDefault();
    if (lenis) {
      lenis.scrollTo(target, { offset: -60, duration: 1.2 });
    } else {
      gsap.to(window, { duration: 1.1, scrollTo: { y: target, offsetY: 60 }, ease: EASE_INOUT });
    }
  });
});

/* =========================================================
   NAV SCROLL STATE + PROGRESS BAR
   ========================================================= */
const nav = document.getElementById('nav');
ScrollTrigger.create({
  start: 'top -40', end: 99999,
  onUpdate: self => nav && nav.classList.toggle('scrolled', self.scroll() > 40),
});

const progressBar = document.querySelector('.scroll-progress > span');
if (progressBar) {
  ScrollTrigger.create({
    start: 0, end: () => document.body.scrollHeight - window.innerHeight,
    onUpdate: self => { progressBar.style.width = (self.progress * 100).toFixed(2) + '%'; },
  });
}

/* =========================================================
   POST-LOAD REFRESH
   ========================================================= */
if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(() => ScrollTrigger.refresh());
}
window.addEventListener('load', () => ScrollTrigger.refresh());

/* =========================================================
   REDUCED-MOTION FALLBACK — bail early
   ========================================================= */
if (reduced) {
  document.querySelector('.intro-curtain')?.style.setProperty('display', 'none');
  document.querySelectorAll('.reveal,.hero-title,.hero-sub,.hero-cta,.hero-float-card,.eyebrow,.build-line,.trap-line')
    .forEach(el => gsap.set(el, { opacity: 1, y: 0, scale: 1 }));
  /* nothing else runs */
} else {

/* =========================================================
   MODULE 0 — PRE-STATE: hide animated elements
   ========================================================= */
  /* Split section titles into masked words. The hero title is deliberately
     NOT split — the redesigned hero animates as a single block so the
     headline doesn't depend on a long word-by-word stagger. */
  const titleSelectors = [
    '.section-title',
    '.chapter-build .build-text .section-title',
    '.subjects-head .section-title',
  ];
  titleSelectors.forEach(sel => document.querySelectorAll(sel).forEach(splitWords));

  /* ensure trap-line inner text is never masked */
  document.querySelectorAll('.trap-line .split-word-i').forEach(w => gsap.set(w, { yPercent: 0 }));

  gsap.set('.hero .eyebrow',     { y: 18, opacity: 0 });
  gsap.set('.hero-title',        { y: 24, opacity: 0 });
  gsap.set('.hero-sub',          { y: 18, opacity: 0 });
  gsap.set('.hero-cta',          { y: 14, opacity: 0 });
  gsap.set('.hero-float-card',   { opacity: 0, scale: 0.92 });
  gsap.set('.reveal',            { y: 28, opacity: 0 });
  gsap.set('.wipe-line',         { width: '0%' });

/* =========================================================
   MODULE 1 — CUSTOM CURSOR
   ========================================================= */
  if (desktop && !isTouch) {
    document.body.classList.add('has-cursor');
    const dot  = document.querySelector('.cursor-dot');
    const ring = document.querySelector('.cursor-ring');
    let mx = window.innerWidth/2, my = window.innerHeight/2;
    let dx = mx, dy = my, rx = mx, ry = my;
    window.addEventListener('mousemove', e => { mx = e.clientX; my = e.clientY; });
    gsap.ticker.add(() => {
      dx += (mx - dx) * 0.35;
      dy += (my - dy) * 0.35;
      rx += (mx - rx) * 0.14;
      ry += (my - ry) * 0.14;
      if (dot)  dot.style.transform  = `translate(${dx}px, ${dy}px)`;
      if (ring) ring.style.transform = `translate(${rx}px, ${ry}px)`;
    });
    document.querySelectorAll('a,button,.subject-card,.feature,.step,.subject-panel,.dialog,.brand')
      .forEach(el => {
        el.addEventListener('mouseenter', () => document.body.classList.add('cursor-hover'));
        el.addEventListener('mouseleave', () => document.body.classList.remove('cursor-hover'));
      });
  }

/* =========================================================
   MODULE 2 — INTRO CURTAIN (lifts on first paint)
   ========================================================= */
  const curtain = document.querySelector('.intro-curtain');
  const curtainTL = gsap.timeline({
    defaults: { ease: EASE_OUT },
    delay: 0.15,
    onComplete: () => {
      if (curtain) curtain.style.display = 'none';
    },
  });

  curtainTL
    .from('.curtain-brand', { scale: 0, rotate: -35, opacity: 0, duration: 0.7, ease: 'back.out(1.7)' })
    .from('.curtain-word',  { y: 18, opacity: 0, duration: 0.45 }, '-=0.15')
    .to(curtain, {
      yPercent: -100,
      duration: 1.35,
      ease: 'power4.inOut',
      delay: 0.4,
    });

/* =========================================================
   MODULE 3 — HERO ENTRY (plays after curtain starts lifting)
   ========================================================= */
  const heroIn = gsap.timeline({ defaults: { ease: EASE_EXPO }, delay: 0.6 });
  heroIn
    .to('.hero .eyebrow', { y: 0, opacity: 1, duration: 0.55 }, 0)
    .to('.hero-title',    { y: 0, opacity: 1, duration: 0.85 }, 0.1)
    .to('.hero-sub',      { y: 0, opacity: 1, duration: 0.65 }, 0.5)
    .to('.hero-cta',      { y: 0, opacity: 1, duration: 0.55 }, 0.65)
    .to('.hero-float-card', {
      opacity: 1, scale: 1, duration: 0.7, stagger: 0.12, ease: 'back.out(1.4)',
    }, 0.4);

/* =========================================================
   MODULE 4 — HERO & GENERAL PARALLAX
   ========================================================= */
  gsap.to('.hero-bg', {
    yPercent: 28, ease: 'none',
    scrollTrigger: { trigger: '.hero', start: 'top top', end: 'bottom top', scrub: true },
  });
  gsap.to('.hero-content', {
    opacity: 0.18, y: -50, ease: 'none',
    scrollTrigger: { trigger: '.hero', start: 'top top', end: 'bottom 30%', scrub: true },
  });

  /* subtle parallax on chapter backgrounds */
  document.querySelectorAll('.chapter-trap,.chapter-build').forEach(section => {
    gsap.to(section, {
      backgroundPositionY: '15%',
      ease: 'none',
      scrollTrigger: { trigger: section, start: 'top bottom', end: 'bottom top', scrub: true },
    });
  });

/* =========================================================
   MODULE 5 — CHAPTER 01: THE TRAP (pinned crossfade)
   ========================================================= */
  const trap = document.querySelector('.chapter-trap');
  if (trap && desktop) {
    const lines = trap.querySelectorAll('.trap-line');
    const dots  = trap.querySelectorAll('.trap-progress span');

    gsap.set(lines, { opacity: 0, y: 24 });

    const tl = gsap.timeline({
      defaults: { ease: EASE_INOUT },
      scrollTrigger: {
        trigger: trap, start: 'top top', end: '+=360%',
        pin: true, scrub: 0.8, anticipatePin: 1,
      },
    });

    const seg = 0.95;

    lines.forEach((line, i) => {
      const inPos  = i * seg;
      const outPos = inPos + seg - 0.32;

      tl.to(line, { opacity: 1, y: 0, duration: 0.45 }, inPos)
        .call(() => dots.forEach((d, j) => d.classList.toggle('on', j <= i)), null, inPos + 0.35);

      if (i < lines.length - 1) {
        tl.to(line, { opacity: 0, y: -16, duration: 0.35 }, outPos);
      }
    });
  } else if (trap) {
    trap.querySelectorAll('.trap-line').forEach((line, i) => {
      gsap.to(line, {
        opacity: 1, y: 0, duration: 0.65, ease: EASE_OUT, delay: i * 0.1,
        scrollTrigger: { trigger: line, start: 'top 80%' },
      });
    });
  }

/* =========================================================
   MODULE 6 — CHAPTER 02: FLIP DIALOGS
   ========================================================= */
  gsap.utils.toArray('.chapter-flip .dialog').forEach(d => {
    const side = d.dataset.side === 'left' ? -90 : 90;
    gsap.fromTo(d,
      { x: side, opacity: 0 },
      {
        x: 0, opacity: 1, duration: 1.05, ease: EASE_EXPO,
        scrollTrigger: { trigger: d, start: 'top 84%', toggleActions: 'play none none reverse' },
      });
  });

  /* chapter-2 heading reveal with stagger */
  const flipHead = document.querySelector('.chapter-flip .chapter-head');
  if (flipHead) {
    gsap.fromTo(flipHead.children,
      { y: 24, opacity: 0 },
      {
        y: 0, opacity: 1, duration: 0.7, ease: EASE_OUT, stagger: 0.08,
        scrollTrigger: { trigger: flipHead, start: 'top 86%', toggleActions: 'play none none reverse' },
      });
  }

/* =========================================================
   MODULE 7 — CHAPTER 03: CONSTELLATION BUILD (pinned)
   ========================================================= */
  const build = document.querySelector('.chapter-build');
  if (build && desktop) {
    const stars    = build.querySelectorAll('.build-stars circle');
    const lines    = build.querySelectorAll('.build-lines line');
    const clusters = build.querySelector('.build-clusters');
    const labels   = build.querySelector('.build-labels');
    const bullets  = build.querySelectorAll('.build-line');

    gsap.set(stars,    { opacity: 0, scale: 0, transformOrigin: 'center' });
    gsap.set(lines,    { strokeDashoffset: 600 });
    gsap.set(clusters, { opacity: 0 });
    gsap.set(labels,   { opacity: 0 });

    const tl = gsap.timeline({
      defaults: { ease: EASE_INOUT },
      scrollTrigger: {
        trigger: build, start: 'top top', end: '+=340%',
        pin: true, scrub: 0.8, anticipatePin: 1,
        onUpdate: self => {
          const p = self.progress;
          bullets.forEach((b, i) => b.classList.toggle('on', p > i * 0.238));
        },
      },
    });

    tl.to(stars,    { opacity: 1, scale: 1, duration: 1.2, ease: 'back.out(1.4)', stagger: 0.07 })
      .to(lines,    { strokeDashoffset: 0, duration: 1.3, ease: 'power2.inOut', stagger: 0.04 }, '+=0.15')
      .to(clusters, { opacity: 1, duration: 0.8, ease: 'power2.out' }, '+=0.15')
      .to(labels,   { opacity: 1, duration: 0.7 }, '+=0.1');
  } else if (build) {
    build.querySelectorAll('.build-line').forEach(l => l.classList.add('on'));
    gsap.set('.build-stars circle,.build-clusters,.build-labels', { opacity: 1, scale: 1 });
    gsap.set('.build-lines line', { strokeDashoffset: 0 });
  }

/* =========================================================
   MODULE 8 — CHAPTER 04: HORIZONTAL SUBJECTS + SNAP
   ========================================================= */
  const subjects = document.querySelector('.chapter-subjects');
  if (subjects && desktop) {
    const track = subjects.querySelector('.subjects-track');
    const panels = gsap.utils.toArray('.subject-panel');

    const horizTween = gsap.to(track, {
      x: () => -(track.scrollWidth - window.innerWidth + 40),
      ease: 'none',
      scrollTrigger: {
        trigger: subjects, start: 'top top',
        end: () => '+=' + (track.scrollWidth - window.innerWidth + 80),
        scrub: 0.9, pin: true, anticipatePin: 1, invalidateOnRefresh: true,
        snap: {
          snapTo: (progress) => {
            const scrollDist = track.scrollWidth - window.innerWidth + 40;
            const hw = window.innerWidth / 2;
            const positions = panels.map(p => {
              const cx = p.offsetLeft + p.offsetWidth / 2;
              return Math.max(0, Math.min(1, (cx - hw) / scrollDist));
            });
            let best = positions[0], minD = Infinity;
            positions.forEach(p => { const d = Math.abs(progress - p); if (d < minD) { minD = d; best = p; } });
            return best;
          },
          duration: 0.4,
          ease: 'power2.out',
        },
      },
    });

    /* each panel scales/lifts as it crosses screen-centre */
    panels.forEach(panel => {
      gsap.fromTo(panel,
        { scale: 0.93, y: 28, opacity: 0.5 },
        {
          scale: 1, y: 0, opacity: 1, duration: 0.55, ease: EASE_OUT,
          scrollTrigger: {
            trigger: panel,
            containerAnimation: horizTween,
            start: 'left center-=130',
            end: 'left center',
            scrub: true,
            onEnter:      () => panel.classList.add('is-center'),
            onLeave:      () => panel.classList.remove('is-center'),
            onEnterBack:  () => panel.classList.add('is-center'),
            onLeaveBack:  () => panel.classList.remove('is-center'),
          },
        });

      gsap.fromTo(panel,
        { scale: 1, y: 0, opacity: 1 },
        {
          scale: 0.93, y: 28, opacity: 0.5, duration: 0.55, ease: EASE_INOUT,
          scrollTrigger: {
            trigger: panel,
            containerAnimation: horizTween,
            start: 'left center',
            end: 'left center+=130',
            scrub: true,
          },
        });
    });

    /* entry: first panel starts at full glory */
    gsap.set(panels[0], { scale: 1, y: 0, opacity: 1 });
    panels[0].classList.add('is-center');
  }

/* =========================================================
   MODULE 9 — SCROLL REVEALS
   ========================================================= */
  gsap.utils.toArray('.reveal').forEach(el => {
    gsap.to(el, {
      y: 0, opacity: 1, duration: 0.85, ease: EASE_OUT,
      scrollTrigger: { trigger: el, start: 'top 88%', toggleActions: 'play none none reverse' },
    });
  });

  /* section titles — masked word reveals */
  document.querySelectorAll('.section-title').forEach(title => {
    const words = title.querySelectorAll('.split-word-i');
    if (!words.length) return;
    gsap.fromTo(words,
      { yPercent: 110 },
      {
        yPercent: 0, duration: 0.95, ease: EASE_EXPO, stagger: 0.04,
        scrollTrigger: { trigger: title, start: 'top 86%', toggleActions: 'play none none reverse' },
      });
  });

/* =========================================================
   MODULE 10 — INTERACTIONS: 3D tilt + magnetic + brand hover
   ========================================================= */
  document.querySelectorAll('.subject-card').forEach(card => {
    card.addEventListener('mousemove', e => {
      const r = card.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width  - 0.5) * 2;
      const py = ((e.clientY - r.top)  / r.height - 0.5) * 2;
      gsap.to(card, {
        rotateX: -py * 3.5, rotateY: px * 3.5, duration: 0.4, ease: EASE_OUT,
        transformPerspective: 800,
      });
    });
    card.addEventListener('mouseleave', () =>
      gsap.to(card, { rotateX: 0, rotateY: 0, duration: 0.6, ease: EASE_OUT }));
  });

  document.querySelectorAll('.btn-primary').forEach(btn => {
    btn.addEventListener('mousemove', e => {
      const r = btn.getBoundingClientRect();
      gsap.to(btn, {
        x: (e.clientX - (r.left + r.width/2))  * 0.18,
        y: (e.clientY - (r.top  + r.height/2)) * 0.22,
        duration: 0.4, ease: EASE_OUT,
      });
    });
    btn.addEventListener('mouseleave', () =>
      gsap.to(btn, { x: 0, y: 0, duration: 0.55, ease: EASE_OUT }));
  });

  /* brand hover — SVG circles pulse in sequence (CSS handles base, JS reinforces) */
  const brandMarks = document.querySelectorAll('.brand-mark');
  brandMarks.forEach(mark => {
    const circles = mark.querySelectorAll('circle');
    mark.parentElement.addEventListener('mouseenter', () => {
      circles.forEach((c, i) => {
        gsap.fromTo(c,
          { scale: 1, opacity: 1 },
          { scale: 1.45, duration: 0.28, ease: 'power2.out', delay: i * 0.04, yoyo: true, repeat: 1 });
      });
    });
  });

} /* end -- !reduced guard */
