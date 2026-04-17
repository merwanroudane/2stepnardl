/* ============================================================
   main.js — twostep-nardl docs interactions
   ============================================================ */

// ─── Navbar scroll ──────────────────────────────────────────
const navbar = document.getElementById('navbar');
if (navbar) {
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 30);
  });
}

// ─── Hamburger menu ─────────────────────────────────────────
const hamburger = document.getElementById('hamburger');
const navLinks  = document.querySelector('.nav-links');
if (hamburger && navLinks) {
  hamburger.addEventListener('click', () => {
    navLinks.classList.toggle('open');
  });
  document.addEventListener('click', (e) => {
    if (!navbar.contains(e.target)) navLinks.classList.remove('open');
  });
}

// ─── Copy buttons ───────────────────────────────────────────
document.querySelectorAll('.copy-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const text = btn.dataset.copy || btn.closest('pre')?.innerText || '';
    navigator.clipboard.writeText(text).then(() => {
      btn.classList.add('copied');
      const origHTML = btn.innerHTML;
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`;
      setTimeout(() => { btn.classList.remove('copied'); btn.innerHTML = origHTML; }, 1800);
    });
  });
});

// ─── Code tabs ──────────────────────────────────────────────
document.querySelectorAll('.code-tabs').forEach(tabGroup => {
  const tabs   = tabGroup.querySelectorAll('.tab');
  const panels = tabGroup.nextElementSibling?.querySelectorAll('.code-panel');
  if (!panels) return;
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      const target = document.getElementById(tab.dataset.tab);
      if (target) target.classList.add('active');
    });
  });
});

// ─── Intersection Observer (feature cards) ──────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry, i) => {
    if (entry.isIntersecting) {
      setTimeout(() => {
        entry.target.classList.add('visible');
      }, i * 80);
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('[data-aos]').forEach(el => observer.observe(el));

// ─── Gallery lightbox ────────────────────────────────────────
const lightbox    = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightboxImg');
const lightboxCap = document.getElementById('lightboxCaption');
const lightboxClose = document.getElementById('lightboxClose');

document.querySelectorAll('.gallery-item').forEach(item => {
  item.addEventListener('click', () => {
    const img = item.querySelector('img');
    const title = item.dataset.title || '';
    lightboxImg.src = img.src;
    lightboxImg.alt = img.alt;
    lightboxCap.textContent = title;
    lightbox.classList.add('open');
    document.body.style.overflow = 'hidden';
  });
});

if (lightboxClose) {
  lightboxClose.addEventListener('click', closeLightbox);
}
if (lightbox) {
  lightbox.addEventListener('click', (e) => {
    if (e.target === lightbox) closeLightbox();
  });
}
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeLightbox();
});

function closeLightbox() {
  if (lightbox) {
    lightbox.classList.remove('open');
    document.body.style.overflow = '';
  }
}

// ─── Animated wave canvas (Hero) ────────────────────────────
const canvas = document.getElementById('waveCanvas');
if (canvas) {
  const ctx = canvas.getContext('2d');
  let t = 0;

  function resizeCanvas() {
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  }
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  function drawWave() {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    [[0.5, '#6366f1', 0], [0.3, '#06b6d4', Math.PI], [0.2, '#10b981', Math.PI*0.5]].forEach(([amp, col, off]) => {
      ctx.beginPath();
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.5;
      for (let x = 0; x <= W; x += 2) {
        const y = H/2 + amp * H * 0.3 * Math.sin(x / 120 + t + off);
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    t += 0.008;
    requestAnimationFrame(drawWave);
  }
  drawWave();
}

// ─── API sidebar active link on scroll ──────────────────────
const apiSections = document.querySelectorAll('.api-section[id]');
const sidebarLinks = document.querySelectorAll('.api-sidebar a');
if (apiSections.length && sidebarLinks.length) {
  const activateLink = () => {
    const fromTop = window.scrollY + 100;
    apiSections.forEach(section => {
      if (section.offsetTop <= fromTop && section.offsetTop + section.offsetHeight > fromTop) {
        sidebarLinks.forEach(a => a.classList.remove('active'));
        const match = document.querySelector(`.api-sidebar a[href="#${section.id}"]`);
        if (match) match.classList.add('active');
      }
    });
  };
  window.addEventListener('scroll', activateLink);
}

// ─── Smooth anchor scroll with offset ───────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', (e) => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {
      e.preventDefault();
      const top = target.getBoundingClientRect().top + window.scrollY - 80;
      window.scrollTo({ top, behavior: 'smooth' });
    }
  });
});
