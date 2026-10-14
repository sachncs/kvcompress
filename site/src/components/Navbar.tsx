import { useEffect, useState } from 'react';
import { Logo } from './Logo';
import { ThemeToggle, GitHubButton } from './Icons';
import { navLinks } from '../lib/content';

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 16);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <header
      className={`fixed inset-x-0 top-0 z-50 transition-all duration-500 ease-out-expo ${
        scrolled
          ? 'backdrop-blur-xl bg-ink-950/70 light:bg-white/70 border-b border-white/[0.06] light:border-ink-900/[0.06]'
          : ''
      }`}
    >
      <div className="container-x flex h-14 items-center justify-between">
        <a href="#top" className="flex items-center" aria-label="kvfold home">
          <Logo />
        </a>
        <nav className="hidden items-center gap-7 md:flex">
          {navLinks.map((l) => (
            <a
              key={l.href}
              href={l.href}
              target={l.href.startsWith('http') ? '_blank' : undefined}
              rel={l.href.startsWith('http') ? 'noreferrer' : undefined}
              className="nav-link light:nav-link-light"
            >
              {l.label}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          <GitHubButton />
          <ThemeToggle />
          <button
            type="button"
            className="inline-flex h-9 w-9 items-center justify-center rounded-full text-ink-400 ring-1 ring-white/10 md:hidden light:text-ink-700 light:ring-ink-900/10"
            aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((open) => !open)}
          >
            <span aria-hidden className="text-lg leading-none">{menuOpen ? '×' : '☰'}</span>
          </button>
        </div>
      </div>
      {menuOpen && (
        <nav className="border-t border-white/[0.06] bg-ink-950/95 px-6 py-4 backdrop-blur-xl md:hidden light:border-ink-900/[0.06] light:bg-white/95">
          <div className="container-x flex flex-col gap-1 px-0">
            {navLinks.map((l) => (
              <a
                key={l.href}
                href={l.href}
                target={l.href.startsWith('http') ? '_blank' : undefined}
                rel={l.href.startsWith('http') ? 'noreferrer' : undefined}
                className="rounded-xl px-3 py-3 text-sm text-ink-300 hover:bg-white/[0.05] hover:text-ink-50 light:text-ink-700 light:hover:bg-ink-900/[0.04] light:hover:text-ink-900"
                onClick={() => setMenuOpen(false)}
              >
                {l.label}
              </a>
            ))}
          </div>
        </nav>
      )}
    </header>
  );
}
