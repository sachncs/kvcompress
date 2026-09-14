import { useEffect, useState } from 'react';
import { Logo } from './Logo';
import { ThemeToggle, GitHubButton } from './Icons';
import { navLinks } from '../lib/content';

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);

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
            <a key={l.href} href={l.href} className="nav-link light:nav-link-light">
              {l.label}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          <GitHubButton />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}