import { Logo } from '../components/Logo';
import { footerLinks, site } from '../lib/content';

export function Footer() {
  return (
    <footer className="relative border-t border-white/[0.06] pb-10 pt-16 light:border-ink-900/[0.06]">
      <div className="container-x">
        <div className="grid grid-cols-2 gap-10 sm:grid-cols-3 lg:grid-cols-6">
          <div className="col-span-2">
            <Logo />
            <p className="mt-4 max-w-xs text-sm text-ink-400 light:text-ink-600">
              {site.tagline} A faithful Python re-implementation of the JoLT paper, with a few
              production extras.
            </p>
          </div>
          <FooterColumn title="Product" links={footerLinks.product} />
          <FooterColumn title="Resources" links={footerLinks.resources} />
          <FooterColumn title="Community" links={footerLinks.community} />
          <FooterColumn title="Legal" links={footerLinks.legal} />
        </div>

        <div className="mt-14 hairline light:hairline-light" />
        <div className="mt-6 flex flex-col items-start justify-between gap-3 sm:flex-row sm:items-center">
          <p className="text-xs text-ink-500 light:text-ink-500">
            © {new Date().getFullYear()} {site.author}. Released under {site.license}.
          </p>
          <p className="text-xs text-ink-500 light:text-ink-500">
            Built with Vite, React, Tailwind, and Framer Motion.
          </p>
        </div>
      </div>
    </footer>
  );
}

function FooterColumn({
  title,
  links,
}: {
  title: string;
  links: readonly { label: string; href: string }[];
}) {
  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-widest text-ink-500">
        {title}
      </h3>
      <ul className="mt-4 space-y-2.5">
        {links.map((l) => (
          <li key={l.label}>
            <a
              href={l.href}
              className="text-sm text-ink-300 transition-colors duration-200 hover:text-ink-50 light:text-ink-700 light:hover:text-ink-900"
            >
              {l.label}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}