import { useState } from 'react';
import { Reveal } from '../lib/Reveal';
import { faqs, site } from '../lib/content';

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      className={`transition-transform duration-300 ${open ? 'rotate-180' : ''}`}
    >
      <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function FAQ() {
  const [open, setOpen] = useState<number | null>(0);
  return (
    <section className="relative py-24 sm:py-32">
      <div className="container-x">
        <div className="grid max-w-5xl grid-cols-1 gap-12 lg:grid-cols-[1fr_1.6fr]">
          <Reveal>
            <div>
              <span className="chip light:chip-light">FAQ</span>
              <h2 className="mt-5 text-balance text-display-3 font-semibold tracking-tighter">
                Honest answers, brief.
              </h2>
              <p className="mt-4 text-sm text-ink-400 light:text-ink-600">
                The five questions that come up most often. Reach out on GitHub for anything else.
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <ul className="divide-y divide-white/[0.06] light:divide-ink-900/[0.08]">
              {faqs.map((f, i) => {
                const isOpen = open === i;
                return (
                  <li key={f.q}>
                    <button
                      type="button"
                      onClick={() => setOpen(isOpen ? null : i)}
                      className="flex w-full items-center justify-between gap-6 py-5 text-left transition-colors duration-200 hover:text-ink-50 light:hover:text-ink-900"
                      aria-expanded={isOpen}
                    >
                      <span className="text-sm font-medium tracking-tight">{f.q}</span>
                      <span className="text-ink-500 light:text-ink-500">
                        <ChevronIcon open={isOpen} />
                      </span>
                    </button>
                    <div
                      className={`grid overflow-hidden transition-all duration-500 ease-out-expo ${
                        isOpen ? 'grid-rows-[1fr] pb-5 opacity-100' : 'grid-rows-[0fr] opacity-0'
                      }`}
                    >
                      <div className="min-h-0">
                        <p className="text-sm leading-relaxed text-ink-400 light:text-ink-600">
                          {f.a}
                        </p>
                      </div>
                    </div>
                  </li>
                );
              })}
            </ul>
          </Reveal>
        </div>

        <Reveal delay={0.2}>
          <div className="mt-20 rounded-3xl bg-gradient-to-b from-white/[0.04] to-transparent p-8 ring-1 ring-white/[0.08] light:from-ink-900/[0.04] light:ring-ink-900/[0.08]">
            <div className="flex flex-col items-start gap-6 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="font-mono text-xs uppercase tracking-widest text-ink-500">
                  Based on the paper
                </p>
                <h3 className="mt-2 max-w-2xl text-xl font-semibold tracking-tight light:text-ink-900">
                  {site.paper.title}
                </h3>
                <p className="mt-2 text-sm text-ink-400 light:text-ink-600">
                  {site.paper.authors} · {site.paper.year} · arXiv:{site.paper.arxiv}
                </p>
              </div>
              <a
                href={site.paper.url}
                target="_blank"
                rel="noreferrer"
                className="btn-ghost light:btn-light-ghost shrink-0"
              >
                Read on arXiv
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M7 17L17 7M9 7h8v8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </a>
            </div>
            <p className="mt-6 border-t border-white/[0.06] pt-6 text-xs text-ink-500 light:border-ink-900/[0.08]">
              kvfold is an independent re-implementation by {site.author}. The algorithm and
              theoretical results belong to the paper&rsquo;s authors.
            </p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}