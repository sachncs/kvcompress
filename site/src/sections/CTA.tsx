import { Reveal } from '../lib/Reveal';

export function CTA() {
  return (
    <section id="docs" className="relative pb-24 pt-16 sm:pb-32 sm:pt-24">
      <div className="container-x">
        <Reveal>
          <div className="relative overflow-hidden rounded-3xl bg-ink-900 p-10 ring-1 ring-white/10 sm:p-16 light:bg-white light:ring-ink-900/[0.08]">
            <div
              aria-hidden
              className="pointer-events-none absolute -top-32 left-1/2 -z-0 h-72 w-[700px] -translate-x-1/2 rounded-full bg-accent-500/30 blur-3xl"
            />
            <div
              aria-hidden
              className="pointer-events-none absolute inset-0 -z-0 grid-bg opacity-40"
            />
            <div className="relative mx-auto max-w-2xl text-center">
              <span className="chip light:chip-light mx-auto">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                Ready when you are
              </span>
              <h2 className="mt-6 text-balance text-display-2 font-semibold tracking-tighter light:text-ink-900">
                Ship smaller caches today.
              </h2>
              <p className="mt-5 text-base text-ink-300 light:text-ink-600">
                One import. One call. Read the source — every public symbol is documented,
                type-checked, and tested.
              </p>
              <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
                <a href="#install" className="btn-primary light:btn-light-primary">
                  pip install kvfold
                </a>
                <a
                  href="https://github.com/sachncs/kvcompress"
                  target="_blank"
                  rel="noreferrer"
                  className="btn-ghost light:btn-light-ghost"
                >
                  View on GitHub
                </a>
              </div>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}