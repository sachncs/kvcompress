import { Reveal } from '../lib/Reveal';
import { codeSnippets, site } from '../lib/content';

export function Install() {
  return (
    <section id="install" className="relative py-24 sm:py-32">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent light:via-ink-900/10"
      />
      <div className="container-x">
        <Reveal>
          <div className="mx-auto max-w-2xl text-center">
            <span className="chip light:chip-light">Get started</span>
            <h2 className="mt-5 text-balance text-display-2 font-semibold tracking-tighter">
              Install from source. One import. You're in.
            </h2>
            <p className="mt-5 text-base text-ink-300 light:text-ink-600">
              The package is pre-release. Clone the repository, install editable, then validate on your workload.
            </p>
          </div>
        </Reveal>

        <Reveal delay={0.1}>
          <div className="mx-auto mt-12 max-w-3xl overflow-hidden rounded-3xl bg-ink-900 ring-1 ring-white/10 light:bg-white light:ring-ink-900/[0.08]">
            <div className="flex items-center justify-between border-b border-white/[0.06] px-5 py-3 light:border-ink-900/[0.06]">
              <div className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-ink-700/80 light:bg-ink-200" />
                <span className="h-2.5 w-2.5 rounded-full bg-ink-700/80 light:bg-ink-200" />
                <span className="h-2.5 w-2.5 rounded-full bg-ink-700/80 light:bg-ink-200" />
              </div>
              <span className="font-mono text-xs text-ink-500">bash</span>
            </div>
            <pre className="code overflow-x-auto p-6 text-ink-100 light:text-ink-900">
              <code>{codeSnippets.install}</code>
            </pre>
          </div>
        </Reveal>

        <Reveal delay={0.18}>
          <div className="mx-auto mt-8 grid max-w-3xl grid-cols-1 gap-3 sm:grid-cols-3">
            {[
              { k: 'Python', v: '3.11 – 3.13' },
              { k: 'PyTorch', v: '≥ 2.5' },
              { k: 'License', v: site.license },
            ].map((b) => (
              <div
                key={b.k}
                className="flex items-center justify-between rounded-2xl bg-white/[0.02] px-5 py-4 ring-1 ring-white/[0.06] light:bg-white light:ring-ink-900/[0.06]"
              >
                <span className="text-xs uppercase tracking-widest text-ink-500">{b.k}</span>
                <span className="font-mono text-xs text-ink-200 light:text-ink-800">{b.v}</span>
              </div>
            ))}
          </div>
        </Reveal>
      </div>
    </section>
  );
}
