import { Reveal } from '../lib/Reveal';
import {
  memoryBenchmark,
  speedBenchmark,
  fidelityBenchmark,
  type MemoryRow,
  type SpeedRow,
  type FidelityRow,
} from '../lib/content';

function Bar({ value, max }: { value: number; max: number }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.06] light:bg-ink-900/[0.06]">
      <div
        className="h-full rounded-full bg-gradient-to-r from-accent-400 to-accent-300"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function MemoryBars({ rows }: { rows: MemoryRow[] }) {
  const max = Math.max(...rows.map((r) => parseFloat(r.achieved.replace('×', '')) || 0));
  return (
    <div className="space-y-3">
      {rows.map((r) => {
        const v = parseFloat(r.achieved.replace('×', '')) || 0;
        return (
          <div key={r.name} className="flex items-center gap-3 text-xs">
            <span className="w-24 shrink-0 text-ink-300 light:text-ink-700">{r.name}</span>
            <Bar value={v} max={max} />
            <span className="w-14 shrink-0 text-right font-mono text-ink-200 light:text-ink-800">
              {r.achieved}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function SpeedBars({ rows }: { rows: SpeedRow[] }) {
  const max = Math.max(...rows.map((r) => parseFloat(r.compress.replace('ms', '')) || 0));
  return (
    <div className="space-y-3">
      {rows.map((r) => {
        const v = parseFloat(r.compress.replace('ms', '')) || 0;
        return (
          <div key={r.name} className="space-y-1.5 text-xs">
            <div className="flex items-center gap-3">
              <span className="w-24 shrink-0 text-ink-300 light:text-ink-700">{r.name}</span>
              <span className="ml-auto font-mono text-ink-200 light:text-ink-800">
                {r.compress} · {r.decompress}
              </span>
            </div>
            <Bar value={v} max={max} />
          </div>
        );
      })}
    </div>
  );
}

function FidelityBars({ rows }: { rows: FidelityRow[] }) {
  const max = Math.max(...rows.map((r) => parseFloat(r.kerr) || 0));
  return (
    <div className="space-y-3">
      {rows.map((r) => {
        const v = parseFloat(r.kerr) || 0;
        return (
          <div key={r.name} className="space-y-1.5 text-xs">
            <div className="flex items-center gap-3">
              <span className="w-24 shrink-0 text-ink-300 light:text-ink-700">{r.name}</span>
              <span className="ml-auto font-mono text-ink-200 light:text-ink-800">
                {r.kerr} / {r.verr} · {r.ratio}
              </span>
            </div>
            <Bar value={v} max={max} />
          </div>
        );
      })}
    </div>
  );
}

export function Benchmarks() {
  return (
    <section id="benchmarks" className="relative py-24 sm:py-32">
      <div className="container-x">
        <Reveal>
          <div className="mx-auto max-w-2xl text-center">
            <span className="chip light:chip-light">Benchmarks</span>
            <h2 className="mt-5 text-balance text-display-2 font-semibold tracking-tighter">
              Measured, not marketed.
            </h2>
            <p className="mt-5 text-base text-ink-300 light:text-ink-600">
              Reference numbers captured on a developer CPU box. Reproducible with the scripts
              shipped in the repo.
            </p>
          </div>
        </Reveal>

        <div className="mx-auto mt-14 grid max-w-6xl grid-cols-1 gap-4 lg:grid-cols-3">
          <Reveal>
            <article className="surface light:surface-light h-full p-6">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-sm font-semibold uppercase tracking-widest text-ink-300 light:text-ink-700">
                    {memoryBenchmark.title}
                  </h3>
                  <p className="mt-1 text-xs text-ink-500">{memoryBenchmark.method}</p>
                </div>
                <span className="font-mono text-[10px] uppercase tracking-wider text-emerald-300/80 light:text-emerald-600">
                  ▲ smaller
                </span>
              </div>
              <div className="mt-6">
                <MemoryBars rows={memoryBenchmark.rows} />
              </div>
              <p className="mt-6 text-[11px] text-ink-500">{memoryBenchmark.note}</p>
            </article>
          </Reveal>

          <Reveal delay={0.06}>
            <article className="surface light:surface-light h-full p-6">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-sm font-semibold uppercase tracking-widest text-ink-300 light:text-ink-700">
                    {speedBenchmark.title}
                  </h3>
                  <p className="mt-1 text-xs text-ink-500">{speedBenchmark.method}</p>
                </div>
                <span className="font-mono text-[10px] uppercase tracking-wider text-amber-300/80 light:text-amber-600">
                  ● wall-time
                </span>
              </div>
              <div className="mt-6">
                <SpeedBars rows={speedBenchmark.rows} />
              </div>
              <p className="mt-6 text-[11px] text-ink-500">{speedBenchmark.note}</p>
            </article>
          </Reveal>

          <Reveal delay={0.12}>
            <article className="surface light:surface-light h-full p-6">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-sm font-semibold uppercase tracking-widest text-ink-300 light:text-ink-700">
                    {fidelityBenchmark.title}
                  </h3>
                  <p className="mt-1 text-xs text-ink-500">{fidelityBenchmark.method}</p>
                </div>
                <span className="font-mono text-[10px] uppercase tracking-wider text-accent-300/80 light:text-accent-600">
                  ◆ relative
                </span>
              </div>
              <div className="mt-6">
                <FidelityBars rows={fidelityBenchmark.rows} />
              </div>
              <p className="mt-6 text-[11px] text-ink-500">{fidelityBenchmark.note}</p>
            </article>
          </Reveal>
        </div>

        <Reveal delay={0.2}>
          <p className="mx-auto mt-8 max-w-3xl text-center text-xs text-ink-500 light:text-ink-500">
            These are CPU-only reference numbers, not paper-reproduction data. GPU benchmarks and
            needle-in-haystack sweeps are tracked on the project roadmap.
          </p>
        </Reveal>
      </div>
    </section>
  );
}