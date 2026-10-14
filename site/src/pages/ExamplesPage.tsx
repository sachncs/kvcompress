const examples = [
  {
    number: '01',
    title: 'Hugging Face quickstart',
    body: 'Wrap a causal LM and generate normally. The cache adapter is model-scoped and reversible.',
    code: `from transformers import GPT2LMHeadModel, GPT2Tokenizer
from kvfold import enable_compression

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
handle = enable_compression(model, method="flash", ratio=3.0)

try:
    ids = tok.encode("The capital of France is", return_tensors="pt")
    out = model.generate(ids, max_new_tokens=15)
    print(tok.decode(out[0]))
    print(handle.stats_dict())
finally:
    handle.disable()`,
  },
  {
    number: '02',
    title: 'Direct cache compression',
    body: 'Use the compressor and Cache directly for offline analysis or a custom serving integration.',
    code: `import torch
from kvfold import Cache, Jolt

key = torch.randn(8, 256, 64)
value = torch.randn_like(key)
compressor = Jolt(ratio=2.0, bits=(0,))

cache = Cache(compressor=compressor)
cache.store(layer=0, key=key, value=value)
key_hat, value_hat = cache.retrieve(0)
print(cache.compression_ratio())`,
  },
  {
    number: '03',
    title: 'Compare methods',
    body: 'Run the same synthetic K/V tensors through JoLT, Flash, Low, and IntQuant to compare error and bytes.',
    code: `from kvfold import Flash, IntQuant, Jolt, Low

methods = [
    ("jolt @ 3x", Jolt(ratio=3.0, bits=(0, 4, 8))),
    ("flash @ 3x", Flash(ratio=3.0, bits=(0, 4, 8))),
    ("low @ rank=32", Low(rank=32)),
    ("int4 per-channel", IntQuant(bits=4, per_channel=True)),
]`,
  },
  {
    number: '04',
    title: 'Long-context smoke test',
    body: 'Exercise generation with a buried needle. Treat this as a smoke test, not a quality guarantee.',
    code: `for method in ("pass", "flash"):
    handle = enable_compression(
        model, method=method, target_memory=0.33
    )
    try:
        output = model.generate(ids, max_new_tokens=20)
    finally:
        handle.disable()`,
  },
];

export function ExamplesPage() {
  return (
    <main className="page-shell">
      <section className="container-x pt-32 sm:pt-40">
        <div className="max-w-3xl">
          <p className="eyebrow">Examples · source of truth</p>
          <h1 className="page-title">Start with a pattern. Then measure your own.</h1>
          <p className="page-lede">The examples below are kept on the product site so you can understand the integration before opening the repository. The full runnable files remain in source control.</p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a className="btn-primary light:btn-light-primary" href="?page=onboarding">Set up the repo <span aria-hidden>→</span></a>
            <a className="btn-ghost light:btn-light-ghost" href="?page=docs">Read the contract <span aria-hidden>→</span></a>
          </div>
        </div>
      </section>
      <section className="container-x mt-16 grid gap-6 pb-24 lg:grid-cols-2">
        {examples.map((example) => (
          <article className="surface overflow-hidden light:surface-light" key={example.number}>
            <div className="p-6">
              <div className="flex items-center gap-3"><span className="font-mono text-xs text-blue-300 light:text-blue-700">{example.number}</span><h2 className="text-xl font-semibold tracking-tight">{example.title}</h2></div>
              <p className="mt-3 max-w-xl text-sm leading-6 text-ink-400 light:text-ink-600">{example.body}</p>
            </div>
            <div className="border-t border-white/[0.08] bg-[#0a0b10] light:border-ink-900/[0.08] light:bg-[#f7f8fb]"><pre className="overflow-x-auto p-6 font-mono text-xs leading-6 text-ink-200 light:text-ink-700"><code>{example.code}</code></pre></div>
          </article>
        ))}
      </section>
    </main>
  );
}
