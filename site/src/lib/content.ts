export const site = {
  name: 'kvfold',
  tagline: 'Universal plug-and-play KV cache compression.',
  description:
    'kvfold compresses the KV cache of any decoder-only LLM with a single function call. JoLT, Flash, int4, fp8 — every modern strategy behind one tiny, unified interface.',
  repo: 'sachncs/kvcompress',
  author: 'sachin',
  authorEmail: 'sachncs@gmail.com',
  license: 'Apache-2.0',
  paper: {
    title: 'A JoLT for the KV Cache: Near-Lossless KV Cache Compression via Joint Tucker and JL-Residual Allocation for LLMs',
    authors: 'Krishnan, R. & Schulz, V.',
    year: '2026',
    arxiv: '2607.12550',
    url: 'https://arxiv.org/abs/2607.12550',
  },
} as const;

export const navLinks = [
  { label: 'Methods', href: '#methods' },
  { label: 'How it works', href: '#how' },
  { label: 'Benchmarks', href: '#benchmarks' },
  { label: 'Integrations', href: '#integrations' },
  { label: 'Docs', href: 'https://github.com/sachncs/kvcompress#readme' },
];

export const hero = {
  eyebrow: 'v0.2.0 · Open Source · Apache-2.0',
  title: 'KV cache, folded',
  subtitle:
    'A single function call reduces decoder-only LLM KV-cache memory — with explicit quality and compatibility trade-offs, inside Hugging Face.',
  primaryCta: { label: 'pip install kvfold', href: '#install' },
  secondaryCta: { label: 'Read the paper', href: 'https://arxiv.org/abs/2607.12550' },
  codeLines: [
    { text: 'from', tone: 'kw' },
    { text: ' kvfold ' },
    { text: 'import', tone: 'kw' },
    { text: ' enable_compression\n' },
    { text: '\n' },
    { text: 'enable_compression(\n    model,\n    method=', tone: 'kw' },
    { text: '"flash"', tone: 'str' },
    { text: ',\n    target_memory=', tone: 'kw' },
    { text: '"25%"', tone: 'str' },
    { text: ',\n', tone: 'kw' },
    { text: ')\n' },
    { text: 'model.generate(', tone: 'kw' },
    { text: '...', tone: 'str' },
    { text: ')', tone: 'kw' },
  ] as { text: string; tone?: 'kw' | 'str' | 'comment' | 'fn' }[],
};

export const valueProps = [
  {
    title: 'One call. Every model.',
    description:
      'A single function call wraps any Hugging Face causal LM. No model surgery, no rewriting forward passes.',
  },
  {
    title: 'Built around the JoLT paper.',
    description:
      'Faithful implementation of Krishnan & Schulz (2026) — partial Tucker decomposition with a JL-rotated low-bit residual.',
  },
  {
    title: 'Deployment paths, clearly scoped.',
    description:
      'vLLM export/import, an offload integration, and optional Triton kernels for NVIDIA hardware.',
  },
  {
    title: 'Honest failure modes.',
    description:
      'Cache, allocator, and adapter boundaries expose explicit contracts and failure context.',
  },
];

export const methods = [
  {
    name: 'jolt',
    label: 'JoLT',
    description:
      'Partial Tucker decomposition with a JL-rotated low-bit residual. Pin head and layer modes; truncate token and feature modes.',
    ratio: '2–4×',
    quality: 'high fidelity in tested regimes',
  },
  {
    name: 'flash',
    label: 'Flash',
    description:
      'The randomized-SVD fast variant. Context-aware q_cap. ~5–13× faster than JoLT at matched quality.',
    ratio: '2–4×',
    quality: 'matches JoLT',
    recommended: true,
  },
  {
    name: 'low',
    label: 'Low',
    description:
      'Matrix-SVD baseline. Clean, predictable, sharp at low ranks — useful for ablations and tight budgets.',
    ratio: '3–8×',
    quality: 'tunable rank',
  },
  {
    name: 'int4',
    label: 'IntQuant',
    description:
      'Per-channel uniform int quantization. Tiny payloads, slightly higher error. Excellent floor for hybrid strategies.',
    ratio: '5–8×',
    quality: 'quantization-noise',
  },
  {
    name: 'fp8',
    label: 'Float8',
    description:
      'IEEE-style E4M3 / E5M2. Drop-in dtype halving with proper NaN/Inf handling and per-tensor scales.',
    ratio: '2×',
    quality: 'exact dtype',
  },
  {
    name: 'pass',
    label: 'Pass',
    description:
      'Passthrough baseline. Useful for A/B tests, regression guards, and confirming a measurement pipeline.',
    ratio: '1×',
    quality: 'baseline',
  },
];

export const howSteps = [
  {
    n: '01',
    title: 'Wrap the cache.',
    body:
      'enable_compression patches transformers.DynamicCache in-place. Attention forward passes write into a compressed cache without modification.',
  },
  {
    n: '02',
    title: 'Pick a strategy.',
    body:
      'Choose JoLT, Flash, int4, fp8, or low-rank. Or plug your own compressor in — the interface is three methods.',
  },
  {
    n: '03',
    title: 'Set a budget.',
    body:
      'Pass a ratio (3.0) or a memory target ("25%"). A joint Lagrangian allocator bisects λ to hit the global byte budget across layers.',
  },
  {
    n: '04',
    title: 'Generate.',
    body:
      'Reconstruction is a fused einsum on read. Stats are exposed via the handle. Disable at any time to restore baseline behaviour.',
  },
];

export type MemoryRow = { name: string; achieved: string; size: string };
export type SpeedRow = { name: string; compress: string; decompress: string };
export type FidelityRow = { name: string; kerr: string; verr: string; ratio: string };

export const memoryBenchmark = {
  title: 'Memory',
  method: 'JoLT · Flash · Low',
  rows: [
    { name: 'Identity', achieved: '2.00×', size: '524 KB' },
    { name: 'JoLT 3×', achieved: '3.96×', size: '258 KB' },
    { name: 'Flash 3×', achieved: '3.96×', size: '258 KB' },
    { name: 'Low-rank 4×', achieved: '7.75×', size: '132 KB' },
  ] satisfies MemoryRow[],
  note: '1 MB synthetic K+V tensor, m=8, T=256, dh=64.',
};

export const speedBenchmark = {
  title: 'Speed',
  method: 'compress / decompress',
  rows: [
    { name: 'JoLT', compress: '21.5 ms', decompress: '0.44 ms' },
    { name: 'Flash', compress: '17.6 ms', decompress: '0.43 ms' },
    { name: 'Low-rank', compress: '4.0 ms', decompress: '0.21 ms' },
  ] satisfies SpeedRow[],
  note: 'target_ratio = 3.0, single-thread CPU.',
};

export const fidelityBenchmark = {
  title: 'Fidelity',
  method: 'K / V relative error',
  rows: [
    { name: 'JoLT', kerr: '0.235', verr: '0.215', ratio: '3.81×' },
    { name: 'Flash', kerr: '0.235', verr: '0.216', ratio: '3.81×' },
    { name: 'Low-rank-64', kerr: '0.013', verr: '0.014', ratio: '3.76×' },
    { name: 'Int4 per-ch.', kerr: '0.200', verr: '0.186', ratio: '7.94×' },
  ] satisfies FidelityRow[],
  note: 'm=8, T=256, dh=128, target_ratio = 2.0.',
};

export const benchmarkEvidence = {
  docsUrl: 'https://github.com/sachncs/kvcompress/blob/master/docs/benchmarks/running.md',
  memoryCommand: 'python -m scripts.run_memory_benchmark --T 1024 --dh 128 --ratios 2 3 4 5 8',
  speedCommand: 'python -m scripts.run_speed_benchmark --T 1024 --dh 128 --ratio 3',
  fidelityCommand: 'python -m scripts.run_table2_reconstruction --T 1024 --ratio 2.0',
} as const;

export const integrations = [
  {
    name: 'Hugging Face',
    description:
      'Transparent DynamicCache interception. Works with GPT-2, Llama, Mistral, Qwen, and any decoder-only causal LM that uses cache_implementation.',
    badge: 'first-class',
  },
  {
    name: 'vLLM',
    description:
      'Shape A (export_kv / import_kv) for everywhere vLLM runs. Shape B (Offload subclass) for production GPU deployments.',
    badge: 'two shapes',
  },
  {
    name: 'Triton',
    description:
      'Optional fused Tucker reconstruction, JL projection, and INT8 quantize kernels. Falls back to PyTorch einsum on non-NVIDIA systems.',
    badge: 'optional',
  },
];

export const codeSnippets = {
  install: `# Install from PyPI
pip install kvfold

# Optional extras
pip install "kvfold[triton]"   # Triton kernels for reconstruction / JL
pip install "kvfold[vllm]"     # vLLM adapter
pip install "kvfold[bench]"    # matplotlib, pandas, datasets
pip install "kvfold[dev]"      # pytest, ruff, hypothesis, mypy`,
  quickstart: `import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvfold import enable_compression

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# One call. The cache is compressed transparently during generate().
handle = enable_compression(model, method="flash", target_memory="25%")

ids = tok("Hello, my name is", return_tensors="pt").input_ids.to(model.device)
out = model.generate(ids, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0]))

print("compression ratio:", handle.stats.compression_ratio)
handle.disable()   # restore baseline behaviour`,
  methods: `enable_compression(model, method="jolt",  ratio=3.0)
enable_compression(model, method="flash", ratio=2.5)
enable_compression(model, method="low",   rank=128)
enable_compression(model, method="int4",  per_channel=True)
enable_compression(model, method="fp8")
enable_compression(model, method="pass")`,
} as const;

export const faqs = [
  {
    q: 'Does it change the model?',
    a: 'No weights are modified. kvfold swaps the cache implementation under the hood via Hugging Face’s cache_implementation mechanism.',
  },
  {
    q: 'Will my generation quality drop?',
    a: 'Quality depends on method, ratio, model, sequence length, and workload. The repository’s synthetic CPU benchmarks show the trade-offs; validate on your own model before rollout.',
  },
  {
    q: 'Does it work with vLLM?',
    a: 'Yes — Shape A (export_kv / import_kv) works everywhere vLLM runs. Shape B (a subclass of vllm.v1.kv_offload.KVCacheOffloadWorker) targets production GPU deployments.',
  },
  {
    q: 'What models are supported?',
    a: 'Any decoder-only causal LM. The HF adapter handles family-specific cache layout shims via a registry. Encoder-decoder and Mamba-style models are out of scope.',
  },
  {
    q: 'Is this the official paper code?',
    a: 'No. kvfold is an independent Python re-implementation of the algorithm by sachin. The theoretical results belong to the paper’s authors; cite the paper, not this repo.',
  },
];

export const footerLinks = {
  product: [
    { label: 'Methods', href: '#methods' },
    { label: 'How it works', href: '#how' },
    { label: 'Benchmarks', href: '#benchmarks' },
    { label: 'Integrations', href: '#integrations' },
  ],
  resources: [
    { label: 'Documentation', href: 'https://github.com/sachncs/kvcompress#readme' },
    { label: 'Research notes', href: 'https://github.com/sachncs/kvcompress/tree/master/docs/research' },
    { label: 'API reference', href: 'https://github.com/sachncs/kvcompress/tree/master/docs/user' },
    { label: 'Examples', href: 'https://github.com/sachncs/kvcompress/tree/master/examples' },
  ],
  community: [
    { label: 'GitHub', href: 'https://github.com/sachncs/kvcompress' },
    { label: 'Issues', href: 'https://github.com/sachncs/kvcompress/issues' },
    { label: 'Discussions', href: 'https://github.com/sachncs/kvcompress/discussions' },
    { label: 'Changelog', href: 'https://github.com/sachncs/kvcompress/blob/master/CHANGELOG.md' },
  ],
  legal: [
    { label: 'License (Apache-2.0)', href: 'https://github.com/sachncs/kvcompress/blob/master/LICENSE' },
    { label: 'Security', href: 'https://github.com/sachncs/kvcompress/blob/master/SECURITY.md' },
    { label: 'Code of Conduct', href: 'https://github.com/sachncs/kvcompress/blob/master/CODE_OF_CONDUCT.md' },
    { label: 'Contributing', href: 'https://github.com/sachncs/kvcompress/blob/master/CONTRIBUTING.md' },
  ],
} as const;
