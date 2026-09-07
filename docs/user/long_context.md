# Long context

KV-cache compression matters most when the cache is large, i.e., at long
context. `kvcompress` exposes two long-context tools:

1. **Needle-in-haystack** (`scripts/run_needle_haystack.py`): a single-fact
   retrieval test that buries a known phrase in a long context and asks
   the model to repeat it.
2. **RULER-style multi-needle** (planned for v0.2; the paper reports 100%
   accuracy at 8K context for Mistral at 2-3×).

## Needle-in-haystack

```bash
python -m scripts.run_needle_haystack \
    --model gpt2 \
    --context-length 512 \
    --needle "The secret code is 12345." \
    --max-new 20
```

The script runs the same prompt through both `identity` (no compression)
and your chosen method and prints both outputs plus the compression stats.

## Expected results

On GPT-2 (small), the needle is rarely retrieved exactly because the
model wasn't trained for it; the script is mainly a smoke test for
*does compression break the long-context pipeline?*. On Mistral-7B /
LLaMA-2-13B, the paper shows 100% retrieval accuracy at 2-3× on
RULER single-needle at 8K context.

## Production long-context

For production deployments at 16K-128K tokens:

- Use `flashjolt` (not `jolt`) — the speedup matters at long contexts.
- Pick `target_memory="33%"` (3×) for the free zone.
- For batch / prompt-caching where many prefixes are resident at once,
  consider `target_memory="20%"` (5×) — values dominate the cache at
  long contexts and JoLT is robust to 5× on GQA.

See `benchmarks/long_context.py` (planned) for a fuller harness.