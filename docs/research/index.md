# Researcher notes

These documents cover the algorithm, math, and reproduction details. They
are aimed at people who want to understand *why* JoLT works and how to
extend or modify it.

## Contents

- [Math](math.md) — restating the paper's Eqs. 1-4 in our notation.
- [Algorithm walkthrough](algorithm.md) — end-to-end JoLT + FlashJoLT.
- [Spectral motivation](spectral_motivation.md) — what the paper calls
  "spectral structure".
- [Comparison with baselines](comparison_with_baselines.md) — Palu, xKV,
  KIVI, TurboQuant.
- [Free zone](free_zone.md) — the near-lossless regime and its
  boundaries.
- [Ablations](ablations.md) — A1-A5 from paper Section 7.
- [Extending](extending.md) — TT, t-SVD, etc.
- [Reproduction notes](reproduction_notes.md) — how the paper's numbers
  were obtained and what we tested locally.
- [Paper notes](paper_notes.md) — caveats and limitations.