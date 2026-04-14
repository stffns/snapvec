---
tags: [ml, reading, vector-search]
---

# Vector search — reading list

## Papers I keep coming back to

- **Johnson-Lindenstrauss lemma** — the reason random projections preserve
  distances.  The recipe behind every random-rotation trick in ANN.
- **Product Quantization** (Jégou et al., 2010) — split a vector into M
  sub-vectors, quantise each with k-means.  Requires training; optimal
  for the distribution it was trained on, worse off-distribution.
- **TurboQuant** (Zandieh et al., ICLR 2026) — rotate first (Hadamard),
  then apply optimal scalar quantisation for N(0,1).  No training
  required because the rotation makes any distribution look Gaussian.
- **HNSW** (Malkov & Yashunin, 2016) — graph-based ANN.  Sub-linear
  scan, great at N > 10M, loses to flat search on tiny indices.

## Mental model

If N is small (< 1M), a flat compressed scan beats graph indices on
cold-start and predictable latency.  Once N > 10M, the graph wins.
The break-even depends on your embedding dim and your cache behaviour
more than on the algorithm.

## Quantisation trade-offs

Bits per coordinate governs the MSE of the quantised reconstruction.
4 bits is the sweet spot for most RAG workloads (~95% recall@10 on real
embeddings).  2 bits is fine if scale matters more than precision —
you lose ~10 points of recall in exchange for 2× smaller storage.
