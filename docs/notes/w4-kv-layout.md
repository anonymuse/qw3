# W4 note — does the frozen KV layout survive M3 and 32K contexts?

Analysis only; no contract edits. Frozen (ADR-005 §1): per layer, two f32
buffers `[n_kv_heads, max_ctx, head_dim]` (K and V).

## Sizes

Both real models have identical KV geometry (`n_kv_heads = 4`, `head_dim =
128`): K+V = 1024 elements per position per layer → 4 KiB/pos f32, 2 KiB f16.

| K+V per layer | 8K ctx | 32K ctx |
|---|---|---|
| f32 | 32 MiB | 128 MiB |
| f16 | 16 MiB | 64 MiB |

| K+V total | 8K f32 | 32K f32 | 8K f16 | 32K f16 |
|---|---|---|---|---|
| 30B (48 layers) | 1.5 GiB | 6 GiB | 0.75 GiB | 3 GiB |
| 235B (94 layers) | 2.94 GiB | 11.75 GiB | 1.47 GiB | 5.87 GiB |

## M3 distributed split (per-layer caches on different nodes)

Works as-is. The cache is already a **per-layer** object pair: a node hosting
layers `i..j` allocates only those caches, `kvAppend`/`gqaAttention` for a
layer run entirely on the node that owns it, and only activations
(`[n_tokens, hidden]`) cross the wire. Nothing in the layout couples layers.
A 235B node carrying ~24 layers holds ~3 GiB of f32 KV at 32K — comfortable.

## Flags (decide BEFORE M3)

1. **KV dtype is frozen f32 and `AttnArgs`/`KvAppendArgs` carry no cache dtype
   field.** Capacity is fine (table above); decode *bandwidth* is the real
   cost: every decoded token streams the whole valid cache once per layer — at
   32K that is up to 128 MiB/layer, ~12 GiB per token across all 94 layers at
   f32. f16 KV halves it (and is the llama.cpp default). Moving later means a
   contract amendment (dtype field or a frozen-f16 rule), fixture
   regeneration, and shader variants — if f16 KV is wanted for M3 performance
   targets, amend now while only one attention kernel exists.
2. **Head-major means no in-place context growth.** Each kv head's slab is
   contiguous over `max_ctx`, so growing `max_ctx` needs a re-strided copy.
   Non-issue if caches are allocated up front at the model `max_ctx` — but
   that makes 32K·f32 = 128 MiB/layer resident from token 0. A runtime
   `max_ctx` knob below the model maximum keeps short sessions cheap; no
   contract change needed (`max_ctx` is already a per-call argument).
3. **Paging/eviction would be awkward** in head-major layout: a page of
   consecutive positions is `n_kv_heads` disjoint byte ranges. Token-major
   `[max_ctx, n_kv_heads, head_dim]` would page contiguously, but it scatters
   the per-head K stream that the attention inner loop reads sequentially —
   the hot path the frozen layout gets right. M3 has no paging requirement, so
   keep the layout; revisit only if paged/sliding-window KV enters scope.

**Verdict:** the frozen layout supports the M3 per-layer split and 32K
contexts without change. The one item worth an ADR-005 amendment decision
before M3 is KV dtype (f32 vs f16): it is load-bearing for decode bandwidth
and gets more expensive to change with every kernel added.
