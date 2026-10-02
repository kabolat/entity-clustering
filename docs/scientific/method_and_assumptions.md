# Method and assumptions

The method clusters PV **systems**, not individual daily profiles. Each system
contributes its complete daily 96-point profiles. Pooled profiles are assigned
to k-means words; each system becomes a word-count document; LDA maps that
document to positive Dirichlet parameters. Missing daily profiles are excluded
from its document, so sparse systems have correspondingly less concentrated
entity distributions.

The configured distance is symmetric KL divergence or Bhattacharyya distance
between the Dirichlet distributions. Agglomerative clustering with average or
complete linkage operates on that precomputed matrix.

For each cluster, a pointwise set of quantiles represents its members. The
`vanilla` score includes the target system in its representation; the
`leave_one_out` score removes it. Lower quantile loss is better. Configurations
with clusters smaller than the configured minimum are saved as skipped, rather
than evaluated with an undefined leave-one-out representation.

Capacity normalisation follows the historical implementation: negative power is
set to zero, non-zero values are floored at `1e-3`, then profiles are divided by
the estimated AC capacity and multiplied by 1000.
