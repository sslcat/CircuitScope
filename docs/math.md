# Mathematics of the experiment

## Attention inside GPT-2 Small

Let $X\in\mathbb R^{T\times d}$ be the residual stream entering a block. Each row represents one token position. GPT-2 starts with token and learned position embeddings. Here $d=768$, with 12 heads of width $d_h=64$. For a head $h$, let $U=\mathrm{LN}_1(X)$. The learned affine maps form queries, keys and values.

$$Q_h=UW_h^Q+b_h^Q,\quad K_h=UW_h^K+b_h^K,\quad V_h=UW_h^V+b_h^V.$$

A query represents what the current position looks for, a key supplies a comparison vector at a potential source, and a value carries the information aggregated from that source. At destination $i$, causal masking excludes later positions. Scaled dot products and a row-wise softmax give

$$s^h_{ij}=q^h_i(k^h_j)^\top/\sqrt{d_h},\qquad
a^h_{ij}=\begin{cases}\exp(s^h_{ij})/\sum_{k\le i}\exp(s^h_{ik})&j\le i\\0&j>i.\end{cases}$$

The head writes a weighted sum $z^h_i=\sum_{j\le i}a^h_{ij}v^h_j$. Concatenating heads and multiplying by the attention output matrix is equivalent to summing their projected writes. With dropout disabled, one complete pre-normalized block computes

$$Y_i=X_i+b^O+\sum_h z_i^h W_h^O,\qquad
X'_i=Y_i+\mathrm{MLP}(\mathrm{LN}_2(Y_i)).$$

The residual stream accumulates these writes. After the final block, next-token logits are $\ell=\mathrm{LN}_f(X'_{q})W_U$, where $q=T-1$. GPT-2 ties the unembedding to the token embedding. Later blocks and normalization can transform a write, so the ultimate change in a logit can differ from a head's immediate projection. The forward function in [model.py](../circuitscope/model.py) reads attention weights from eager GPT-2 attention and reshapes the native input of the attention output projection into batch, position, head and head-width axes.

## Task and scores

For fixed clean target $t$ and foil $f$, define $p(k)=\exp(\ell_k)/\sum_v\exp(\ell_v)$ and target loss $L=-\log p(t)$. The two-choice margin in nats is

$$M=\log p(t)-\log p(f)=\ell_t-\ell_f.$$

The clean context $C$ binds the final cue to $t$. The swap $S$ binds it to $f$. Both answers have identical positions and counts across these two contexts. The neutral query is novel. The paired behavioral gap $D=M(C)-M(S)$ measures the shift in relative preference. The target loss saving $L(S)-L(C)$ equals $\log p_C(t)-\log p_S(t)$. Pairwise accuracy counts $M>0$ as one and ties as one half. Top-1 compares the target to the entire vocabulary. All swap metrics keep the clean labels. The neutral comparator and the fixed IDs are implemented in [task.py](../circuitscope/task.py).

For discovery, $j_t$ is the earlier target position. Rank heads by the paired attention score

$$I_h=\mathbb E[a^h_{q,j_t}(C)-a^h_{q,j_t}(S)].$$

This is a single-query induction score. An attention head that tracks a fixed position equally in both contexts scores zero. Its attention can still be large even if its value carries little relevant information.

## Interventions

For set $H$, zeroing replaces $z_q^h$ with the zero vector for each $h\in H$. Other positions and the shared output bias stay intact. Denote the new margin by $M_{0H}$. To measure dependence on this site, use

$$A_H=D-[M_{0H}(C)-M_{0H}(S)].$$

A clean-to-swap patch replaces $z_q^h(S)$ with its aligned clean value $z_q^h(C)$. Denote the patched margin by $M_{S\leftarrow C,H}$. The restoration effect is

$$P_H=M_{S\leftarrow C,H}-M(S).$$

A positive effect moves the relative prediction toward the clean answer. The reported fraction restored is a ratio of means $\bar P_H/\bar D$. All downstream operations run normally. In a multilayer joint patch, each donor is the original clean activation, even if an earlier patch would have changed the later recipient. A whole-layer ablation here zeros all attention heads at the query, leaving the block's MLP and residual path intact. Controls and the effect calculations are in [run.py](../circuitscope/run.py).

For either intervention effect $E$, $E_{\{a,b\}}-E_{\{a\}}-E_{\{b\}}$ measures departure from additivity on this scale. It can arise through downstream nonlinearities or shared information. It does not identify a directed edge.

The supplementary source intervention removes a single summand from the attention output: $z_q^h-a^h_{qj}v_j^h$. Subtracting this term keeps the other source weights and values fixed. See the [source-intervention specification](source_intervention.md) for the paired comparisons.

## Inference

For $S$ equally weighted seeds with $n_s$ trials, $\hat\mu=S^{-1}\sum_s n_s^{-1}\sum_i e_{si}$. Draw $n_s$ trial indices with replacement within each seed, reuse the same indices for all paired quantities, and take the 2.5th and 97.5th percentiles of the resulting mean estimates. This estimates prompt-sampling uncertainty for a fixed checkpoint and discovery selection. [analysis.py](../circuitscope/analysis.py)

Attention alignment alone cannot establish that changing a site changes the prediction. Ablation tests whether a site's ordinary output contributes under a specified replacement. Patching tests whether differences carried at the site can shift the prediction in a particular recipient context. Both manipulate the computation directly. Neither identifies all upstream sources of the information, and both can create internal states outside the model's usual distribution.
