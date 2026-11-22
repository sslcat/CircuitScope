# Scientific and publication audit

I audited the experiment at `efa5804`. The primary endpoint, head groups, intervention code and archived results are those of the original study. I used a separate implementation and a fresh environment for verification. Codex assisted with both the original study and this audit.

## What the head edits change

I traced the Hugging Face GPT-2 implementation from the QKV projection to the attention output projection. It divides the 768 features into 12 contiguous 64-feature heads. Eager attention returns the weighted values in position-major, head-major order before flattening them for `c_proj`. CircuitScope hooks this **input**, reshapes it to [batch, 26, 12, 64], and replaces only the selected head slices at position 25. Layers and heads are zero-based. L5H5 uses channels 320–383 of block 5, and L6H9 uses channels 576–639 of block 6. The shared projection bias is applied once. [Pinned GPT-2 implementation](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/modeling_gpt2.py) · [Production hook](../circuitscope/model.py)

Each donor comes from the intact clean row at the same batch and query index as the swapped recipient. The prompt generator changes exactly two earlier cues, preserves the answer indices and keeps the final query identical. In a joint patch, the later donor still comes from the original clean run. Other heads in the later block may respond naturally to the earlier intervention. Hooks are scoped to one forward pass, edits use clones, dropout is disabled and the KV cache is off.

I separately reconstructed each selected head’s final-query attention from raw Q, K and V, then added the replacement delta **after** the output projection using only the selected rows of its weight matrix. On the first four held-out seed-41 pairs in pretrained CPU FP32, the largest final-logit differences from the production implementation were 0.0000382 under joint patching and 0.0000267 under joint zeroing. The reference edit preserved every earlier-position logit exactly. The test suite also checks individual head and position isolation, weighted-source reconstruction, native-forward agreement and hook cleanup after an exception. I found no evidence of accidental transfer outside the specified sites. [Independent calculation](../tests/test_intervention_reference.py) · [Existing site checks](../tests/test_study.py)

## Selection and controls

The first Git commit (`4c32df3`) fixes the discovery ranking, top-two rule, control draw, seeds, intervention sites, replacement rules and endpoints. The source-term comparison appears in `5ce58f1`, before the results commit. I checked that the original config and protocols have identical contents in the results commit. The runner computes its complete 12 × 12 attention score matrix using only the 48 seed-1701 pairs, writes the selected groups, then generates the 192 evaluation pairs. There is no evaluation-to-selection feedback in this code. I regenerated the prompts, checked the archived trial IDs and positions, and found 240 distinct prefix token sets, including across the split. Sorting the saved discovery scores recovers both candidates. Git establishes the recorded order. It cannot independently establish the absence of unrecorded exploratory work. [Protocol](protocol.md) · [Runner](../circuitscope/run.py) · [Artifact checks](../tests/test_archived_results.py)

| Comparison | What I can conclude | Important limit |
| --- | --- | --- |
| L5H10 and L6H11 | One seeded noncandidate per selected layer controls for depth and intervention size. | The draw does not match activation norm, attention score or causal importance. |
| L10H2 and L8H2 | Two further heads sampled uniformly without replacement from the remaining noncandidates give a global comparison of the same size. | One draw cannot estimate the distribution across random head pairs. L5H5’s ablation advantage over its assigned L10H2 control includes zero. |
| All 12 query heads in layer 5 or 6 | These edits put each candidate’s effect in the context of a larger perturbation of its layer. | The MLP and residual path remain available. Nonlinear effects preclude a percentage decomposition by head. |
| Swapped cues and novel query | The swap fixes answer frequency, position and query identity. The novel cue tests signed preference without a previous cue binding. | Moving the cues changes surrounding representations. Opposing changes can cancel in the novel-query mean. |
| Self-patch and final-residual transplant | These bound implementation error at the identity and full-restoration endpoints. | They test intervention mechanics. They cannot establish selectivity of particular heads. |

The published control seed was 208. The controls were drawn before held-out scoring and were distinct from each other and both candidates. This supports a comparison with these four controls. It leaves open whether other heads would show similar effects.

## Uncertainty and recovery

I independently reconstructed all 76 endpoint means, seed means and 95% interval endpoints from the trial CSV. I represented each bootstrap replicate by a count for each whole prompt row and multiplied those counts by the matrix of all outcomes. This preserves the covariance of clean, swap, ablation, patch and control measurements. The results matched the saved statistics to absolute tolerance 1e−10. [Trial table](../results/trials.csv) · [Independent interval check](../tests/test_archived_results.py)

For recovery I divided the mean shift from the swapped baseline (5.7062 nats) by the mean intact clean-minus-swap gap (13.7111 nats). Recomputing both means on the same resampled rows gave **41.6% [39.0%, 44.2%]**. The analogous ablation fraction, **15.5% [13.4%, 17.6%]**, uses the change in the gap after zeroing in both contexts. Each denominator is shared across the two estimates. The numerators represent different counterfactuals. Only 76 of 192 patched prompts actually favored the clean answer.

Sampling a fresh prefix for each trial makes the pair a defensible unit for inference under this generator. Repeated token IDs across trials are possible by design. Each pair supplies one final query, and paired conditions are kept together. Leaving out each seed in turn gave intact-gap means of 13.31–14.06 nats, joint ablation means of 2.05–2.22 and joint patch means of 5.46–6.01. These ranges describe sensitivity to the three batches of prompts. They are not confidence intervals.

The bootstrap conditions on one template, checkpoint, selected set and control draw. Seeds restart one generator and provide little information about variation across tasks or model instances. Multiple endpoints use pointwise intervals, with singles and source comparisons interpreted descriptively. Near ceiling, percentile intervals can be too narrow. In particular, zero observed failures yields a degenerate bootstrap interval and still permits failures on future prompts. An exact replay tests reproducibility and adds no independent observations.

## Relation to earlier work

Wang et al. explicitly identify heads 5.5 and 6.9 in GPT-2 Small. Their Appendix F examines both prefix matching and output alignment with the next token on repeated random sequences. Olsson et al. give the broader induction-head formulation. The head identities and copying association are replications. CircuitScope reports measurements under a particular cue swap, source-term removal and head-output interventions. Priority for those methods or observations has not been established. [Wang et al., ICLR 2023, Appendix F](https://openreview.net/pdf?id=NpsVSN6o4ul) · [Olsson et al., 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)

## Fresh-environment reproduction

I cloned the local results commit into an empty directory and created a Python 3.12.15 environment with system site packages disabled. I installed the documented CUDA wheel and requirements from their package indexes. All six direct versions matched. One transitive dependency, filelock, resolved to 4.2.4 instead of the archived 4.3.0. The installed environment passed `pip check`. The checkpoint command started with an empty weights directory, downloaded the two files from revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, and verified both SHA-256 digests.

With Hugging Face network access disabled for inference, all **14 CPU tests passed**. I ran the original study code and config on the RTX 4070 SUPER in FP16. The GPU runner took **24.24 s** and peaked at **308.37 MiB allocated / 318.00 MiB reserved**. I also ran the figure generator and eight-prompt CPU FP32 check. This was a clean software environment on the original hardware. [Reproduction commands](reproduction.md)

| Comparison to the committed evidence | Outcome |
| --- | --- |
| All 192 trial rows, all summary statistics and frozen head groups | Each output file matched byte for byte |
| Discovery scores, example IDs, positions and clean/swap patterns | All 28 NPZ arrays matched element for element |
| Four PNGs regenerated from the new measurements | Identical pixels and file bytes for all four |
| CPU FP32 precision-check JSON | Identical bytes |
| Primary run metadata | Only elapsed time differed |

The SHA-256 of both trial tables was `0ab85185635def291fd02c2a3db3693875447c9edbe2b6bca1880e6aac3c1333`. I traced the README and report back to the trial table, discovery arrays, summary, figures and run metadata. The 90 clean two-choice wins that missed top-1 follow directly from the joint per-trial outcomes.

## Publication materials

The top-level MIT license covers CircuitScope’s own code and documentation. The synthetic inputs are generated from seeds. Users fetch the separately licensed public GPT-2 checkpoint themselves. The repository contains numerical derivatives and plots. [Repository license](../LICENSE) · [GPT-2 license](https://github.com/openai/gpt-2/blob/master/LICENSE) · [Model card](https://huggingface.co/openai-community/gpt2)

I checked the wheel metadata and license files for the six direct requirements. PyTorch and NumPy use BSD-3-Clause, Transformers and safetensors use Apache-2.0, pytest uses MIT, and Matplotlib uses its PSF-based license. I reviewed installed-package metadata for the recorded transitive environment as well. It includes distinct terms such as MPL-2.0 for certifi, and the binary wheels carry third-party notices. The source repository distributes none of these wheels, CUDA libraries or model weights. The licenses and notices for those components remain relevant to anyone packaging a binary environment. [Resolved environment](../results/environment.txt) · [Matplotlib license](https://matplotlib.org/stable/project/license.html)

The CPU workflow installs a CPU PyTorch wheel on Ubuntu, checks dependency consistency, and runs the tests offline with read-only repository permissions. Hosted CI execution awaits publication. [CI workflow](../.github/workflows/test.yml)

I inspected the versioned file list, numerical archive members and figure metadata. A targeted scan of all three pre-audit commits and the proposed text files found no common credential patterns, private keys or absolute workstation paths. The largest tracked artifact is the 277,305-byte discovery archive. Model weights, environments, caches and duplicate reproduction outputs are ignored.
