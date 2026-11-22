# Study protocol
I specified these decisions before inspecting any model outputs. I use zero-based layer and head indices.

## Task and hypotheses
I sample 24 distinct GPT-2 vocabulary IDs uniformly without replacement from [256, 20000). I prepend endoftext as BOS. I choose two disjoint adjacent pairs, with both cues uniformly ordered by random assignment. I repeat the first cue at the end of the prompt. Its earlier successor is the target, and the other cue's successor is the foil. On the corrupted prompt I exchange the two earlier cues. Both answer tokens stay fixed at the same positions. A neutral prompt replaces the final cue with an unseen token. The target differs from the final token on every trial.

I expect the intact model to prefer the earlier successor of whichever cue is queried. For each prompt I score the target minus foil logit margin M with the clean token labels held fixed. My primary behavioral endpoint is D = M(clean) - M(swap), in nats. The exchangeable no-association reference is mean D = 0 and 50% pairwise choice accuracy. I will measure the intact baseline, target log probability, full-vocabulary top-1 accuracy, and neutral performance before interpreting interventions.

## Selection and confirmatory analysis
On 48 discovery pairs (seed 1701) I rank all heads by the mean clean-minus-swap change in attention from the final query to the fixed target position. I take the top two, resolving ties by layer then head. I choose a distinct noncandidate in each selected head's layer and two further global noncandidates uniformly without replacement using seed 208. I freeze these choices before evaluation.

On 64 new pairs for each of seeds 41, 43 and 47, I zero a head's value-weighted output at the final query just before its output projection in both contexts. The primary ablation effect is the reduction in D. I patch the same site from clean into swap and measure the increase in M(swap). I compare each candidate, the candidate pair and the corresponding control sets. I also zero all heads of each selected layer at this position. I evaluate neutral-prompt margin shifts. Pair-minus-sum departures measure interaction on the margin scale. No intervention alters model parameters.

## Decision rules and uncertainty
I require a positive lower endpoint of a paired 95% bootstrap interval for baseline D. For the two-head causal claim I also require positive lower endpoints for both joint intervention effects and for each joint effect minus its same-layer and global control counterpart. If the upper endpoint of a required contrast is at or below zero, that prediction fails. Intervals spanning zero leave the prediction unresolved. Singles and interactions are descriptive. A single head can be a smaller tested explanation if it passes the same comparisons against its assigned controls. I do not infer global minimality from this subset search.

I resample paired trials within each seed 2,000 times and average seed means with equal weight. Intervals describe prompt sampling conditional on these seeds and the fixed discovery outcome. I show seed means and failures. Intervals are pointwise. I verify alignment, identity patches, and restoration by replacing the complete final residual. I use inference mode, eager causal attention and a bounded batch size. Resource measurements cover the experiment runner and exclude setup and plotting.
