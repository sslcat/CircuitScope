# Supplementary source intervention
I added this comparison before running GPT-2 to distinguish the contribution of a specific earlier token from the total effect of its receiving head. The primary decision criteria stay as recorded in the initial protocol.

For each individual candidate and its assigned same-layer control I subtract the bound successor's contribution from the final-query head output. In clean prompts this source is the target. In swapped prompts it is the foil. I then repeat the intervention at the other answer's position as a negative control. For selected source $j$ the replacement is

$$\tilde z_q^h=z_q^h-a^h_{qj}v_j^h.$$

I keep the softmax denominator and all other source terms fixed. I use the decrease in the paired margin gap as the endpoint. A larger loss from removing the bound source than from removing the other source supports a specific value route from that token through the head. It leaves the mechanism that placed attention there unresolved. For low-precision runs I multiply and subtract in FP32 and cast the donor back to the model dtype. The reconstruction test compares the edited vector against an independently computed sum of all surviving terms.
