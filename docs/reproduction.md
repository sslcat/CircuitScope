# Reproducing the run

I ran this study with Python 3.12. I install the project in a fresh virtual environment from the repository root.

```sh
python -m venv .venv
```

On Windows PowerShell I activate it with `.\.venv\Scripts\Activate.ps1`. On Linux I use `source .venv/bin/activate`. I then install one PyTorch build. For the CUDA 12.8 build used on the RTX 4070 SUPER I run

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only reproduction I use the CPU wheel index in that command, `https://download.pytorch.org/whl/cpu`. After installing PyTorch my remaining steps are

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python -m circuitscope.checkpoint
python -m circuitscope.run --config config.json --device cuda --precision float16 --out reproduction
python -m circuitscope.figures reproduction
python -m circuitscope.precision_check reproduction
```

The checkpoint helper downloads a 548,105,171-byte safetensors file and the small model configuration, then verifies both SHA-256 digests. The tests instantiate a random tiny GPT-2 locally. Experiment inference and plotting run offline after setup. I keep the weights and environment out of version control.

I use `--device cpu --precision float32` to run the study on CPU. Automatic selection uses CUDA when PyTorch reports at least 500 MiB free and selects FP16 under 1,300 MiB free on CUDA. To repeat the recorded precision independent of free memory, pass the device and precision from `results/run.json` explicitly. Minor numerical differences can occur across precision, kernels and hardware. I use a new output directory for each run.

I edit a copy of `config.json` to change the prefix length, token range, discovery and test sample sizes, seeds, batch size or bootstrap draws. Batch size controls peak working memory. Each seed produces the same prompts across batch sizes. A tiny config with a single test seed and fewer prompts is suitable for a pipeline check. I use the checked-in config for scientific comparisons.

I recorded the exact Windows packages in [the environment snapshot](../results/environment.txt). The eight-prompt precision check is a separate CPU run and writes its output alongside the primary results.

## Reading the artifacts

| File | Contents |
| --- | --- |
| `results/run.json` | Exact model revision and hash, device, dtype, package versions, runtime, allocator peaks, hook checks and config |
| `results/discovery.npz` | 12 × 12 discovery score matrix, clean and swapped attention maps for the first discovery trial, its token IDs and answer positions |
| `results/groups.json` | Frozen candidate and comparison groups, ordered as [layer, head] |
| `results/trials.csv` | One row per held-out prompt pair, indexed within its seed from zero |
| `results/summary.json` | Equal-weight seed means, per-seed means and paired percentile intervals for each endpoint |
| `results/figures/` | Plots generated from the saved arrays and tables |

I keep the clean target and foil labels fixed when scoring all three contexts. In the CSV, `logp` and `foil_logp` are full-vocabulary log probabilities, `wins` counts the two-choice win including half credit for ties, and `top1` measures full-vocabulary success. `gap` is the clean-minus-swap margin and `target_logp_gain` is the target loss saving. `ablate_*` is the decrease in the gap under query-head zeroing. `patch_*` is the increase in the swap margin under a clean donor. `neutral_shift_*` is intact minus ablated neutral margin. `source_bound_*` and `source_other_*` measure the decrease in the gap when dropping one source's weighted value. `*_minus_*`, `source_specificity_*`, and `*_interaction` are within-trial contrasts. The nonpositive indicators count failures including ties.

I recover any prompt from its config, seed, and trial index using `circuitscope.task.batches`. Raw vocabulary IDs eliminate tokenizer version and text-boundary differences. The BOS token is 50256 and every prefix consists of unique IDs. I store one example attention matrix per layer and condition to keep the archive small.
