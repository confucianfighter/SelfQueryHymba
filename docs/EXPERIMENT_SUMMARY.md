# Experiment Summary

This repo was split from a larger exploratory workspace after the sentinel multitask format stabilized.

## Key Findings

- The `loss_query_hymba` model was the best-performing local architecture variant for these character-level regex and math traces.
- Mixed corpora without an explicit end marker caused generation drift into the next corpus example.
- The shared format below fixed the drift:

```text
Task: regex_v5
Input:
...

Output:
...
<END>
```

- Pure math-only continuation caused catastrophic forgetting of regex routing.
- Replay with v5 regex examples preserved regex skill while adding addition and subtraction.

## Best Local Checkpoint Before Repo Cleanup

Run name:

`2026-05-14_loss_query_hymba_sentinel_v5_add_sub_70_15_15_4000_continue_lr0005`

The checkpoint itself is intentionally not included.

Eval snapshot:

- v5 sentinel regex: `98/100`
- addition forced answer exact: `95/100`
- subtraction forced answer exact: `100/100`
- old v2 quoted-ref evaluator: `57/100`

## V7 Dream Regex And Alpine Comparison

The v7 regex curriculum moved from short homebrew commands toward a controlled natural-language "dream" regex interface. Examples include forms such as:

```text
/regex all words starting with "jbq"; start select: the rest; end select
/regex sentences that have "a8" after the text "jcje2cwt"
/regex in text: start capture 'a': 8 letters; end capture 'a' followed by "gvn"
```

The generator is `scripts/data/prepare_regex_il_v7_dream_queries.py`. Its outputs were mechanically checked with `scripts/data/validate_regex_il_v7_templates.py`, which regenerates examples from metadata, derives templates from IL, expands quoted refs, and compiles the resulting regex. The 120k v7 corpus passed this check with `120000/120000` examples valid.

`Alpine` is the `previous_loss_scalar_injected_hymba` architecture: a branched Hymba block predicts scalar next-token loss from the penultimate-layer representation, injects that predicted-loss signal back into the token stream, and reruns the branch-residual path so next-character prediction can condition on autoregressive loss-prediction context.

### Regex-Only V7 Training

Both models were trained fresh for 6000 steps on `data/sentinel_regex_v7_dream_queries_120k.txt`, with example batching, `seq_len=384`, `batch_size=8`, and `lr=0.0005`.

| Model | Params | Best eval loss | Best step | Final eval loss | Regex exact output | Regex IL exact | Regex template exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alpine | 3,973,506 | 0.1629 | 5700 | 0.1672 | 92/100 | 96/100 | 92/100 |
| FastHymba baseline | 3,724,288 | 0.1469 | 5800 | 0.1612 | 90/100 | 93/100 | 92/100 |

The plain baseline won next-character loss. Alpine was slightly better on exact held-out generation and IL exactness.

### Adding Math Replay

Both regex-trained checkpoints were then continued for 2000 steps on a 33/33/33 mix:

- 40k v7 regex examples
- 40k addition prose examples
- 40k subtraction prose examples

Corpus: `data/sentinel_regex_v7_addition_subtraction_33_33_33_120k.txt`

| Model | Mixed eval loss | Mixed accuracy | Addition answer exact | Subtraction answer exact | Regex exact output | Regex IL exact |
|---|---:|---:|---:|---:|---:|---:|
| Alpine | 0.1624 | 0.9508 | 100/100 | 100/100 | 95/100 | 99/100 |
| FastHymba baseline | 0.1239 | 0.9559 | 100/100 | 98/100 | 82/100 | 88/100 |

After math replay, the plain baseline still won next-character loss, but Alpine was clearly better on task exactness. Alpine retained regex better and handled subtraction borrow-through-zero cases that the baseline missed.

### Training Cost

Wall-clock timing from run artifacts:

| Run | Alpine | FastHymba | Alpine factor |
|---|---:|---:|---:|
| v7 fresh 6000 | 35.6 min | 21.4 min | 1.66x |
| 33/33/33 continuation 2000 | 13.2 min | 8.9 min | 1.48x |

Practical takeaway: Alpine costs roughly `1.5x-1.7x` training time in these runs. That overhead bought a qualitative difference in mixed-skill exactness, despite worse next-character loss.

## ZigZag Activation And ZagPine

`ZagPine` denotes Alpine with the corrected dynamic ZigZag activation in the loss-predict branch. This activation was first explored under the names Dynamic BasinZag / Zag basin activation. The corrected experiment replaces the loss-branch MLP `GELU` directly; earlier exploratory runs that inserted BasinZag as an extra transform after `loss_branch_norm` are not considered the clean test.

For an input tensor `x`, the dynamic ZigZag activation uses separate value and width projections:

```text
value = value_proj(x)
width_control = width_proj(x)
width = min_width + (max_width - min_width) * sigmoid(width_control)
r = value / (width + eps)
envelope = floor + (1 - floor) / (1 + abs(r) ** (2 * sharpness))
zag = zag_amp * tanh(3 * r) * exp(-0.5 * r * r)
y = value * envelope + width * zag
```

The current safe scalar defaults are:

| Field | Value |
|---|---:|
| `basin_min_width` | `0.35` |
| `basin_max_width` | `3.0` |
| `basin_floor` | `0.08` |
| `basin_zag_amp` | `0.12` |
| `basin_sharpness` | `2.0` |
| `basin_eps` | `1e-6` |

The width projection bias is initialized around 42% of the min-to-max range, so the activation starts moderately wide and does not initially choke gradients. Static ZigZag is available as `static_basin_zag`; it uses the same fixed width and serves as the no-width-modulation ablation.

### Math-Only Addition/Subtraction

The clean math-only corpus was `data/sentinel_addition_subtraction_50_50_40k.txt`, a 20k/20k sentinel mix of addition and subtraction traces. Models used `seq_len=384`, `batch_size=8`, `d_model=128`, `layers=16`, `lr=0.0002`, and Alpine loss prediction pressure `0.2`.

At a 4000-step equivalent:

| Model | Eval loss | Next-char acc | Loss-pred MSE | Addition exact | Subtraction exact |
|---|---:|---:|---:|---:|---:|
| Normal Alpine identity | `0.0860` | `0.9692` | `0.0174` | `99.8%` | `97.6%` |
| ZagPine dynamic ZigZag | `0.0918` | `0.9681` | `0.0252` | `99.8%` | `99.6%` |

Normal Alpine remained better on next-character loss and loss-prediction MSE. ZagPine was stronger on subtraction exactness, consistent with the hypothesis that dynamic per-channel width can help comparison-like operations.

### Current V7 Regex

Because the v7 generator evolved after earlier runs, the fair comparison uses fresh 6000-step runs on the current `data/sentinel_regex_v7_dream_queries_120k.txt` corpus. Both used `seq_len=384`, `batch_size=8`, `d_model=128`, `layers=16`, `lr=0.0005`, and loss prediction pressure `0.2`.

Training metrics:

| Model | Eval loss | Unweighted loss | Next-char acc | Loss-pred MSE |
|---|---:|---:|---:|---:|
| Current identity Alpine | `0.2273` | `0.2066` | `0.9357` | `0.1032` |
| ZagPine dynamic ZigZag | `0.2415` | `0.2261` | `0.9234` | `0.0770` |

Same 50-example greedy v7 exactness eval:

| Model | Exact output | IL exact | Template exact | Expanded regex exact | Parsed IL/template |
|---|---:|---:|---:|---:|---:|
| Current identity Alpine | `78%` | `82%` | `82%` | `82%` | `100%` |
| ZagPine dynamic ZigZag | `90%` | `92%` | `94%` | `94%` | `100%` |

On current v7, identity Alpine still had better next-character metrics, but ZagPine produced substantially better exact regex outputs and lower loss-prediction MSE.

### Dense Up-Split MLP ZagPine

The strongest practical ZagPine variant so far moves the dynamic ZigZag activation into every block MLP using an up-split design:

```text
up projection: 128 -> 512
value channels: 256
width-control channels: 256
down projection: 256 -> 128
```

This keeps the projection shape comparable to the normal Alpine MLP while giving each active channel a paired dynamic width signal. Attempts to use braided or masked-braided projections reduced parameter count but were slower in full training on the GTX 980 Ti setup, so the current reference path is dense up-split ZagPine.

Fresh v7 regex training for 6000 steps:

| Model | Params | Eval loss | Next-char acc | Loss-pred MSE | Exact output | IL exact | Template exact | Expanded regex exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current identity Alpine | 3.97M | `0.2273` | `93.57%` | `0.1032` | `78%` | `82%` | `82%` | `82%` |
| Dense up-split ZagPine | 3.42M | `0.2425` | `93.09%` | `0.1342` | `84%` | `86%` | `92%` | `92%` |

Dense up-split ZagPine had worse next-character loss than identity Alpine but better exact regex template/expanded-regex generation. Weight inspection did not show smaller weights; the effect appears more likely to come from the activation geometry than from simple weight shrinkage.

### Regex And Four-Math Mixed Replay

The mixed replay corpus uses a `50/12.5/12.5/12.5/12.5` split:

- 50% v7 regex
- 12.5% addition
- 12.5% subtraction
- 12.5% multiplication
- 12.5% division

Dense up-split ZagPine was continued from its 6000-step v7 checkpoint. After 2000 mixed steps, regex exactness temporarily dropped to `80%`, but math exactness was already high. Continuing another 2000 mixed steps recovered regex and improved math.

Best checkpoint from the 8000-to-10000 continuation:

| Metric | Result |
|---|---:|
| Mixed eval loss | `0.1605` |
| Mixed next-char acc | `95.33%` |
| Loss-pred MSE | `0.0514` |
| Regex exact output | `45/50 = 90%` |
| Regex IL exact | `46/50 = 92%` |
| Regex template exact | `45/50 = 90%` |
| Regex expanded exact | `45/50 = 90%` |
| Addition answer exact | `499/500 = 99.8%` |
| Subtraction answer exact | `489/500 = 97.8%` |
| Multiplication answer exact | `499/500 = 99.8%` |
| Division answer exact | `500/500 = 100%` |

Subtraction remains the weakest exact-answer skill. The observed failures are mostly borrow chains and borrow-through-zero cases, which matches the earlier pattern that subtraction behaves more like comparison/control flow than addition or multiplication.

### Open Activation Variant

A plausible next ZagPine ablation is width-scaled ZigZag amplitude. The goal is to prevent wide basins from producing overly large additive zag terms while preserving sharper behavior for narrow basins. The gentlest proposed form is:

```text
effective_zag_amp = zag_amp / sqrt(width + eps)
```

This should be tested as a config-gated option against the existing dense up-split ZagPine on the math-only corpus before using it in mixed regex/math replay.
