# Loss-Query Hymba Character LM

This repository is a cleaned, source-only snapshot of the `loss_query_hymba` experiments from `CompoundStreamingTransformer`.

The focus is a fast Hymba character language model with an auxiliary loss-query side channel:

- `models/CST/lm.py`: `LossQueryFastHymbaCharLM`
- `scripts/train/train_fast_hymba_char_lm.py`: training loop with plateau recovery and checkpoint vocab expansion
- `scripts/data/`: regex, addition, subtraction, and sentinel multitask corpus generators
- `scripts/eval/`: regex and math evaluators plus sample report writer

`Alpine` refers to the `previous_loss_scalar_injected_hymba` architecture: a branched Hymba block predicts scalar next-token loss from the penultimate-layer representation, injects that predicted-loss signal back into the token stream, and reruns the branch-residual path so next-character prediction can condition on autoregressive loss-prediction context.

`ZagPine` refers to Alpine with the corrected dynamic ZigZag activation in the loss-predict branch. The activation was originally called Dynamic BasinZag during exploration; in the implementation it replaces the loss-branch MLP `GELU` rather than adding an extra post-branch transform.

Generated corpora, experiment logs, and checkpoints are intentionally excluded from Git. The `.gitignore` blocks `data/`, `experiments/`, `*.pt`, and other large artifacts.

See `docs/EXPERIMENT_SUMMARY.md` for the current local findings, including the v7 dream-regex curriculum, Alpine versus FastHymba comparisons, mixed regex/math replay results, and training-time overhead.

## Reggie CLI

`reggie` is a grep-style wrapper around the Alpine regex model. It accepts a file or folder plus a plain-English regex request, prepends `/r` when needed, expands quoted refs from the model template, and highlights matches in the terminal.

```powershell
python reggie.py SampleOutputs\reggie_smoke.txt --show-regex
python reggie.py . --max-matches 20
python reggie.py app.log --context 40
```

Omit the query to use interactive mode, where quotes can be typed naturally. Run `python reggie.py --help` for more examples, including replacement previews.

Interactive mode asks whether each result was correct. Answer `n` to save the query, generated IL/template/regex, and sampled matched lines to `SampleOutputs/reggie_failures.jsonl` for later curriculum fixes.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA/CPU environment if the generic requirement is not correct for your machine.

## Generate The Current Sentinel Curriculum

```powershell
python scripts\data\prepare_regex_il_v5_traces.py --output data\regex_il_v5_clear_capture_120k.txt --metadata-output data\regex_il_v5_clear_capture_120k.sources.json --examples 120000 --seed 1111
python scripts\data\prepare_addition_traces.py --output data\addition_traces_1to3_digit.txt --metadata-output data\addition_traces_1to3_digit.sources.json --examples 50000 --seed 1111
python scripts\data\prepare_subtraction_traces.py --output data\subtraction_traces_1to3_digit.txt --metadata-output data\subtraction_traces_1to3_digit.sources.json --examples 50000 --seed 1111

python scripts\data\prepare_tagged_regex_addition_mix.py `
  --output data\sentinel_regex_v5_addition_subtraction_70_15_15_100k.txt `
  --metadata-output data\sentinel_regex_v5_addition_subtraction_70_15_15_100k.sources.json `
  --v5-examples 70000 `
  --v2-examples 0 `
  --addition-examples 15000 `
  --subtraction-examples 15000 `
  --sentinel-format `
  --sentinel "<END>" `
  --seed 20260513
```

## Train Loss-Query Hymba

```powershell
python scripts\train\train_fast_hymba_char_lm.py `
  --data-path data\sentinel_regex_v5_addition_subtraction_70_15_15_100k.txt `
  --architecture loss_query_hymba `
  --steps 2000 `
  --batch-size 8 `
  --seq-len 384 `
  --d-model 128 `
  --num-heads 4 `
  --layers 16 `
  --lr 0.0005 `
  --eval-every 100 `
  --checkpoint-every 500 `
  --sample-every 0 `
  --loss-prediction-alpha 0.05 `
  --run-name loss_query_hymba_sentinel_v5_add_sub
```

Continuation runs can use `--resume-checkpoint` and `--vocab-checkpoint`. If the new corpus has additional characters, use `--allow-vocab-expansion`.

## Evaluate

```powershell
python scripts\eval\evaluate_regex_il_v5_trace_lm.py `
  --checkpoint experiments\mods\001_first_pass_hymba_cst\runs\loss_query_hymba_sentinel_v5_add_sub\checkpoint_final.pt `
  --examples 100 `
  --seq-len 384 `
  --max-new-chars 340 `
  --task-prefix "Task: regex_v5" `
  --shared-output-format `
  --stop-token "<END>"

python scripts\eval\evaluate_addition_trace_forced.py `
  --checkpoint experiments\mods\001_first_pass_hymba_cst\runs\loss_query_hymba_sentinel_v5_add_sub\checkpoint_final.pt `
  --examples 100 `
  --task-prefix "Task: addition_prose" `
  --shared-output-format `
  --sentinel "<END>"

python scripts\eval\evaluate_subtraction_trace_forced.py `
  --checkpoint experiments\mods\001_first_pass_hymba_cst\runs\loss_query_hymba_sentinel_v5_add_sub\checkpoint_final.pt `
  --examples 100 `
  --task-prefix "Task: subtraction_prose" `
  --shared-output-format `
  --sentinel "<END>"
```

## Best Local Result Before Cleanup

The source workspace best checkpoint was not copied here. Its run name was:

`2026-05-14_loss_query_hymba_sentinel_v5_add_sub_70_15_15_4000_continue_lr0005`

Held-out evals from that run:

- v5 sentinel regex: `98/100`
- addition forced answer exact: `95/100`
- subtraction forced answer exact: `100/100`
- old v2 quoted-ref: `57/100`
