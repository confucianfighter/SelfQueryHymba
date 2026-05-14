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

