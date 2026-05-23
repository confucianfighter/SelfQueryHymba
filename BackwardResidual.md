                     BACKWARD RESIDUAL CONVEYOR BELT
                  "Detached Chained Convective Recurrence"

Idea, run two passes per sample, on the first pass, cache outputs of each layer. On the second pass inject those t-1 outputs input the previous layer along with t.

Since our Alpine -- Autoregressive Loss Predict injEction already requires two passes, we shouldn't be incurring very much extra as far as compute.

┌─────────────────────────────────────────────────────────────────────┐
│                         NORMAL FORWARD PASS                        │
└─────────────────────────────────────────────────────────────────────┘

Token Stream
──────────────────────────────────────────────────────────────────────► t


                ┌──────────┐
h0 ───────────► │ Layer 1  │ ──────────► h1
                └──────────┘
                       │
                       │ cache detached signal
                       ▼
                 stopgrad(h1)


                ┌──────────┐
h1 ───────────► │ Layer 2  │ ──────────► h2
                └──────────┘
                       │
                       │ cache detached signal
                       ▼
                 stopgrad(h2)


                ┌──────────┐
h2 ───────────► │ Layer 3  │ ──────────► h3
                └──────────┘
                       │
                       │ cache detached signal
                       ▼
                 stopgrad(h3)



═══════════════════════════════════════════════════════════════════════
            DETACHED BACKWARD CONVEYOR (cheap reusable signal)
═══════════════════════════════════════════════════════════════════════


            h3.detach()
                 │
                 ▼
          small projection
                 │
                 ▼
        "semantic pressure"
                 │
                 ▼
            injected into
              lower layer


                ┌──────────┐
h1 ───────────► │ Layer 2  │◄──── proj(h3.detach())
                └──────────┘
                        

                ┌──────────┐
h0 ───────────► │ Layer 1  │◄──── proj(h2.detach())
                └──────────┘



┌─────────────────────────────────────────────────────────────────────┐
│ WHY THIS MAY BE "ALMOST FREE"                                      │
└─────────────────────────────────────────────────────────────────────┘

Normal transformer cost:
    huge attention matrices
    full gradient graph
    quadratic token interactions

Conveyor cost:
    one cached tensor
    one projection
    one add/gate

So instead of:

    recomputing semantic structure repeatedly

you get:

    reusable detached high-level hints


┌─────────────────────────────────────────────────────────────────────┐
│ KEY INSIGHT                                                        │
└─────────────────────────────────────────────────────────────────────┘

detach() converts:

    COMPUTATION GRAPH
            into
    REUSABLE DATA

So the conveyor can be:
    cached
    compressed
    EMA averaged
    reused across passes
    reused across segments

WITHOUT recursive backprop explosion.


┌─────────────────────────────────────────────────────────────────────┐
│ RELATION TO TRANSFORMERXL                                          │
└─────────────────────────────────────────────────────────────────────┘

TransformerXL:
    past time ─────────────► future time

Your idea:
    deeper layers ─────────► shallower layers


TXL memory update:
```python
mem[i] = torch.cat((mem[i], x.detach()), dim=1)