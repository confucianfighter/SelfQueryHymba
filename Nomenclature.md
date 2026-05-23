Zag:

A class of activation functions in which an envelope is modulated by a channel, and value is passed through that. The actual envelope and configuration can differ

SZag (swish zag), RZag(ReluZag), GZag(GeluZag): A variant of zag in which the raw w channel is also passed through a a more traditional activation function.

Expectron: Our target design in which a single activation can learn to pass w as an expectation.

MLXP: Multilayer Expectron

Braiding: taking a single large projection and breaking it down into two, and then mixing the outputs by some kind of interleave accomplished through tensor views or reshapes. Below is an example of how we might accomplish that

value, width = up.chunk(2, dim=-1)
activated = zag(value, width)
paired = torch.stack([activated, width.detach()], dim=-1).flatten(-2)
down = down_proj(paired)

Alpine: stands for "Autoregressive Loss Predict InjEction." It's a pattern we have used successfully on every benchmark in which a block branches out from the output of Layer n-1, it projects to a scalar loss predict, and its normal output is added to the residual at then input of L of n. That scalar loss predict is then mixed into the next input. With this technique, all inputs need to be processed sequentially during inference. But during training, the first pass on a sample gathers loss predictions, and on the second pass those loss predictions are fed in with the inputs. 

DCR "Detached Convective Recursion":
A potential technique wherein each layer output at t is cached, then then mixed in with the input to the previous layer at t+1. Training for this should be accomplished in the same way as alpine, and layer outputs should be gathered on the same first pass as alpine. What makes this unique is that we are detaching when we send backward so that we don't incur exponential training costs. The hope is that states are sent backward like a conveyor belt, helping previous layers to disambiguate and refine their representations.

ZagPine: a series of models that implement Zag Activation (have an MLXP) and Alpine loss predict regression.

BraidedZagPine: a potential variant of ZagPine in which braiding is used to reduce parameters, mainly in the MLXP, but potentially in other places.









