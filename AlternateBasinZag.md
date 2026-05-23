Width-Modulated Flat-Wing Activation (“Skateboard Ramp”)

Goal:
Create an activation with:

* linear / information-rich center
* smooth abrupt shoulders (“crick”)
* stable flat outer wings, such that the output is always between -1 and 1. -1 and 1 should be both flat, 
* preserved polarity
* independently controllable center width and wing amplitude

Core equation:

z = x / (width + eps)

y = amp * z / ((1 + |z|^q)^(1/q))

Equivalent envelope form:

y = x * f(x, width)

where:

f(x,width) =
1 / ((1 + |x/width|^q)^(1/q))

Properties:

* width controls only the center width
* amp controls only the flat wing saturation level
* q controls shoulder sharpness / skateboard-ramp abruptness
* polarity is preserved automatically because the envelope is nonnegative
* output smoothly approaches ±amp in the wings
* fully continuous and differentiable
* behaves like a smooth bounded linear unit with flat outer attractor regions

PyTorch reference:

def flat_wing(x, width, amp=1.0, q=8.0, eps=1e-6):
z = x / (width + eps)
return amp * z / ((1.0 + z.abs().pow(q)).pow(1.0 / q))

Notes:

* higher q -> sharper shoulders and flatter wings
* lower q -> softer tanh/bell-like transition
* width modulation changes center operating range without changing wing height
* interesting candidate for expectation-shaped throughput activations
