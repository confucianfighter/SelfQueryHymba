We should first try ablating the floor / wings and see if basinzag works better when the outskirts go to zero quick.

My guess is that this is more stable.

Earlier, we ablated the zag component of the basinzag function, and performance collapsed. So another thing we can try is zag only with learned width factor:

gate = torch.tanh(width_factor * torch.abs(value))
y = value * gate

With this version, on the down projection we would want to detach the width modulation signals and concatenate with activaiton output values before the downprojection.

While we definitely tried oblating zag, I'm not sure what we had left in there, We also should try a clean envelope only version: 

r = value / (width + eps)

gate = floor + (1.0 - floor) / (1.0 + torch.abs(r).pow(2.0 * sharpness))

value_mod = value * gate

out = torch.cat([value_mod, gate.detach()], dim=-1)

We would want a nice range on max width: min_width = 0.25
max_width = 6.0
initial_fraction = 0.25 to 0.35