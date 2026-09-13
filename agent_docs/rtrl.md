# RTRL: the math

This is the implementation spec for `src/rtrl_flight/rtrl/`. It is intentionally precise about shapes and recursions because `tests/test_rtrl_sensitivity.py` checks this exactly against finite differences — if the doc and the test ever disagree, the test wins and this doc is stale (fix the doc).

## Why naive RTRL is intractable, and why RTUs fix it

For a general RNN with hidden size N, RTRL maintains the sensitivity tensor `S_t = ∂h_t/∂θ` for every parameter θ. For a dense recurrent weight matrix, `h_t` depends on all N components of `h_{t-1}` through an N×N matrix, so `∂h_t/∂θ` is an N×N×|θ| object and updating it costs O(N^3) per step, O(N^4) total — abandoned for anything but toy networks decades ago.

**Recurrent Trace Units (RTUs)** use an elementwise (diagonal) linear recurrence instead of a dense one. Nonlinearity is pushed entirely into the readout, never into the recurrence itself. The consequence: `∂h_t[i]/∂h_{t-1}[j] = 0` for `i ≠ j`. The sensitivity of any parameter that only feeds unit `i` stays localized to unit `i` forever — it never needs to be tracked against the other N-1 units. Storage and per-step update cost drop from O(N^2) per parameter to O(1) per parameter, i.e. O(N·D) total for input weights and O(N) for recurrence gains — genuinely exact RTRL, not an approximation, at a cost comparable to a forward pass. This is the whole reason CLAUDE.md forbids "improving" this with Kronecker/sparsity tricks: those tricks exist to approximate RTRL for architectures where it's intractable. Ours already isn't.

## The RTU cell

Hidden state `h ∈ R^N`. Input `x_t ∈ R^D` (the flattened observation, or the previous layer's output if RTUs are stacked).

**Recurrence (linear, diagonal — this is the part that must stay diagonal):**

```
h_t[i] = a[i] * h_{t-1}[i] + z_t[i],   for each unit i = 1..N
z_t = W_in @ x_t + b_in                 (dense, but has no recurrent dependency)
```

`a ∈ R^N` is the per-unit recurrence gain, parametrized as `a[i] = tanh(a_raw[i])` so `|a[i]| < 1` always (stability: no exploding hidden state regardless of gradient steps on `a_raw`).

**Readout (nonlinearity lives here, not in the recurrence):**

```
y_t = W_out @ tanh(h_t) + b_out
```

Trainable parameters: `W_in (N×D)`, `b_in (N)`, `a_raw (N)`, `W_out (O×N)`, `b_out (O)`. Only `W_in`, `b_in`, `a_raw` participate in the recurrent sensitivity recursion below — `W_out`/`b_out` only ever need an instantaneous (non-recurrent) gradient, same as any feedforward layer.

## Forward sensitivity recursion

For each recurrent parameter `θ` in `{W_in[i,k], b_in[i], a_raw[i]}`, define `S_t^θ[i] = ∂h_t[i]/∂θ`. Because the recurrence is diagonal, `S_t^θ[j] = 0` for all `j ≠ i` where `i` is the one unit `θ` feeds — we only ever store one scalar per (unit, parameter) pair, not a full N-vector.

Recursion (`∂a[i]/∂a_raw[i] = 1 - a[i]^2` from the tanh parametrization):

```
S_t^{W_in[i,k]} = a[i] * S_{t-1}^{W_in[i,k]} + x_t[k]
S_t^{b_in[i]}   = a[i] * S_{t-1}^{b_in[i]}   + 1
S_t^{a_raw[i]}  = a[i] * S_{t-1}^{a_raw[i]}  + (1 - a[i]^2) * h_{t-1}[i]
```

Initialize all `S_0^θ = 0` at episode start (`Controller.reset()` clears these along with `h_0 = 0`).

## Turning sensitivities into a gradient (the online update)

At every step, once `y_t` is produced and a loss/error signal `L_t` is available (imitation: supervised error vs. expert action; online RTRL: whatever the deployed objective is — see `experiments.md`), compute the **instantaneous** credit assignment through the readout only (no recursion needed here, it's a single local backward pass):

```
dL_t/dh_t[i] = sum_o ( dL_t/dy_t[o] * W_out[o,i] * (1 - tanh(h_t[i])^2) )
```

Then combine with the stored forward sensitivities to get the true RTRL gradient for each recurrent parameter:

```
dL_t/dθ = sum_i ( dL_t/dh_t[i] * S_t^θ[i] )     # sum collapses to one term when θ feeds a single unit i
```

`W_out`/`b_out` gradients are the ordinary instantaneous backprop gradient (no `S` term — they're not part of the recurrence).

`Controller.update()` for RTRL-RTU does, per step: forward pass → compute `dL_t/dh_t` → combine with `S_t` for the recurrent-parameter gradients → step an optimizer (Adam state persists across steps, same as any online learner) → advance `S_t` to `S_{t+1}` using the recursion above with the *new* `h_t`. This is what makes it "real-time": no unroll, no replay buffer, one step of work per env step, causal (never looks at future timesteps).

### What `dL_t/dy_t[o]` actually is (# updated at final health-check — this doc left it abstract; here's the concrete, currently-implemented objective, in `src/rtrl_flight/controllers/rtrl_rtu.py`'s `update()`)

**Per-channel tracking loss, not a scalar broadcast to every output.** An earlier version set `dL_t/dy_t[o]` to the *same* value for all three channels (`-reward`, then a running-mean/std-normalized version of it) — mathematically valid as an instance of the general formula above, but it turned out to be the actual root cause of a real failure: broadcasting an identical value to every channel every step, for ~15000 online steps, is *systematically correlated in direction*, not just noisy, and drove `output_layer`'s weights to saturation regardless of how well-scaled that shared value was (clipping/decay/norm-capping could only ever bound the symptom, never the underlying directional bias). The fix, now the actual implementation:

```
pitch_error = next_obs[9]     # error/pitch-error-rad
roll_error  = next_obs[10]    # error/roll-error-rad
targets = [-roll_error, -pitch_error, 0.0]   # [aileron, elevator, rudder]

channel_loss = mean_o (y_t[o] - targets[o])^2
loss_scale   = 1 + max(0, -normalized_reward)     # normalized_reward: running-mean/std of reward
dL_t/dy_t[o] = (2 / 3) * (y_t[o] - targets[o]) * loss_scale
```

Aileron's target is `-roll_error` and elevator's is `-pitch_error` (reduce the corresponding error); rudder has no direct tracking target (its role is yaw-rate damping, not error-tracking) so its target is a constant `0.0`. The reward-normalized `loss_scale` is a *multiplicative weight* on this per-channel loss (upweight bad steps, downweight good ones) — it is not the loss itself, unlike the earlier broadcast version. This makes `dL_t/dy_t` a genuine per-channel vector (each entry depends on that channel's own logit and target), which is what actually stops the systematic drift — see `src/rtrl_flight/controllers/rtrl_rtu.py`'s `update()` docstring/comments for the full incident history.

Practical safeguards that remain in `update()` alongside the correct objective above (none of these substitute for it — they exist because online RTRL on a stiff flight-dynamics task has real failure modes even with a correct gradient): a hidden-state clamp (`RTUCell`'s recurrence is linear/unbounded by design, so raw-scale inputs like altitude can otherwise blow up `h`), a sensitivity-Frobenius-norm sanity check (reset `S` to zero if it exceeds a fixed limit), and a non-finite-reward guard (skip the update entirely if `reward` is `nan`/`inf` or `h` already is — `AttitudeHoldTask` has no safety-bound termination yet, see `environment.md`, so a mid-episode divergence can otherwise poison the optimizer state for the rest of training).

## Why warm-start is required (not optional)

Cold-start online RTRL on raw flight dynamics diverges — noted as known behavior in CLAUDE.md, not a bug to chase. Two reasons this is expected, not surprising:

1. RTRL's instantaneous gradient is a valid but high-variance estimator early in training, when `W_out` and `a` haven't settled — the credit assignment through `dL_t/dh_t` is noisy when the readout is near-random.
2. Attitude-hold dynamics are stiff enough that a randomly-initialized controller can put the aircraft into a state the online learner has never had to recover from, compounding the noisy-gradient problem with an out-of-distribution one.

Imitation warm-start (offline, supervised, can use full BPTT since it's offline) gets `W_in`, `a`, `W_out` into a regime where the online RTRL gradient is low-variance and the aircraft stays near the training distribution before online adaptation ever has to do real work. See `training/` and `experiments.md` for the warm-start protocol.

## The correctness test

`tests/test_rtrl_sensitivity.py`: build a small RTU cell (small N, small D, toy scalar loss), run the forward-sensitivity recursion above for T steps, compute `dL_T/dθ` via the formula above, and independently compute the same gradient via `torch.autograd` on the fully unrolled T-step graph (equivalent to truncated BPTT with truncation length T, which is exact for a length-T episode). These must agree to ~1e-5. This is checking that the *online, causal, no-unroll* computation is mathematically identical to the *offline, unrolled* one — that's the entire claim RTRL makes, and it's the one thing in this repo that must never regress silently.

Optionally, per-parameter finite-difference check (`torch.autograd.gradcheck`-style: perturb `θ` by `ε`, rerun the forward rollout, compare `(L(θ+ε) - L(θ-ε)) / 2ε` to the analytic `dL_T/dθ`) as a second, autograd-independent cross-check — this is the one referenced in `CLAUDE.md`.

## Deferred (do not add without discussion)

- Complex-valued / 2×2-rotation-block recurrence (à la LRU) for richer per-unit dynamics — would still be diagonal-block and RTRL-tractable, but changes the sensitivity recursion's shape and isn't needed for the attitude-hold task's dynamics. Out of scope for the 14-day window.
- Any sparsification or Kronecker-factored approximation to the sensitivity tensor — explicitly forbidden in CLAUDE.md; RTUs already make the exact computation cheap, so there is nothing to approximate.
