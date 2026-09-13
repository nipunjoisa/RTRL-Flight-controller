"""Approximate warm-start: map a trained BPTTLSTMController checkpoint onto
a fresh RTRLRTUController. An exact translation between an LSTM and an RTU
is not possible -- they are different recurrent architectures (LSTM has
multiplicative gates and a separate cell/hidden state; RTU has a purely
linear, diagonal recurrence) -- so this produces a *reasonable starting
point* for online RTRL to refine from, not a lossless conversion. See
agent_docs/rtrl.md, "Why warm-start is required": cold-start online RTRL on
raw flight dynamics diverges, so RTRL-RTU needs something better than a
from-scratch random init even though nothing here is mathematically exact.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

# Per-unit recurrence gain RTU is initialized to (a = tanh(a_raw) = 0.9).
# Deliberately NOT extracted from the LSTM's forget gate (weight_hh_l0):
# the LSTM's forget gate is an input-dependent, nonlinear (sigmoid of a
# learned affine function of h_{t-1} and x_t) per-step quantity, while RTU's
# `a` is a single learned constant per unit -- there is no principled way to
# collapse the former into the latter. 0.9 is a conservative, stable
# "moderately slow forgetting" default (see RTUCell: |a| < 1 always via the
# tanh parametrization, so 0.9 leaves headroom for online RTRL to adjust it
# in either direction without immediately saturating).
DEFAULT_LAMBDA = 0.9


def load_bptt_checkpoint(path: str | Path) -> dict[str, Any]:
    """Loads a checkpoint saved by BPTTLSTMController.save(). Mandatory --
    per agent_docs/rtrl.md, RTRL-RTU must never cold-start on raw flight
    dynamics, so a missing checkpoint is a hard failure, not a silent
    fallback to random init.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"BPTT-LSTM checkpoint not found at {path!s} -- RTRL-RTU warm-start is "
            "mandatory (agent_docs/rtrl.md: cold-start online RTRL on raw flight "
            "dynamics diverges). Run scripts/train_bptt.py first to produce one."
        )
    return torch.load(path, weights_only=True)


def apply_warmstart(rtu_controller: Any, bptt_state_dict: dict[str, Any]) -> None:
    """Mutates rtu_controller's parameters in place. Mapping decisions:

    - W_in: the LSTM's `weight_ih_l0` is (4*lstm_hidden, input_size), with
      rows stacked as [input_gate; forget_gate; cell_gate; output_gate]
      (PyTorch's LSTM gate order). We take the *input gate*'s block (the
      first lstm_hidden rows) as a rough proxy for "how the raw input maps
      into hidden units" -- an arbitrary choice among the four gates, but
      the input gate is the one closest in spirit to RTU's un-gated
      z_t = W_in @ x_t + b_in (it directly gates how much of the new input
      enters the state, rather than gating existing state or output).
      Copied only over the (min(rtu_hidden, lstm_hidden), input_size)
      overlap -- if input_size itself doesn't match, nothing sane can be
      copied and W_in/b_in are left at RTUCell's own random init.
    - b_in: same input-gate block, from `bias_ih_l0` -- summed with the
      corresponding `bias_hh_l0` slice, since PyTorch's LSTM keeps input-side
      and hidden-side biases separate per gate while RTU has one combined
      bias per unit.
    - a_raw: NOT extracted from `weight_hh_l0` -- always set to
      atanh(DEFAULT_LAMBDA) (see that constant's docstring above).
    - W_out/b_out: copied exactly from BPTT's output_layer only if the
      shapes match exactly (both a plain Linear(hidden_size, ACTION_DIM),
      same semantics) -- this is the one mapping that CAN be exact, but only
      when hidden_size matches between the two controllers. Left at RTU's
      own random init otherwise; a mismatched Linear(hidden_a, 3) has no
      defensible partial-copy into Linear(hidden_b, 3).
    """
    cell = rtu_controller.cell
    lstm_state = bptt_state_dict["lstm_state_dict"]
    lstm_hidden = bptt_state_dict["hidden_size"]

    weight_ih_l0 = lstm_state["weight_ih_l0"]  # (4*lstm_hidden, input_size)
    bias_ih_l0 = lstm_state["bias_ih_l0"]
    bias_hh_l0 = lstm_state["bias_hh_l0"]

    n = min(cell.hidden_size, lstm_hidden)
    input_gate_W = weight_ih_l0[:n]  # (n, input_size) -- first lstm_hidden rows are the input gate
    input_gate_b = bias_ih_l0[:n] + bias_hh_l0[:n]

    with torch.no_grad():
        if input_gate_W.shape[1] == cell.W_in.shape[1]:
            cell.W_in[:n, :].copy_(input_gate_W)
            cell.b_in[:n].copy_(input_gate_b)
        # else: input_size mismatch -- nothing sane to copy, leave random init.

        cell.a_raw.fill_(math.atanh(DEFAULT_LAMBDA))

        output_state = bptt_state_dict["output_layer_state_dict"]
        if output_state["weight"].shape == rtu_controller.output_layer.weight.shape:
            rtu_controller.output_layer.weight.copy_(output_state["weight"])
            rtu_controller.output_layer.bias.copy_(output_state["bias"])
        # else: hidden_size mismatch -- leave RTU's own random-init readout.
