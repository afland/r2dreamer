import torch
from torch import nn

from tools import weight_init_


def straight_through_heaviside(x):
    """Binary step with straight-through gradient estimator."""
    return x + (torch.where(x > 0, 1.0, 0.0) - x).detach()


def retanh(x):
    """Rectified tanh: max(0, tanh(x))."""
    return torch.clamp(torch.tanh(x), min=0)


class GateLord(nn.Module):
    """Soft per-unit gating via rectified tanh.

    A single continuous gate per hidden unit simultaneously controls
    whether and how much to update.  Sparsity comes from retanh
    pushing most gate values to zero.
    """

    def __init__(self, input_size, hidden_size, gate_noise_scale=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_noise_scale = gate_noise_scale
        self._gu = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self.apply(weight_init_)

    def forward(self, x, hidden):
        """
        Args:
            x: (B, input_size), hidden: (B, H)
        Returns:
            h_new: (B, H), h_new: (B, H), gate: (B, H)
        """
        gu = self._gu(torch.cat([x, hidden], dim=-1))
        gate_pre, update = gu.chunk(2, dim=-1)
        update = torch.tanh(update)
        if self.training:
            gate_pre = gate_pre + torch.randn_like(gate_pre) * self.gate_noise_scale
        gate = retanh(gate_pre)
        h_new = hidden + gate * (update - hidden)
        return h_new, h_new, gate


class GateLordBinary(nn.Module):
    """Per-unit binary boundary + learned scale.

    Decouples the update decision from its magnitude: a hard binary
    boundary (STE heaviside) per hidden unit decides *whether* to
    update, and a separate sigmoid scale controls *how much*.  Three
    independent projections give each component its own parameters.
    """

    def __init__(self, input_size, hidden_size, gate_noise_scale=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_noise_scale = gate_noise_scale
        in_dim = input_size + hidden_size
        self._event = nn.Linear(in_dim, hidden_size)
        self._scale = nn.Linear(in_dim, hidden_size)
        self._update = nn.Linear(in_dim, hidden_size)
        self.apply(weight_init_)

    def forward(self, x, hidden):
        """
        Args:
            x: (B, input_size), hidden: (B, H)
        Returns:
            h_new: (B, H), h_new: (B, H), boundary: (B, H)
        """
        z = torch.cat([x, hidden], dim=-1)
        event_pre = self._event(z)
        scale_pre = self._scale(z)
        update = torch.tanh(self._update(z))
        if self.training:
            event_pre = event_pre + torch.randn_like(event_pre) * self.gate_noise_scale
        boundary = straight_through_heaviside(event_pre)
        scale = torch.sigmoid(scale_pre)
        gate = boundary * scale
        h_new = hidden + gate * (update - hidden)
        return h_new, h_new, boundary


class TimeLord(nn.Module):
    """Scalar binary boundary + per-unit learned scale.

    Like GateLordBinary but the boundary is a single scalar shared
    across all hidden units — a clean "did a context-change event
    happen?" signal.  Per-unit scale and update still allow rich
    updates when the boundary fires.
    """

    def __init__(self, input_size, hidden_size, gate_noise_scale=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_noise_scale = gate_noise_scale
        in_dim = input_size + hidden_size
        self._event = nn.Linear(in_dim, 1)
        self._scale = nn.Linear(in_dim, hidden_size)
        self._update = nn.Linear(in_dim, hidden_size)
        self.apply(weight_init_)

    def forward(self, x, hidden):
        """
        Args:
            x: (B, input_size), hidden: (B, H)
        Returns:
            h_new: (B, H), h_new: (B, H), boundary: (B, 1)
        """
        z = torch.cat([x, hidden], dim=-1)
        event_pre = self._event(z)
        scale_pre = self._scale(z)
        update = torch.tanh(self._update(z))
        if self.training:
            event_pre = event_pre + torch.randn_like(event_pre) * self.gate_noise_scale
        boundary = straight_through_heaviside(event_pre)
        scale = torch.sigmoid(scale_pre)
        gate = boundary * scale
        h_new = hidden + gate * (update - hidden)
        return h_new, h_new, boundary


def sparse_loss(gates, free_nats=0.0):
    """Sparsity loss over gate activations (sparse_over_time_only).

    Takes the max over the hidden dimension at each timestep (binary
    indicator of *any* gate opening), sums over time, then normalises.

    Args:
        gates: (B, T, H) — gate activations from GateLord.
        free_nats: minimum allowed sparsity (free bits).

    Returns:
        Scalar loss per batch element, averaged over the batch.
    """
    # (B, T)
    gate_max = gates.max(dim=-1).values
    # Straight-through binarisation: 1 if any gate > 0, else 0.
    gate_binary = straight_through_heaviside(gate_max)
    # (B,) — count of timesteps with any gate open.
    sparse = gate_binary.sum(dim=1)
    if free_nats > 0:
        sparse = torch.clamp(sparse - free_nats, min=0)
    # Normalise by sequence length.
    return sparse.mean() / gates.shape[1]
