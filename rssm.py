import torch
from torch import distributions as torchd
from torch import nn

import distributions as dists
from gatelord import GateLord, GateLordBinary, TimeLord
from networks import BlockLinear, LambdaLayer
from tools import rpad, weight_init_


class Deter(nn.Module):
    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU", context_dim=0):
        super().__init__()
        self.blocks = int(blocks)
        self.dynlayers = int(dynlayers)
        self.context_dim = int(context_dim)
        act = getattr(torch.nn, act)
        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        num_inputs = 3
        if self.context_dim > 0:
            self._dyn_in3 = nn.Sequential(
                nn.Linear(context_dim, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
            )
            num_inputs = 4
        self._dyn_hid = nn.Sequential()
        in_ch = (num_inputs * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(f"dyn_hid_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(f"norm_{i}", nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act())
            in_ch = deter
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)
        self.flat2group = lambda x: x.reshape(*x.shape[:-1], self.blocks, -1)
        self.group2flat = lambda x: x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch, deter, action, context=None):
        """Deterministic state transition (block-GRU style)."""
        # (B, S, K), (B, D), (B, A), optional (B, C)
        B = action.shape[0]

        # Flatten stochastic state and normalize action magnitude.
        # (B, S*K)
        stoch = stoch.reshape(B, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        # (B, U)
        x0 = self._dyn_in0(deter)
        x1 = self._dyn_in1(stoch)
        x2 = self._dyn_in2(action)

        # Concatenate projected inputs and broadcast over blocks.
        if self.context_dim > 0 and context is not None:
            x3 = self._dyn_in3(context)
            x = torch.cat([x0, x1, x2, x3], -1)
        else:
            x = torch.cat([x0, x1, x2], -1)
        # (B, G, N*U)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)

        # Combine per-block deterministic state with per-block inputs.
        # (B, G, D/G + N*U) -> (B, D + N*U*G)
        x = self.group2flat(torch.cat([self.flat2group(deter), x], -1))

        # (B, D)
        x = self._dyn_hid(x)
        # (B, 3*D)
        x = self._dyn_gru(x)

        # Split GRU-style gates block-wise.
        # (B, G, 3*D/G)
        gates = torch.chunk(self.flat2group(x), 3, dim=-1)

        # (B, D)
        reset, cand, update = (self.group2flat(x) for x in gates)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        # (B, D)
        return update * cand + (1 - update) * deter


class RSSM(nn.Module):
    def __init__(self, config, embed_size, act_dim, has_context=False):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        act = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._initial = str(config.initial)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._dyn_layers = int(config.dyn_layers)
        self._blocks = int(config.blocks)
        self.flat_stoch = self._stoch * self._discrete

        # Context support (THICK/CRSSM)
        self._has_context = has_context
        if self._has_context:
            self._context_size = int(config.context)
            self._gate_noise_scale = float(config.gate_noise_scale)
            self._coarse_layers = int(config.coarse_layers)
            self._sparse_free = float(config.sparse_free)

            # Coarse projection: flat_stoch + action -> hidden
            self._coarse_proj = nn.Sequential(
                nn.Linear(self.flat_stoch + act_dim, self._hidden, bias=True),
                nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32),
                act(),
            )

            # GateLord cell
            gate_type = str(getattr(config, "gate_type", "gatelord"))
            gate_cls = {"gatelord": GateLord, "gatelord_binary": GateLordBinary, "timelord": TimeLord}[gate_type]
            self._gatelord = gate_cls(
                input_size=self._hidden,
                hidden_size=self._context_size,
                gate_noise_scale=self._gate_noise_scale,
            )

            # Coarse prior: context -> stoch logits
            self._coarse_net = nn.Sequential()
            inp_dim = self._context_size
            for i in range(self._coarse_layers):
                self._coarse_net.add_module(f"coarse_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
                self._coarse_net.add_module(f"coarse_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
                self._coarse_net.add_module(f"coarse_a_{i}", act())
                inp_dim = self._hidden
            self._coarse_net.add_module("coarse_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
            self._coarse_net.add_module(
                "coarse_lambda",
                LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
            )

            context_dim = self._context_size
            self.feat_size = self.flat_stoch + self._deter + self._context_size
            self.coarse_feat_size = self.flat_stoch + self._context_size
        else:
            context_dim = 0
            self.feat_size = self.flat_stoch + self._deter

        self._deter_net = Deter(
            self._deter,
            self.flat_stoch,
            act_dim,
            self._hidden,
            blocks=self._blocks,
            dynlayers=self._dyn_layers,
            act=config.act,
            context_dim=context_dim,
        )

        self._obs_net = nn.Sequential()
        inp_dim = self._deter + context_dim + embed_size
        for i in range(self._obs_layers):
            self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act())
            inp_dim = self._hidden
        self._obs_net.add_module("obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._obs_net.add_module(
            "obs_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )

        self._img_net = nn.Sequential()
        inp_dim = self._deter + context_dim
        for i in range(self._img_layers):
            self._img_net.add_module(f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(f"img_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_net_a_{i}", act())
            inp_dim = self._hidden
        self._img_net.add_module("img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete))
        self._img_net.add_module(
            "img_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )
        self.apply(weight_init_)

    def initial(self, batch_size):
        """Return an initial latent state (stoch, deter, context_or_None)."""
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        if self._has_context:
            context = torch.zeros(batch_size, self._context_size, dtype=torch.float32, device=self._device)
        else:
            context = None
        return stoch, deter, context

    def observe(self, embed, action, initial, reset):
        """Posterior rollout using observations.

        Returns:
            stochs:         (B, T, S, K)
            deters:         (B, T, D)
            contexts:       (B, T, C) or None
            logits:         (B, T, S, K)
            coarse_logits:  (B, T, S, K) or None
            gates:          (B, T, H_gate) or None
        """
        L = action.shape[1]
        stoch, deter, context = initial
        stochs, deters, logits = [], [], []
        contexts = [] if self._has_context else None
        coarse_logits = [] if self._has_context else None
        gates = [] if self._has_context else None
        for i in range(L):
            stoch, deter, context, logit, coarse_logit, gate = self.obs_step(
                stoch, deter, context, action[:, i], embed[:, i], reset[:, i]
            )
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
            if self._has_context:
                contexts.append(context)
                coarse_logits.append(coarse_logit)
                gates.append(gate)
        return (
            torch.stack(stochs, dim=1),
            torch.stack(deters, dim=1),
            torch.stack(contexts, dim=1) if contexts is not None else None,
            torch.stack(logits, dim=1),
            torch.stack(coarse_logits, dim=1) if coarse_logits is not None else None,
            torch.stack(gates, dim=1) if gates is not None else None,
        )

    def obs_step(self, stoch, deter, context, prev_action, embed, reset):
        """Single posterior step.

        Returns:
            stoch:          (B, S, K)
            deter:          (B, D)
            context:        (B, C) or None
            logit:          (B, S, K) — posterior logits
            coarse_logit:   (B, S, K) or None — coarse prior logits
            gate:           (B, H_gate) or None
        """
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )

        if self._has_context:
            context = torch.where(rpad(reset, context.dim() - int(reset.dim())), torch.zeros_like(context), context)

            # 1. Coarse context update via GateLord.
            flat_stoch = stoch.reshape(stoch.shape[0], -1)
            coarse_input = self._coarse_proj(torch.cat([flat_stoch, prev_action], dim=-1))
            coarse_out, context, gate = self._gatelord(coarse_input, context)

            # 2. Deterministic transition with context.
            deter = self._deter_net(stoch, deter, prev_action, context)

            # 3. Posterior: condition on deter + context + embed.
            x = torch.cat([deter, context, embed], dim=-1)
            logit = self._obs_net(x)
            stoch = self.get_dist(logit).rsample()

            # 4. Coarse prior logits from coarse output.
            coarse_logit = self._coarse_net(coarse_out)

            return stoch, deter, context, logit, coarse_logit, gate
        else:
            # Standard RSSM (no context).
            deter = self._deter_net(stoch, deter, prev_action)
            x = torch.cat([deter, embed], dim=-1)
            logit = self._obs_net(x)
            stoch = self.get_dist(logit).rsample()
            return stoch, deter, None, logit, None, None

    def img_step(self, stoch, deter, context, prev_action):
        """Single prior step (no observation).

        Returns:
            stoch:   (B, S, K)
            deter:   (B, D)
            context: (B, C) or None
            gate:    (B, H_gate) or None
        """
        if self._has_context:
            flat_stoch = stoch.reshape(stoch.shape[0], -1)
            coarse_input = self._coarse_proj(torch.cat([flat_stoch, prev_action], dim=-1))
            _, context, gate = self._gatelord(coarse_input, context)
            deter = self._deter_net(stoch, deter, prev_action, context)
            stoch, _ = self.prior(deter, context)
            return stoch, deter, context, gate
        else:
            deter = self._deter_net(stoch, deter, prev_action)
            stoch, _ = self.prior(deter)
            return stoch, deter, None, None

    def prior(self, deter, context=None):
        """Compute prior distribution parameters and sample stoch."""
        if context is not None:
            inp = torch.cat([deter, context], dim=-1)
        else:
            inp = deter
        logit = self._img_net(inp)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def imagine_with_action(self, stoch, deter, context, actions):
        """Roll out prior dynamics given a sequence of actions.

        Returns:
            stochs:   (B, T, S, K)
            deters:   (B, T, D)
            contexts: (B, T, C) or None
            gates:    (B, T, H_gate) or None
        """
        L = actions.shape[1]
        stochs, deters = [], []
        contexts = [] if self._has_context else None
        gates = [] if self._has_context else None
        for i in range(L):
            stoch, deter, context, gate = self.img_step(stoch, deter, context, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
            if self._has_context:
                contexts.append(context)
                gates.append(gate)
        return (
            torch.stack(stochs, dim=1),
            torch.stack(deters, dim=1),
            torch.stack(contexts, dim=1) if contexts is not None else None,
            torch.stack(gates, dim=1) if gates is not None else None,
        )

    def get_feat(self, stoch, deter, context=None):
        """Flatten stoch and concatenate with deter (and context if present)."""
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        if context is not None:
            return torch.cat([stoch, deter, context], -1)
        return torch.cat([stoch, deter], -1)

    def get_coarse_feat(self, stoch, context):
        """Flatten stoch and concatenate with context (no deter). Only valid when _has_context."""
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        return torch.cat([stoch, context], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        return dyn_loss, rep_loss

    def coarse_kl_loss(self, post_logit, coarse_logit, free):
        """KL losses between posterior and coarse prior.

        Returns:
            coarse_dyn: KL(sg(post) || coarse) — trains coarse prior.
            coarse_rep: KL(post || sg(coarse)) — trains posterior.
        """
        kld = dists.kl
        coarse_rep = kld(post_logit, coarse_logit.detach()).sum(-1)
        coarse_dyn = kld(post_logit.detach(), coarse_logit).sum(-1)
        coarse_rep = torch.clip(coarse_rep, min=free)
        coarse_dyn = torch.clip(coarse_dyn, min=free)
        return coarse_dyn, coarse_rep
