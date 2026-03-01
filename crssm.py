import torch
from torch import distributions as torchd
from torch import nn

import distributions as dists
from gatelord import GateLord, GateLordBinary, TimeLord
from networks import LambdaLayer
from rssm import Deter, RSSM
from tools import rpad, weight_init_


class CRSSM(RSSM):
    """RSSM extended with GateLord coarse context (THICK).

    Maintains a slow-changing ``context`` vector alongside the standard
    ``deter`` and ``stoch`` states.  The context is updated by a GateLord
    cell and integrated into the deterministic transition, posterior,
    and prior computations.
    """

    def __init__(self, config, embed_size, act_dim):
        # Temporarily prevent RSSM.__init__ from building nets we'll override.
        # We call super().__init__ to get all the standard bookkeeping, then
        # rebuild the parts that need context.
        super().__init__(config, embed_size, act_dim)

        self._context_size = int(config.context)
        self._gate_noise_scale = float(config.gate_noise_scale)
        self._coarse_layers = int(config.coarse_layers)
        self._sparse_free = float(config.sparse_free)
        act = getattr(torch.nn, config.act)

        # --- Coarse projection: flat_stoch + action -> hidden ---
        self._coarse_proj = nn.Sequential(
            nn.Linear(self.flat_stoch + act_dim, self._hidden, bias=True),
            nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32),
            act(),
        )

        # --- GateLord cell ---
        gate_type = str(getattr(config, "gate_type", "gatelord"))
        gate_cls = {"gatelord": GateLord, "gatelord_binary": GateLordBinary, "timelord": TimeLord}[gate_type]
        self._gatelord = gate_cls(
            input_size=self._hidden,
            hidden_size=self._context_size,
            gate_noise_scale=self._gate_noise_scale,
        )

        # --- Coarse prior: context -> stoch logits ---
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

        # --- Override Deter to accept context ---
        self._deter_net = Deter(
            self._deter,
            self.flat_stoch,
            act_dim,
            self._hidden,
            blocks=self._blocks,
            dynlayers=self._dyn_layers,
            act=config.act,
            context_dim=self._context_size,
        )

        # --- Override obs_net to include context ---
        self._obs_net = nn.Sequential()
        inp_dim = self._deter + self._context_size + embed_size
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

        # --- Override img_net to include context ---
        self._img_net = nn.Sequential()
        inp_dim = self._deter + self._context_size
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

        # --- Feature sizes ---
        self.feat_size = self.flat_stoch + self._deter + self._context_size
        self.coarse_feat_size = self.flat_stoch + self._context_size

        self.apply(weight_init_)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def initial(self, batch_size):
        """Return an initial latent state (stoch, deter, context)."""
        stoch, deter = super().initial(batch_size)
        context = torch.zeros(batch_size, self._context_size, dtype=torch.float32, device=self._device)
        return stoch, deter, context

    # ------------------------------------------------------------------
    # Posterior (observation-conditioned) steps
    # ------------------------------------------------------------------

    def observe(self, embed, action, initial, reset):
        """Posterior rollout using observations.

        Returns:
            stochs:        (B, T, S, K)
            deters:         (B, T, D)
            contexts:       (B, T, C)
            logits:         (B, T, S, K)
            coarse_logits:  (B, T, S, K)
            gates:          (B, T, H_gate)
        """
        L = action.shape[1]
        stoch, deter, context = initial
        stochs, deters, contexts, logits, coarse_logits, gates = [], [], [], [], [], []
        for i in range(L):
            stoch, deter, context, logit, coarse_logit, gate = self.obs_step(
                stoch, deter, context, action[:, i], embed[:, i], reset[:, i]
            )
            stochs.append(stoch)
            deters.append(deter)
            contexts.append(context)
            logits.append(logit)
            coarse_logits.append(coarse_logit)
            gates.append(gate)
        return (
            torch.stack(stochs, dim=1),
            torch.stack(deters, dim=1),
            torch.stack(contexts, dim=1),
            torch.stack(logits, dim=1),
            torch.stack(coarse_logits, dim=1),
            torch.stack(gates, dim=1),
        )

    def obs_step(self, stoch, deter, context, prev_action, embed, reset):
        """Single posterior step with coarse context.

        Returns:
            stoch:         (B, S, K)
            deter:          (B, D)
            context:        (B, C)
            logit:          (B, S, K) — posterior logits
            coarse_logit:   (B, S, K) — coarse prior logits
            gate:           (B, H_gate)
        """
        # Reset states on episode boundary.
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        context = torch.where(rpad(reset, context.dim() - int(reset.dim())), torch.zeros_like(context), context)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )

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

    # ------------------------------------------------------------------
    # Prior (imagination) steps
    # ------------------------------------------------------------------

    def img_step(self, stoch, deter, context, prev_action):
        """Single prior step (no observation) with coarse context.

        Returns:
            stoch:   (B, S, K)
            deter:    (B, D)
            context:  (B, C)
            gate:     (B, H_gate)
        """
        # 1. Coarse context update.
        flat_stoch = stoch.reshape(stoch.shape[0], -1)
        coarse_input = self._coarse_proj(torch.cat([flat_stoch, prev_action], dim=-1))
        _, context, gate = self._gatelord(coarse_input, context)

        # 2. Deterministic transition with context.
        deter = self._deter_net(stoch, deter, prev_action, context)

        # 3. Prior sampling.
        stoch, _ = self.prior(deter, context)

        return stoch, deter, context, gate

    def prior(self, deter, context=None):
        """Compute prior distribution parameters and sample stoch.

        When context is provided (CRSSM mode), it is concatenated with
        deter before the prior MLP.
        """
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
            deters:    (B, T, D)
            contexts:  (B, T, C)
            gates:     (B, T, H_gate)
        """
        L = actions.shape[1]
        stochs, deters, contexts, gates = [], [], [], []
        for i in range(L):
            stoch, deter, context, gate = self.img_step(stoch, deter, context, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
            contexts.append(context)
            gates.append(gate)
        return (
            torch.stack(stochs, dim=1),
            torch.stack(deters, dim=1),
            torch.stack(contexts, dim=1),
            torch.stack(gates, dim=1),
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_feat(self, stoch, deter, context):
        """Flatten stoch and concatenate with deter and context."""
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        return torch.cat([stoch, deter, context], -1)

    def get_coarse_feat(self, stoch, context):
        """Flatten stoch and concatenate with context (no deter)."""
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        return torch.cat([stoch, context], -1)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

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
