import copy
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import networks
import rssm
import tools
from gatelord import sparse_loss
from networks import Projector
from optim import LaProp, clip_grad_agc_
from tools import to_f32


class Dreamer(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        self.rep_loss = str(config.rep_loss)

        # THICK config
        self._thick_enabled = bool(config.thick.enabled)
        if self._thick_enabled:
            self._sparse_free = float(config.rssm.sparse_free)
            self._use_coarse_critic = bool(config.thick.coarse_critic)
            self._psi = float(config.thick.psi)

        # World model components
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        self.rssm = rssm.RSSM(config.rssm, self.embed_size, self.act_dim, has_context=self._thick_enabled)
        self.reward = networks.MLPHead(config.reward, self.rssm.feat_size)
        self.cont = networks.MLPHead(config.cont, self.rssm.feat_size)

        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
            self.act_discrete = True
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
            self.act_discrete = True
        else:
            config.actor.dist = config.actor.dist.cont

        # Actor-critic components
        self.actor = networks.MLPHead(config.actor, self.rssm.feat_size)
        self.value = networks.MLPHead(config.critic, self.rssm.feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        self._loss_scales = dict(config.loss_scales)
        self._log_grads = bool(config.log_grads)

        modules = {
            "rssm": self.rssm,
            "actor": self.actor,
            "value": self.value,
            "reward": self.reward,
            "cont": self.cont,
            "encoder": self.encoder,
        }

        if self.rep_loss == "dreamer":
            self.decoder = networks.MultiDecoder(
                config.decoder,
                self.rssm._deter,
                self.rssm.flat_stoch,
                shapes,
            )
            recon = self._loss_scales.pop("recon")
            self._loss_scales.update({k: recon for k in self.decoder.all_keys})
            modules.update({"decoder": self.decoder})
        elif self.rep_loss == "r2dreamer" or self.rep_loss == "infonce":
            self.prj = Projector(self.rssm.feat_size, self.embed_size)
            modules.update({"projector": self.prj})
            self.barlow_lambd = float(config.r2dreamer.lambd)
        elif self.rep_loss == "dreamerpro":
            dpc = config.dreamer_pro
            self.warm_up = int(dpc.warm_up)
            self.num_prototypes = int(dpc.num_prototypes)
            self.proto_dim = int(dpc.proto_dim)
            self.temperature = float(dpc.temperature)
            self.sinkhorn_eps = float(dpc.sinkhorn_eps)
            self.sinkhorn_iters = int(dpc.sinkhorn_iters)
            self.ema_update_every = int(dpc.ema_update_every)
            self.ema_update_fraction = float(dpc.ema_update_fraction)
            self.freeze_prototypes_iters = int(dpc.freeze_prototypes_iters)
            self.aug_max_delta = float(dpc.aug.max_delta)
            self.aug_same_across_time = bool(dpc.aug.same_across_time)
            self.aug_bilinear = bool(dpc.aug.bilinear)

            self._prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim))
            self.obs_proj = nn.Linear(self.embed_size, self.proto_dim)
            self.feat_proj = nn.Linear(self.rssm.feat_size, self.proto_dim)
            self._ema_encoder = copy.deepcopy(self.encoder)
            self._ema_obs_proj = copy.deepcopy(self.obs_proj)
            for param in self._ema_encoder.parameters():
                param.requires_grad = False
            for param in self._ema_obs_proj.parameters():
                param.requires_grad = False
            self._ema_updates = 0
            modules.update({
                "prototypes": self._prototypes,
                "obs_proj": self.obs_proj,
                "feat_proj": self.feat_proj,
                "ema_encoder": self._ema_encoder,
                "ema_obs_proj": self._ema_obs_proj,
            })

        # THICK-specific modules
        if self._thick_enabled:
            if self.rep_loss == "dreamer":
                self.coarse_decoder = networks.MultiDecoder(
                    config.decoder,
                    self.rssm._context_size,
                    self.rssm.flat_stoch,
                    shapes,
                )
                modules["coarse_decoder"] = self.coarse_decoder
            elif self.rep_loss in ("r2dreamer", "infonce"):
                self.coarse_prj = Projector(self.rssm.coarse_feat_size, self.embed_size)
                modules["coarse_projector"] = self.coarse_prj

            if self._use_coarse_critic:
                self.coarse_value = networks.MLPHead(config.critic, self.rssm.coarse_feat_size)
                self._slow_coarse_value = copy.deepcopy(self.coarse_value)
                for param in self._slow_coarse_value.parameters():
                    param.requires_grad = False
                modules["coarse_value"] = self.coarse_value

        # Build optimizer
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}")

        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                self._named_params[name] = module
            else:
                for param_name, param in module.named_parameters():
                    self._named_params[f"{name}.{param_name}"] = param
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        self._scaler = GradScaler()

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        self.train()
        self.clone_and_freeze()
        if config.compile:
            print("Compiling update function with torch.compile...")
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")

    def _update_slow_target(self):
        """Update slow-moving value target network(s)."""
        should_update = self._slow_value_updates % self.slow_target_update == 0
        if should_update:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
                if self._thick_enabled and self._use_coarse_critic:
                    for v, s in zip(self.coarse_value.parameters(), self._slow_coarse_value.parameters()):
                        s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def train(self, mode=True):
        super().train(mode)
        self._slow_value.train(False)
        if self._thick_enabled and getattr(self, "_use_coarse_critic", False) and hasattr(self, "_slow_coarse_value"):
            self._slow_coarse_value.train(False)
        return self

    def clone_and_freeze(self):
        self._frozen_encoder = copy.deepcopy(self.encoder)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.encoder.named_parameters(), self._frozen_encoder.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_rssm = copy.deepcopy(self.rssm)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.rssm.named_parameters(), self._frozen_rssm.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_reward = copy.deepcopy(self.reward)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.reward.named_parameters(), self._frozen_reward.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_cont = copy.deepcopy(self.cont)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.cont.named_parameters(), self._frozen_cont.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_actor = copy.deepcopy(self.actor)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.actor.named_parameters(), self._frozen_actor.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_value = copy.deepcopy(self.value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.value.named_parameters(), self._frozen_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_slow_value = copy.deepcopy(self._slow_value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self._slow_value.named_parameters(), self._frozen_slow_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        if self._thick_enabled and self._use_coarse_critic and hasattr(self, "coarse_value"):
            self._frozen_coarse_value = copy.deepcopy(self.coarse_value)
            for (name_orig, param_orig), (name_new, param_new) in zip(
                self.coarse_value.named_parameters(), self._frozen_coarse_value.named_parameters()
            ):
                assert name_orig == name_new
                param_new.data = param_orig.data
                param_new.requires_grad_(False)

            self._frozen_slow_coarse_value = copy.deepcopy(self._slow_coarse_value)
            for (name_orig, param_orig), (name_new, param_new) in zip(
                self._slow_coarse_value.named_parameters(), self._frozen_slow_coarse_value.named_parameters()
            ):
                assert name_orig == name_new
                param_new.data = param_orig.data
                param_new.requires_grad_(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.clone_and_freeze()
        return self

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step."""
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        embed = self._frozen_encoder(p_obs)
        prev_stoch = state["stoch"]
        prev_deter = state["deter"]
        prev_context = state.get("context", None)
        prev_action = state["prev_action"]

        stoch, deter, context, _, _, _ = self._frozen_rssm.obs_step(
            prev_stoch, prev_deter, prev_context, prev_action, embed, obs["is_first"]
        )
        feat = self._frozen_rssm.get_feat(stoch, deter, context)
        action_dist = self._frozen_actor(feat)
        action = action_dist.mode if eval else action_dist.rsample()
        new_state = {"stoch": stoch, "deter": deter, "prev_action": action}
        if context is not None:
            new_state["context"] = context
        return action, TensorDict(new_state, batch_size=state.batch_size)

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter, context = self.rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        state = {"stoch": stoch, "deter": deter, "prev_action": action}
        if context is not None:
            state["context"] = context
        return TensorDict(state, batch_size=(B,))

    @torch.no_grad()
    def video_pred(self, data, initial):
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        return self._video_pred(p_data, initial)

    def _video_pred(self, data, initial):
        """Video prediction utility."""
        if self.rep_loss != "dreamer":
            raise NotImplementedError("video_pred requires decoder and is only supported when rep_loss == 'dreamer'.")

        B = min(data["action"].shape[0], 6)
        embed = self.encoder(data)

        post_stoch, post_deter, post_context, _, _, _ = self.rssm.observe(
            embed[:B, :5],
            data["action"][:B, :5],
            tuple(val[:B] if val is not None else None for val in initial),
            data["is_first"][:B, :5],
        )
        recon = self.decoder(post_stoch, post_deter)["image"].mode()[:B]
        init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
        init_context = post_context[:, -1] if post_context is not None else None
        prior_stoch, prior_deter, _, _ = self.rssm.imagine_with_action(
            init_stoch,
            init_deter,
            init_context,
            data["action"][:B, 5:],
        )
        openl = self.decoder(prior_stoch, prior_deter)["image"].mode()
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:B]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)

    def context_heatmaps(self):
        """Generate context and gate heatmap images for TensorBoard.

        Returns a dict of name -> numpy array in (C, H, W) format for add_image,
        or empty dict if THICK is disabled or no data yet.
        """
        if not self._thick_enabled or not hasattr(self, "_last_context"):
            return {}
        import numpy as np

        results = {}
        # Context heatmap: (T, C) -> (C, T) image
        ctx = self._last_context.cpu().float()  # (T, C)
        # Normalize per-dimension to [0, 1] for visibility
        cmin = ctx.min(dim=0, keepdim=True).values
        cmax = ctx.max(dim=0, keepdim=True).values
        ctx_norm = (ctx - cmin) / (cmax - cmin + 1e-8)
        # (C, T) -> (1, C, T) grayscale image
        results["context_heatmap"] = ctx_norm.T.unsqueeze(0).numpy()

        # Gate heatmap: (T, H_gate) -> (H_gate, T) image
        gate = self._last_gates.cpu().float()  # (T, H_gate)
        # For TimeLord H_gate=1, for GateLord/Binary H_gate=context_size
        # Clamp to [0, 1] (gates are already in this range but be safe)
        gate_norm = gate.clamp(0, 1)
        results["gate_heatmap"] = gate_norm.T.unsqueeze(0).numpy()

        return results

    def update(self, replay_buffer):
        """Sample a batch from replay and perform one optimization step."""
        data, index, initial = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        self._update_slow_target()
        if self.rep_loss == "dreamerpro":
            self.ema_update()
        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16):
            (stoch, deter, context), mets = self._cal_grad(p_data, initial)
        self._scaler.unscale_(self._optimizer)
        if self.rep_loss == "dreamerpro" and self._ema_updates < self.freeze_prototypes_iters:
            self._prototypes.grad.zero_()
        if self._log_grads:
            old_params = [p.data.clone().detach() for p in self._named_params.values()]
            grads = [p.grad for p in self._named_params.values() if p.grad is not None]
            grad_norm = tools.compute_global_norm(grads)
            grad_rms = tools.compute_rms(grads)
            mets["opt/grad_norm"] = grad_norm
            mets["opt/grad_rms"] = grad_rms
        self._agc(self._named_params.values())
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)
        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        if self._log_grads:
            updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
            update_rms = tools.compute_rms(updates)
            params_rms = tools.compute_rms(self._named_params.values())
            mets["opt/param_rms"] = params_rms
            mets["opt/update_rms"] = update_rms
        metrics.update(mets)
        replay_buffer.update(index, stoch.detach(), deter.detach(), context.detach() if context is not None else None)
        return metrics

    def _cal_grad(self, data, initial):
        """Compute gradients for one batch."""
        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout and KL losses ===
        embed = self.encoder(data)
        post_stoch, post_deter, post_context, post_logit, coarse_logit, gates = self.rssm.observe(
            embed, data["action"], initial, data["is_first"]
        )

        _, prior_logit = self.rssm.prior(post_deter, post_context)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)

        # THICK coarse KL + sparsity losses
        if self._thick_enabled:
            coarse_dyn, coarse_rep = self.rssm.coarse_kl_loss(post_logit, coarse_logit, self.kl_free)
            losses["coarse_dyn"] = torch.mean(coarse_dyn)
            losses["coarse_rep"] = torch.mean(coarse_rep)
            losses["sparse"] = sparse_loss(gates, free_nats=self._sparse_free)

        # === Representation / auxiliary losses ===
        feat = self.rssm.get_feat(post_stoch, post_deter, post_context)
        if self.rep_loss == "dreamer":
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key])) for key, dist in self.decoder(post_stoch, post_deter).items()
            }
            losses.update(recon_losses)
            if self._thick_enabled:
                coarse_recon_losses = {
                    f"coarse_{key}": torch.mean(-dist.log_prob(data[key]))
                    for key, dist in self.coarse_decoder(post_stoch, post_context).items()
                }
                losses["coarse_recon"] = sum(coarse_recon_losses.values())
        elif self.rep_loss == "r2dreamer":
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            x2 = embed.reshape(B * T, -1).detach()
            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)
            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
            if self._thick_enabled:
                coarse_feat = self.rssm.get_coarse_feat(post_stoch, post_context)
                cx1 = self.coarse_prj(coarse_feat.reshape(B * T, -1))
                cx2 = embed.reshape(B * T, -1).detach()
                cx1_norm = (cx1 - cx1.mean(0)) / (cx1.std(0) + 1e-8)
                cx2_norm = (cx2 - cx2.mean(0)) / (cx2.std(0) + 1e-8)
                cc = torch.mm(cx1_norm.T, cx2_norm) / (B * T)
                c_invariance = (torch.diagonal(cc) - 1.0).pow(2).sum()
                c_off_diag_mask = ~torch.eye(cx1.shape[-1], dtype=torch.bool, device=cx1.device)
                c_redundancy = cc[c_off_diag_mask].pow(2).sum()
                losses["coarse_barlow"] = c_invariance + self.barlow_lambd * c_redundancy
        elif self.rep_loss == "infonce":
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            x2 = embed.reshape(B * T, -1).detach()
            logits = torch.matmul(x1, x2.T)
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(norm_logits.shape[0]).long().to(self.device)
            losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
            if self._thick_enabled:
                coarse_feat = self.rssm.get_coarse_feat(post_stoch, post_context)
                cx1 = self.coarse_prj(coarse_feat.reshape(B * T, -1))
                cx2 = embed.reshape(B * T, -1).detach()
                c_logits = torch.matmul(cx1, cx2.T)
                c_norm_logits = c_logits - torch.max(c_logits, 1)[0][:, None]
                losses["coarse_infonce"] = F.cross_entropy(c_norm_logits, labels)
        elif self.rep_loss == "dreamerpro":
            with torch.no_grad():
                data_aug = self.augment_data(data)
                initial_aug = tuple(
                    torch.cat([v, v], dim=0) if v is not None else None for v in initial
                )
                ema_proj = self.ema_proj(data_aug)
            embed_aug = self.encoder(data_aug)
            post_stoch_aug, post_deter_aug, post_context_aug, _, _, _ = self.rssm.observe(
                embed_aug, data_aug["action"], initial_aug, data_aug["is_first"]
            )
            proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj, post_context_aug)
            losses.update(proto_losses)
        else:
            raise NotImplementedError

        # reward and continue
        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))
        cont = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))
        # log
        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())
        if self._thick_enabled:
            # Scalar gate metrics
            gate_open = (gates > 0).float()
            # Fraction of (batch, time, unit) entries where gate is open
            metrics["gate_frac"] = gate_open.mean()
            # Fraction of timesteps where *any* gate unit fires (B, T)
            metrics["gate_step_frac"] = gate_open.max(dim=-1).values.mean()
            # Mean magnitude of non-zero gates (fixed-shape to avoid torch.compile recompilation)
            gate_sum = (gates * gate_open).sum()
            gate_count = gate_open.sum()
            metrics["gate_magnitude"] = gate_sum / gate_count.clamp(min=1)
            # Context change: L2 norm of consecutive differences, averaged
            ctx_diff = post_context[:, 1:] - post_context[:, :-1]
            metrics["context_change_norm"] = ctx_diff.norm(dim=-1).mean()
            # Store for heatmap logging (first batch element only, detached)
            self._last_context = post_context[0].detach()  # (T, C)
            self._last_gates = gates[0].detach()  # (T, H_gate)

        # === Imagination rollout for actor-critic ===
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
            post_context.reshape(-1, *post_context.shape[2:]).detach() if post_context is not None else None,
        )
        imag_feat, imag_action, imag_coarse_feat = self._imagine(start, self.imag_horizon + 1)
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()
        if imag_coarse_feat is not None:
            imag_coarse_feat = imag_coarse_feat.detach()

        imag_reward = self._frozen_reward(imag_feat).mode()
        imag_cont = self._frozen_cont(imag_feat).mean
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont

        if self._thick_enabled and self._use_coarse_critic:
            imag_coarse_value = self._frozen_coarse_value(imag_coarse_feat).mode()
            mixed_value = self._psi * imag_value + (1 - self._psi) * imag_coarse_value
            ret = self._lambda_return(last, term, imag_reward, mixed_value, mixed_value, disc, self.lamb)
        else:
            ret = self._lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)

        ret_offset, ret_scale = self.return_ema(ret)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_feat)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))

        imag_value_dist = self.value(imag_feat)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )

        # Coarse value loss (if enabled)
        if self._thick_enabled and self._use_coarse_critic:
            imag_coarse_value_dist = self.coarse_value(imag_coarse_feat)
            losses["coarse_value"] = torch.mean(
                weight[:, :-1].detach()
                * (-imag_coarse_value_dist.log_prob(tar_padded.detach()))[
                    :, :-1
                ].unsqueeze(-1)
            )

        # log
        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))

        # === Replay-based value learning (keep gradients through world model) ===
        last, term, reward = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        feat = self.rssm.get_feat(post_stoch, post_deter, post_context)
        boot = ret[:, 0].reshape(B, T, 1)
        value = self._frozen_value(feat).mode()
        slow_value = self._frozen_slow_value(feat).mode()
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

        value_dist = self.value(feat)
        losses["repval"] = torch.mean(
            weight[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(
                -1
            )
        )

        # Coarse replay value loss
        if self._thick_enabled and self._use_coarse_critic:
            coarse_feat_replay = self.rssm.get_coarse_feat(post_stoch, post_context)
            coarse_value_dist_replay = self.coarse_value(coarse_feat_replay)
            losses["coarse_repval"] = torch.mean(
                weight[:, :-1]
                * (-coarse_value_dist_replay.log_prob(ret_padded.detach()))[:, :-1].unsqueeze(-1)
            )

        # log
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter, post_context), metrics

    @torch.no_grad()
    def _imagine(self, start, imag_horizon):
        """Roll out the policy in latent space."""
        feats = []
        actions = []
        stoch, deter, context = start
        use_coarse = self._thick_enabled and self._use_coarse_critic
        coarse_feats = [] if use_coarse else None
        for _ in range(imag_horizon):
            feat = self._frozen_rssm.get_feat(stoch, deter, context)
            action = self._frozen_actor(feat).rsample()
            feats.append(feat)
            actions.append(action)
            if use_coarse:
                coarse_feats.append(self._frozen_rssm.get_coarse_feat(stoch, context))
            stoch, deter, context, _ = self._frozen_rssm.img_step(stoch, deter, context, action)

        stacked_feats = torch.stack(feats, dim=1)
        stacked_actions = torch.stack(actions, dim=1)
        stacked_coarse = torch.stack(coarse_feats, dim=1) if coarse_feats is not None else None
        return stacked_feats, stacked_actions, stacked_coarse

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        """
        lamb=1 means discounted Monte Carlo return.
        lamb=0 means fixed 1-step return.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data["image"] = to_f32(data["image"]) / 255.0
        return data

    @torch.no_grad()
    def augment_data(self, data):
        data_aug = {k: torch.cat([v, v], axis=0) for k, v in data.items()}
        image = data_aug["image"].permute(0, 1, 4, 2, 3)
        data_aug["image"] = self.random_translate(
            image,
            self.aug_max_delta,
            same_across_time=self.aug_same_across_time,
            bilinear=self.aug_bilinear,
        )
        data_aug["image"] = data_aug["image"].permute(0, 1, 3, 4, 2)
        return data_aug

    @torch.no_grad()
    def ema_proj(self, data):
        with torch.no_grad():
            embed = self._ema_encoder(data)
            proj = self._ema_obs_proj(embed)
        return F.normalize(proj, p=2, dim=-1)

    @torch.no_grad()
    def ema_update(self):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)
        self._prototypes.data.copy_(prototypes)
        if self._ema_updates % self.ema_update_every == 0:
            mix = self.ema_update_fraction if self._ema_updates > 0 else 1.0
            for s, d in zip(self.encoder.parameters(), self._ema_encoder.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
            for s, d in zip(self.obs_proj.parameters(), self._ema_obs_proj.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self._ema_updates += 1

    def sinkhorn(self, scores):
        shape = scores.shape
        K = shape[0]
        scores = scores.reshape(-1)
        log_Q = F.log_softmax(scores / self.sinkhorn_eps, dim=0)
        log_Q = log_Q.reshape(K, -1)
        N = log_Q.shape[1]
        for _ in range(self.sinkhorn_iters):
            log_row_sums = torch.logsumexp(log_Q, dim=1, keepdim=True)
            log_Q = log_Q - log_row_sums - math.log(K)
            log_col_sums = torch.logsumexp(log_Q, dim=0, keepdim=True)
            log_Q = log_Q - log_col_sums - math.log(N)
        log_Q = log_Q + math.log(N)
        Q = torch.exp(log_Q)
        return Q.reshape(shape)

    def proto_loss(self, post_stoch, post_deter, embed, ema_proj, post_context=None):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)

        obs_proj = self.obs_proj(embed)
        obs_norm = torch.norm(obs_proj, dim=-1)
        obs_proj = F.normalize(obs_proj, p=2, dim=-1)

        B, T = obs_proj.shape[:2]
        obs_proj = obs_proj.reshape(B * T, -1)
        obs_scores = torch.matmul(obs_proj, prototypes.T)
        obs_scores = obs_scores.reshape(B, T, -1).permute(2, 0, 1)
        obs_scores = obs_scores[:, :, self.warm_up :]
        obs_logits = F.log_softmax(obs_scores / self.temperature, dim=0)
        obs_logits_1, obs_logits_2 = torch.chunk(obs_logits, 2, dim=1)

        ema_proj = ema_proj.reshape(B * T, -1)
        ema_scores = torch.matmul(ema_proj, prototypes.T)
        ema_scores = ema_scores.reshape(B, T, -1).permute(2, 0, 1)
        ema_scores = ema_scores[:, :, self.warm_up :]
        ema_scores_1, ema_scores_2 = torch.chunk(ema_scores, 2, dim=1)

        with torch.no_grad():
            ema_targets_1 = self.sinkhorn(ema_scores_1)
            ema_targets_2 = self.sinkhorn(ema_scores_2)
        ema_targets = torch.cat([ema_targets_1, ema_targets_2], dim=1)

        feat = self.rssm.get_feat(post_stoch, post_deter, post_context)
        feat_proj = self.feat_proj(feat)
        feat_norm = torch.norm(feat_proj, dim=-1)
        feat_proj = F.normalize(feat_proj, p=2, dim=-1)

        feat_proj = feat_proj.reshape(B * T, -1)
        feat_scores = torch.matmul(feat_proj, prototypes.T)
        feat_scores = feat_scores.reshape(B, T, -1).permute(2, 0, 1)
        feat_scores = feat_scores[:, :, self.warm_up :]
        feat_logits = F.log_softmax(feat_scores / self.temperature, dim=0)

        swav_loss = -0.5 * torch.mean(torch.sum(ema_targets_2 * obs_logits_1, dim=0)) - 0.5 * torch.mean(
            torch.sum(ema_targets_1 * obs_logits_2, dim=0)
        )
        temp_loss = -torch.mean(torch.sum(ema_targets * feat_logits, dim=0))
        norm_loss = torch.mean(torch.square(obs_norm - 1)) + torch.mean(torch.square(feat_norm - 1))

        return {
            "swav": swav_loss,
            "temp": temp_loss,
            "norm": norm_loss,
        }

    @torch.no_grad()
    def random_translate(self, x, max_delta, same_across_time=False, bilinear=False):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        pad = int(max_delta)

        x_padded = F.pad(x_flat, (pad, pad, pad, pad), "replicate")
        h_padded, w_padded = H + 2 * pad, W + 2 * pad

        eps_h = 1.0 / h_padded
        eps_w = 1.0 / w_padded
        arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h_padded, device=x.device, dtype=x.dtype)[:H]
        arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w_padded, device=x.device, dtype=x.dtype)[:W]
        arange_h = arange_h.unsqueeze(1).repeat(1, W).unsqueeze(2)
        arange_w = arange_w.unsqueeze(0).repeat(H, 1).unsqueeze(2)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(B * T, 1, 1, 1)

        if same_across_time:
            shift = torch.randint(0, 2 * pad + 1, size=(B, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
            shift = shift.repeat(1, T, 1, 1, 1).reshape(B * T, 1, 1, 2)
        else:
            shift = torch.randint(0, 2 * pad + 1, size=(B * T, 1, 1, 2), device=x.device, dtype=x.dtype)

        shift = shift * 2.0 / torch.tensor([w_padded, h_padded], device=x.device, dtype=x.dtype)

        grid = base_grid + shift
        mode = "bilinear" if bilinear else "nearest"
        x_translated = F.grid_sample(x_padded, grid, mode=mode, padding_mode="zeros", align_corners=False)

        return x_translated.reshape(B, T, C, H, W)
