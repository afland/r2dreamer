"""ThickDreamer — Dreamer with THICK temporal hierarchy.

All THICK-specific training logic lives here.  The base Dreamer class
is extended with coarse losses, an optional coarse critic, and the
necessary overrides to propagate the ``context`` vector through
observe / imagine / act.
"""

import copy

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.amp import autocast

import networks
import tools
from crssm import CRSSM
from dreamer import Dreamer
from gatelord import sparse_loss
from networks import Projector
from tools import to_f32


class ThickDreamer(Dreamer):
    """Dreamer with THICK temporal hierarchy.

    Uses Dreamer's hook methods (_make_rssm, _extend_modules) to inject
    CRSSM and coarse modules without rebuilding.
    """

    def __init__(self, config, obs_space, act_space):
        # Store THICK config before super().__init__ since hooks need it.
        self._thick_config = config.thick
        self._sparse_free = float(config.rssm.sparse_free)
        self.use_coarse_critic = bool(config.thick.coarse_critic)
        self._psi = float(config.thick.psi)
        super().__init__(config, obs_space, act_space)

    def _make_rssm(self, config, embed_size, act_dim):
        """Use CRSSM instead of RSSM."""
        return CRSSM(config.rssm, embed_size, act_dim)

    def _extend_modules(self, config, obs_space, act_space, shapes, modules):
        """Add coarse rep loss modules and optional coarse critic."""
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

        if self.use_coarse_critic:
            self.coarse_value = networks.MLPHead(config.critic, self.rssm.coarse_feat_size)
            self._slow_coarse_value = copy.deepcopy(self.coarse_value)
            for param in self._slow_coarse_value.parameters():
                param.requires_grad = False
            modules["coarse_value"] = self.coarse_value

        return modules

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "use_coarse_critic", False) and hasattr(self, "_slow_coarse_value"):
            self._slow_coarse_value.train(False)
        return self

    def _update_slow_target(self):
        """Update slow-moving value target networks."""
        super()._update_slow_target()
        if self.use_coarse_critic:
            if self._slow_value_updates % self.slow_target_update == 0:
                with torch.no_grad():
                    mix = self.slow_target_fraction
                    for v, s in zip(self.coarse_value.parameters(), self._slow_coarse_value.parameters()):
                        s.data.copy_(mix * v.data + (1 - mix) * s.data)

    def clone_and_freeze(self):
        super().clone_and_freeze()
        if self.use_coarse_critic and hasattr(self, "coarse_value"):
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

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter, context = self.rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        return TensorDict(
            {"stoch": stoch, "deter": deter, "context": context, "prev_action": action},
            batch_size=(B,),
        )

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step with coarse context."""
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        embed = self._frozen_encoder(p_obs)
        prev_stoch, prev_deter, prev_context, prev_action = (
            state["stoch"],
            state["deter"],
            state["context"],
            state["prev_action"],
        )
        stoch, deter, context, _, _, _ = self._frozen_rssm.obs_step(
            prev_stoch, prev_deter, prev_context, prev_action, embed, obs["is_first"]
        )
        feat = self._frozen_rssm.get_feat(stoch, deter, context)
        action_dist = self._frozen_actor(feat)
        action = action_dist.mode if eval else action_dist.rsample()
        return action, TensorDict(
            {"stoch": stoch, "deter": deter, "context": context, "prev_action": action},
            batch_size=state.batch_size,
        )

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
        replay_buffer.update(index, stoch.detach(), deter.detach(), context.detach())
        return metrics

    # ------------------------------------------------------------------
    # Core gradient computation
    # ------------------------------------------------------------------

    def _cal_grad(self, data, initial):
        """Compute gradients for one batch (THICK version).

        Adds coarse_dyn, coarse_rep, sparse, and coarse representation
        losses on top of the standard Dreamer losses.  Optionally adds
        a coarse critic with mixed value targets.
        """

        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout ===
        embed = self.encoder(data)
        # CRSSM returns 6 values.
        post_stoch, post_deter, post_context, post_logit, coarse_logit, gates = self.rssm.observe(
            embed, data["action"], initial, data["is_first"]
        )

        # Standard KL losses.
        _, prior_logit = self.rssm.prior(post_deter, post_context)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)

        # Coarse KL losses.
        coarse_dyn, coarse_rep = self.rssm.coarse_kl_loss(post_logit, coarse_logit, self.kl_free)
        losses["coarse_dyn"] = torch.mean(coarse_dyn)
        losses["coarse_rep"] = torch.mean(coarse_rep)

        # Sparsity loss.
        losses["sparse"] = sparse_loss(gates, free_nats=self._sparse_free)

        # === Representation / auxiliary losses ===
        feat = self.rssm.get_feat(post_stoch, post_deter, post_context)

        if self.rep_loss == "dreamer":
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key]))
                for key, dist in self.decoder(post_stoch, post_deter).items()
            }
            losses.update(recon_losses)
            # Coarse reconstruction loss.
            coarse_recon_losses = {
                f"coarse_{key}": torch.mean(-dist.log_prob(data[key]))
                for key, dist in self.coarse_decoder(post_stoch, post_context).items()
            }
            # Sum all coarse recon losses into a single coarse_recon loss.
            losses["coarse_recon"] = sum(coarse_recon_losses.values())
        elif self.rep_loss == "r2dreamer":
            # Standard Barlow Twins.
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            x2 = embed.reshape(B * T, -1).detach()
            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)
            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss

            # Coarse Barlow Twins.
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
            # Standard InfoNCE.
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            x2 = embed.reshape(B * T, -1).detach()
            logits = torch.matmul(x1, x2.T)
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(norm_logits.shape[0]).long().to(self.device)
            losses["infonce"] = F.cross_entropy(norm_logits, labels)

            # Coarse InfoNCE.
            coarse_feat = self.rssm.get_coarse_feat(post_stoch, post_context)
            cx1 = self.coarse_prj(coarse_feat.reshape(B * T, -1))
            cx2 = embed.reshape(B * T, -1).detach()
            c_logits = torch.matmul(cx1, cx2.T)
            c_norm_logits = c_logits - torch.max(c_logits, 1)[0][:, None]
            losses["coarse_infonce"] = F.cross_entropy(c_norm_logits, labels)
        elif self.rep_loss == "dreamerpro":
            with torch.no_grad():
                data_aug = self.augment_data(data)
                initial_aug = (
                    torch.cat([initial[0], initial[0]], dim=0),
                    torch.cat([initial[1], initial[1]], dim=0),
                    torch.cat([initial[2], initial[2]], dim=0),
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

        # Reward and continue losses.
        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))
        cont = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))

        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())
        metrics["gate_frac"] = (gates > 0).float().mean()

        # === Imagination rollout for actor-critic ===
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
            post_context.reshape(-1, *post_context.shape[2:]).detach(),
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

        if self.use_coarse_critic:
            # Mixed value target: psi * coarse_value + (1 - psi) * value
            imag_coarse_value = self._frozen_coarse_value(imag_coarse_feat).mode()
            mixed_value = self._psi * imag_coarse_value + (1 - self._psi) * imag_value
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

        # Coarse value loss (if enabled).
        if self.use_coarse_critic:
            imag_coarse_value_dist = self.coarse_value(imag_coarse_feat)
            losses["coarse_value"] = torch.mean(
                weight[:, :-1].detach()
                * (-imag_coarse_value_dist.log_prob(tar_padded.detach()))[
                    :, :-1
                ].unsqueeze(-1)
            )

        # Log metrics.
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

        # === Replay-based value learning ===
        last_replay, term_replay, reward_replay = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        feat_replay = self.rssm.get_feat(post_stoch, post_deter, post_context)
        boot = ret[:, 0].reshape(B, T, 1)
        value_replay = self._frozen_value(feat_replay).mode()
        slow_value_replay = self._frozen_slow_value(feat_replay).mode()
        disc_replay = 1 - 1 / self.horizon
        weight_replay = 1.0 - last_replay
        ret_replay = self._lambda_return(
            last_replay, term_replay, reward_replay, value_replay, boot, disc_replay, self.lamb
        )
        ret_replay_padded = torch.cat([ret_replay, 0 * ret_replay[:, -1:]], 1)

        value_dist_replay = self.value(feat_replay)
        losses["repval"] = torch.mean(
            weight_replay[:, :-1]
            * (
                -value_dist_replay.log_prob(ret_replay_padded.detach())
                - value_dist_replay.log_prob(slow_value_replay.detach())
            )[:, :-1].unsqueeze(-1)
        )

        # Coarse replay value loss.
        if self.use_coarse_critic:
            coarse_feat_replay = self.rssm.get_coarse_feat(post_stoch, post_context)
            coarse_value_dist_replay = self.coarse_value(coarse_feat_replay)
            losses["coarse_repval"] = torch.mean(
                weight_replay[:, :-1]
                * (-coarse_value_dist_replay.log_prob(ret_replay_padded.detach()))[:, :-1].unsqueeze(-1)
            )

        metrics.update(tools.tensorstats(ret_replay, "ret_replay"))
        metrics.update(tools.tensorstats(value_replay, "value_replay"))
        metrics.update(tools.tensorstats(slow_value_replay, "slow_value_replay"))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics["opt/loss"] = total_loss
        return (post_stoch, post_deter, post_context), metrics

    @torch.no_grad()
    def _imagine(self, start, imag_horizon):
        """Roll out the policy in latent space (with coarse context)."""
        feats = []
        actions = []
        coarse_feats = [] if self.use_coarse_critic else None
        stoch, deter, context = start
        for _ in range(imag_horizon):
            feat = self._frozen_rssm.get_feat(stoch, deter, context)
            action = self._frozen_actor(feat).rsample()
            feats.append(feat)
            actions.append(action)
            if self.use_coarse_critic:
                coarse_feats.append(self._frozen_rssm.get_coarse_feat(stoch, context))
            stoch, deter, context, _ = self._frozen_rssm.img_step(stoch, deter, context, action)

        stacked_feats = torch.stack(feats, dim=1)
        stacked_actions = torch.stack(actions, dim=1)
        stacked_coarse = torch.stack(coarse_feats, dim=1) if coarse_feats is not None else None
        return stacked_feats, stacked_actions, stacked_coarse

    def proto_loss(self, post_stoch, post_deter, embed, ema_proj, post_context=None):
        """Override to pass context to get_feat for CRSSM compatibility.

        Note: DreamerPro coarse rep loss is deferred; this override
        only ensures the standard proto_loss works with CRSSM's 3-arg get_feat.
        """
        B, T = post_stoch.shape[:2]
        if post_context is None:
            post_context = torch.zeros(B, T, self.rssm._context_size, device=post_stoch.device)
        feat = self.rssm.get_feat(post_stoch, post_deter, post_context)

        prototypes = F.normalize(self._prototypes, p=2, dim=-1)

        obs_proj = self.obs_proj(embed)
        obs_norm = torch.norm(obs_proj, dim=-1)
        obs_proj = F.normalize(obs_proj, p=2, dim=-1)

        obs_proj_flat = obs_proj.reshape(B * T, -1)
        obs_scores = torch.matmul(obs_proj_flat, prototypes.T)
        obs_scores = obs_scores.reshape(B, T, -1).permute(2, 0, 1)
        obs_scores = obs_scores[:, :, self.warm_up:]
        obs_logits = F.log_softmax(obs_scores / self.temperature, dim=0)
        obs_logits_1, obs_logits_2 = torch.chunk(obs_logits, 2, dim=1)

        ema_proj = ema_proj.reshape(B * T, -1)
        ema_scores = torch.matmul(ema_proj, prototypes.T)
        ema_scores = ema_scores.reshape(B, T, -1).permute(2, 0, 1)
        ema_scores = ema_scores[:, :, self.warm_up:]
        ema_scores_1, ema_scores_2 = torch.chunk(ema_scores, 2, dim=1)

        with torch.no_grad():
            ema_targets_1 = self.sinkhorn(ema_scores_1)
            ema_targets_2 = self.sinkhorn(ema_scores_2)
        ema_targets = torch.cat([ema_targets_1, ema_targets_2], dim=1)

        feat_proj = self.feat_proj(feat)
        feat_norm = torch.norm(feat_proj, dim=-1)
        feat_proj = F.normalize(feat_proj, p=2, dim=-1)

        feat_proj_flat = feat_proj.reshape(B * T, -1)
        feat_scores = torch.matmul(feat_proj_flat, prototypes.T)
        feat_scores = feat_scores.reshape(B, T, -1).permute(2, 0, 1)
        feat_scores = feat_scores[:, :, self.warm_up:]
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

    def _video_pred(self, data, initial):
        """Video prediction utility (THICK version)."""
        if self.rep_loss != "dreamer":
            raise NotImplementedError("video_pred requires decoder and is only supported when rep_loss == 'dreamer'.")

        B = min(data["action"].shape[0], 6)
        embed = self.encoder(data)

        post_stoch, post_deter, post_context, _, _, _ = self.rssm.observe(
            embed[:B, :5],
            data["action"][:B, :5],
            tuple(val[:B] for val in initial),
            data["is_first"][:B, :5],
        )
        recon = self.decoder(post_stoch, post_deter)["image"].mode()[:B]
        init_stoch, init_deter, init_context = post_stoch[:, -1], post_deter[:, -1], post_context[:, -1]
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
