import pathlib

import numpy as np
import torch

import tools


class OnlineTrainer:
    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._action_repeat = config.action_repeat
        # Video saving to disk every 100k steps
        self._should_save_video = tools.Every(1e5)
        self._video_dir = pathlib.Path(logdir) / "videos"

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        Environment stepping is executed on CPU to avoid GPU<->CPU synchronizations
        in the worker processes. Observations are moved back to GPU asynchronously
        (H2D with non_blocking=True) right before policy inference.
        """
        print("Evaluating the policy...")
        save_video = self._should_save_video(train_step)
        envs = self.eval_envs
        agent.eval()
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        log_metrics = {}
        # cache is only used for video logging / open-loop prediction.
        cache = []
        context_cache = []  # collect context vectors for overlay
        video_frames = []  # full episode frames for disk video (env 0 only)
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        env0_done = False
        while not once_done.all():
            steps += ~done * ~once_done
            # Step environments on CPU.
            # (B, A)
            act_cpu = act.detach().to("cpu")
            # (B,)
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
            # Move observations back to GPU asynchronously for the agent.
            # dict of (B, 1, *)
            trans = trans_cpu.to(agent.device, non_blocking=True)
            # (B,)
            done = done_cpu.to(agent.device)

            # Store transition.
            # We keep the observation and the action that produced it together.
            trans["action"] = act
            if len(cache) < self.batch_length:
                cache.append(trans.clone())
            # Collect full episode frames for env 0 (for disk video saving)
            if save_video and not env0_done and "image" in trans:
                video_frames.append(tools.to_np(trans["image"][0, 0]))  # (H, W, C)
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            # Collect context/gate for video overlay (env 0 only)
            if "context" in agent_state.keys():
                context_cache.append(agent_state["context"][0].detach().cpu())
            returns += trans["reward"][:, 0] * ~once_done
            for key, value in trans.items():
                if key.startswith("log_"):
                    if key not in log_metrics:
                        log_metrics[key] = torch.zeros_like(returns)
                    log_metrics[key] += value[:, 0] * ~once_done
            if done[0]:
                env0_done = True
            once_done |= done
        # dict of (B, T, *)
        cache = torch.stack(cache, dim=1) if len(cache) else None
        self.logger.scalar("episode/eval_score", returns.mean())
        self.logger.scalar("episode/eval_length", steps.to(torch.float32).mean())
        for key, value in log_metrics.items():
            if key == "log_success":
                value = torch.clip(value, max=1.0)  # make sure 1.0 for success episode
            self.logger.scalar(f"episode/eval_{key[4:]}", value.mean())
        if cache is not None and "image" in cache:
            video = tools.to_np(cache["image"][:1])  # (1, T, H, W, C)
            if len(context_cache) > 0:
                video = self._overlay_context(video, context_cache)
            self.logger.video("eval_video", video)
        # Save full eval episode video to disk every 100k steps
        if save_video and len(video_frames) > 0:
            full_video = np.stack(video_frames, axis=0)[None]  # (1, T, H, W, C)
            if len(context_cache) > 0:
                full_video = self._overlay_context(full_video, context_cache)
            frames = full_video[0]  # (T, H, W, C)
            if np.issubdtype(frames.dtype, np.floating):
                frames = np.clip(255 * frames, 0, 255).astype(np.uint8)
            path = self._video_dir / f"step_{train_step:09d}.mp4"
            tools.save_video(path, frames, fps=16)
            print(f"Saved eval video: {path}")
        if self.video_pred_log and cache is not None:
            initial = agent.get_initial_state(1)
            if "context" in initial.keys():
                init_tuple = (initial["stoch"], initial["deter"], initial["context"])
            else:
                init_tuple = (initial["stoch"], initial["deter"], None)
            self.logger.video(
                "eval_open_loop",
                tools.to_np(
                    agent.video_pred(
                        cache[:1],  # give only first batch
                        init_tuple,
                    )
                ),
            )
        self.logger.write(train_step)
        agent.train()

    @staticmethod
    def _overlay_context(video, context_cache):
        """Append context visualization strip to the right of each video frame.

        Args:
            video: (1, T_vid, H, W, C) numpy array, float [0,1] or uint8.
            context_cache: list of (C_ctx,) tensors, one per eval step.

        Returns:
            (1, T_vid, H, W + strip_width, C) numpy array.
        """
        is_float = np.issubdtype(video.dtype, np.floating)
        T_vid = video.shape[1]
        H, W, C = video.shape[2], video.shape[3], video.shape[4]

        # Stack context: (T_ctx, C_ctx)
        ctx = torch.stack(context_cache).float().numpy()
        T_ctx, C_ctx = ctx.shape
        # Truncate or pad to match video length
        if T_ctx > T_vid:
            ctx = ctx[:T_vid]
        elif T_ctx < T_vid:
            ctx = np.pad(ctx, ((0, T_vid - T_ctx), (0, 0)), mode="edge")

        # Normalize globally for consistent coloring across time
        cmin, cmax = ctx.min(), ctx.max()
        ctx_norm = (ctx - cmin) / (cmax - cmin + 1e-8)  # (T, C_ctx) in [0, 1]

        # Render strip: scale C_ctx rows to image height H
        # Each context dim gets H // C_ctx pixels (at least 1)
        pixels_per_dim = max(1, H // C_ctx)
        strip_h = pixels_per_dim * C_ctx
        # (T, C_ctx) -> (T, strip_h) by repeating each dim
        ctx_expanded = np.repeat(ctx_norm, pixels_per_dim, axis=1)  # (T, strip_h)
        # Pad or crop to exactly H
        if strip_h < H:
            ctx_expanded = np.pad(ctx_expanded, ((0, 0), (0, H - strip_h)), mode="constant")
        else:
            ctx_expanded = ctx_expanded[:, :H]

        # Convert to RGB using a blue-red colormap
        # 0 = blue (0, 0, 1), 1 = red (1, 0, 0)
        strip_w = 8
        strip = np.zeros((T_vid, H, strip_w, 3), dtype=np.float32)
        vals = ctx_expanded[:, :, None]  # (T, H, 1)
        strip[..., 0] = vals  # R
        strip[..., 2] = 1.0 - vals  # B

        # Add a 1px black separator
        sep = np.zeros((T_vid, H, 1, 3), dtype=np.float32)

        if not is_float:
            video_f = video.astype(np.float32) / 255.0
        else:
            video_f = video

        # video_f: (1, T, H, W, C) -> (T, H, W, C) for env 0
        frames = video_f[0]  # (T, H, W, C)
        # Only use RGB channels
        frames_rgb = frames[..., :3]
        # Concat: frame | separator | context strip
        combined = np.concatenate([frames_rgb, sep, strip], axis=2)  # (T, H, W+1+strip_w, 3)
        return combined[None]  # (1, T, H, W+1+strip_w, 3)

    def begin(self, agent):
        """Main online training loop.

        The loop is designed to overlap CPU environment stepping and GPU model
        execution. Environments are stepped on CPU, observations are pinned,
        then transferred to GPU with non_blocking=True.
        """
        envs = self.train_envs
        video_cache = []
        step = self.replay_buffer.count() * self._action_repeat
        update_count = 0
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        episode_ids = torch.arange(
            envs.env_num, dtype=torch.int32, device=agent.device
        )  # Increment this to prevent sampling across episode boundaries
        train_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        _last_mem_step = 0
        while step < self.steps:
            if step - _last_mem_step >= 50000:
                import gc; gc.collect(); torch.cuda.empty_cache()
                alloc = torch.cuda.memory_allocated() / 1e9
                res = torch.cuda.memory_reserved() / 1e9
                maxalloc = torch.cuda.max_memory_allocated() / 1e9
                print(f"[MEM step={step}] alloc={alloc:.2f}GB reserved={res:.2f}GB max_alloc={maxalloc:.2f}GB")
                _last_mem_step = step
            # Evaluation
            if self._should_eval(step) and self.eval_episode_num > 0:
                self.eval(agent, step)
            # Save metrics
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        if i == 0 and len(video_cache) > 0:
                            video = torch.stack(video_cache, axis=0)
                            self.logger.video("train_video", tools.to_np(video[None]))
                            video_cache = []
                        self.logger.scalar("episode/score", returns[i])
                        self.logger.scalar("episode/length", lengths[i])
                        self.logger.write(step + i)  # to show all values on tensorboard
                        returns[i] = lengths[i] = 0
            step += int((~done).sum()) * self._action_repeat  # step is based on env side
            lengths += ~done

            # Step environments on CPU to avoid GPU<->CPU sync in the worker processes.
            # (B, A)
            act_cpu = act.detach().to("cpu")
            # (B,)
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)

            # Move observations back to GPU asynchronously for the agent.
            # dict of (B, 1, *)
            trans = trans_cpu.to(agent.device, non_blocking=True)
            # (B,)
            done = done_cpu.to(agent.device)

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # (B, A)
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)

            # Store transition.
            # We keep the observation and the action that produced it together.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            if "context" in agent_state.keys():
                trans["context"] = agent_state["context"]
            trans["episode"] = episode_ids  # Don't lift dim
            if "image" in trans:
                video_cache.append(trans["image"][0])
            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]
            # Update models after enough data has accumulated
            if step // (envs.env_num * self._action_repeat) > self.batch_length + 1:
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(step)
                for _ in range(update_num):
                    _metrics = agent.update(self.replay_buffer)
                    train_metrics = _metrics
                update_count += update_num
                # Log training metrics
                if self._should_log(step):
                    for name, value in train_metrics.items():
                        value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/opt/updates", update_count)
                    if self.video_pred_log:
                        data, _, initial = self.replay_buffer.sample()
                        self.logger.video("open_loop", tools.to_np(agent.video_pred(data, initial)))
                    for name, img in agent.context_heatmaps().items():
                        self.logger.image(f"train/{name}", img)
                    if self.params_hist_log:
                        for name, param in agent._named_params.items():
                            self.logger.histogram(name, tools.to_np(param))
                    self.logger.write(step, fps=True)
