# THICK Implementation Plan for r2dreamer

## Context

THICK (Temporal Hierarchies with Coarse Cognition) adds a sparse temporal gating mechanism (GateLord) on top of the RSSM world model, producing a "coarse context" that captures long-horizon information. This context is integrated into the dynamics, posterior, prior, actor, and critic. Reference implementations exist in `../THICK` (TF/DreamerV2) and `../thix` (JAX/DreamerV3) with inconsistencies resolved per user direction.

**Key design decisions:**
- GateLord: simple (single update gate, no output gate), sparse_over_time_only behavior
- Coarse context always given to actor/critic (not configurable)
- Mixed value target / coarse critic: behind a config flag
- Shared implementation across all variants (DreamerPro coarse rep loss deferred)
- THICK code in its own files to clearly show what was added

---

## File Organization

**New THICK files** (all THICK-specific logic lives here):
- `gatelord.py` — GateLord cell + sparse_loss utility
- `crssm.py` — CRSSM class (extends RSSM with coarse context)
- `thick.py` — ThickDreamer class (extends Dreamer with coarse losses, critic, etc.)

**Minimal changes to existing files:**
- `rssm.py` — Add `context_dim=0` param to `Deter` class (backward-compatible)
- `buffer.py` — Handle optional `context` field (3 lines)
- `trainer.py` — Store `context` in transitions (2 lines)
- `configs/model/_base_.yaml` — Add `thick` config section + loss scales
- `train.py` — Conditional import of ThickDreamer vs Dreamer based on config

---

## 1. New file: `gatelord.py`

Simple GateLord cell. No output gate. Sparse loss uses sparse_over_time_only.

```python
class GateLord(nn.Module):
    def __init__(self, input_size, hidden_size, gate_noise_scale=0.1):
        # _gu: Linear(input_size + hidden_size -> 2 * hidden_size)

    def forward(self, x, hidden):
        # x: (B, input_size), hidden: (B, H)
        gu = self._gu(cat([x, hidden]))
        gate_pre, update = chunk(gu, 2)
        update = tanh(update)
        if self.training: gate_pre += randn * noise_scale
        gate = clamp(tanh(gate_pre), min=0)  # retanh
        h_new = hidden + gate * (update - hidden)
        return h_new, h_new, gate  # output=h_new (no output gate)

def sparse_loss(gates, free_nats=0.0):
    # gates: (B, T, H)
    gate_max = gates.max(dim=-1).values           # (B, T)
    gate_binary = STE_heaviside(gate_max)          # ceil with straight-through
    sparse = gate_binary.sum(dim=1)                # (B,) sum over time
    if free_nats > 0: sparse = clamp(sparse, min=free_nats)
    return sparse / gates.shape[1]                 # normalize by seq_len
```

---

## 2. Modify `rssm.py` — Backward-compatible Deter change

Add `context_dim=0` parameter to `Deter.__init__`. When >0:
- Add `_dyn_in3`: `Linear(context_dim, hidden) + RMSNorm + act`
- Change block input from `(3*hidden + deter//blocks) * blocks` to `(4*hidden + deter//blocks) * blocks`
- Rebuild `_dyn_hid` and `_dyn_gru` with new input size
- `forward(stoch, deter, action, context=None)`: include x3 when context provided

When `context_dim=0` (default), behavior is identical to current code. No changes to `RSSM` class itself.

---

## 3. New file: `crssm.py` — CRSSM class

Extends `RSSM` with coarse context. All coarse-specific logic lives here.

```python
class CRSSM(RSSM):
    """RSSM extended with GateLord coarse context (THICK)."""
```

**Constructor additions** (on top of RSSM.__init__):
- `_context_size`: from config (e.g. 32)
- `_coarse_proj`: `Linear(flat_stoch + act_dim, hidden) + RMSNorm + act` — projects inputs for GateLord
- `_gatelord`: `GateLord(input_size=hidden, hidden_size=context_size)`
- `_coarse_net`: MLP `context_size → stoch*discrete` (coarse prior)
- Override `_deter_net`: `Deter(..., context_dim=context_size)` — adds context input to GRU
- Override `_obs_net`: input `deter + context_size + embed_size` (context in posterior)
- Override `_img_net`: input `deter + context_size` (context in prior)
- `feat_size = flat_stoch + deter + context_size`
- `coarse_feat_size = flat_stoch + context_size`

**New/overridden methods:**

| Method | Signature | Returns |
|--------|-----------|---------|
| `initial(B)` | `int` | `(stoch, deter, context)` — 3-tuple |
| `obs_step(...)` | `stoch, deter, context, prev_action, embed, reset` | `(stoch, deter, context, logit, coarse_logit, gate)` |
| `img_step(...)` | `stoch, deter, context, prev_action` | `(stoch, deter, context, gate)` |
| `observe(...)` | `embed, action, initial, reset` | `(stochs, deters, contexts, logits, coarse_logits, gates)` |
| `prior(...)` | `deter, context` | `(stoch, logit)` |
| `get_feat(...)` | `stoch, deter, context` | `cat(flat_stoch, deter, context)` |
| `get_coarse_feat(...)` | `stoch, context` | `cat(flat_stoch, context)` |
| `coarse_kl_loss(...)` | `post_logit, coarse_logit, free` | `(coarse_dyn, coarse_rep)` |
| `imagine_with_action(...)` | `stoch, deter, context, actions` | `(stochs, deters, contexts, gates)` |

**obs_step data flow:**
1. Reset states on `is_first`
2. `coarse_input = _coarse_proj(cat(flat_stoch, prev_action))`
3. `coarse_out, context, gate = _gatelord(coarse_input, context)`
4. `deter = _deter_net(stoch, deter, prev_action, context)`
5. `logit = _obs_net(cat(deter, context, embed))`, `stoch = sample(logit)`
6. `coarse_logit = _coarse_net(coarse_out)`

---

## 4. New file: `thick.py` — ThickDreamer class

Extends `Dreamer` with all THICK training logic. Overrides key methods.

```python
class ThickDreamer(Dreamer):
    """Dreamer with THICK temporal hierarchy."""
```

**Constructor additions** (on top of Dreamer.__init__):
- Replace `self.rssm = RSSM(...)` with `self.rssm = CRSSM(...)`
- Coarse rep loss modules:
  - **dreamer**: `self.coarse_decoder = MultiDecoder(config.decoder, context_size, flat_stoch, shapes)`
  - **r2dreamer/infonce**: `self.coarse_prj = Projector(rssm.coarse_feat_size, embed_size)`
- Optional coarse critic: `self.coarse_value = MLPHead(config.critic, rssm.coarse_feat_size)` + slow target copy
- All new modules added to optimizer's `_named_params`

**Overridden methods:**

### `_cal_grad(data, initial)` — Full override
Main changes vs parent:
1. `observe` returns 6 values (stochs, deters, contexts, logits, coarse_logits, gates)
2. `prior` takes context
3. `get_feat` takes context
4. **New losses**: `coarse_dyn`, `coarse_rep`, `sparse`
5. **Coarse rep losses per variant**: `coarse_barlow` / `coarse_infonce` / `coarse_recon`
6. Imagination starts from 3-tuple, `_imagine` returns coarse_feats
7. When `coarse_critic` enabled: mixed value target, coarse value loss, coarse replay value loss
8. Returns `(stoch, deter, context)` 3-tuple

### `_imagine(start, imag_horizon)` — Full override
```python
stoch, deter, context = start
for _ in range(imag_horizon):
    feat = self._frozen_rssm.get_feat(stoch, deter, context)
    action = self._frozen_actor(feat).rsample()
    feats.append(feat)
    actions.append(action)
    if self.use_coarse_critic:
        coarse_feats.append(self._frozen_rssm.get_coarse_feat(stoch, context))
    stoch, deter, context, _ = self._frozen_rssm.img_step(stoch, deter, context, action)
return feats, actions, coarse_feats_or_None
```

### `act(obs, state, eval)` — Override
Same logic but with context in state dict and 6-return obs_step.

### `get_initial_state(B)` — Override
Returns TensorDict with `stoch, deter, context, prev_action`.

### `update(replay_buffer)` — Override
Passes context to `replay_buffer.update(index, stoch, deter, context)`.

### `_update_slow_target()` — Override
Also updates `_slow_coarse_value` when coarse critic enabled.

### `clone_and_freeze()` — Override
Calls `super().clone_and_freeze()` then freezes coarse critic modules.

### `video_pred` — Override (or skip)
Handle 3-tuple initial. Only relevant for dreamer rep_loss.

---

## 5. Modify `configs/model/_base_.yaml`

```yaml
thick:
  enabled: False
  context: 32
  gate_noise_scale: 0.1
  coarse_layers: 1
  sparse_free: 0.0
  coarse_critic: False
  psi: 0.9

# Add to rssm section:
rssm:
  context: ${model.thick.context}
  gate_noise_scale: ${model.thick.gate_noise_scale}
  coarse_layers: ${model.thick.coarse_layers}
  sparse_free: ${model.thick.sparse_free}

# Add to loss_scales:
loss_scales:
  coarse_dyn: 1.0
  coarse_rep: 0.1
  sparse: 10.0
  coarse_barlow: 0.05
  coarse_infonce: 1.0
  coarse_recon: 1.0
  coarse_value: 1.0
  coarse_repval: 0.3
```

---

## 6. Modify `buffer.py` (minimal)

### `sample()` (~line 38):
```python
if "context" in sample_td.keys():
    initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0], sample_td["context"][:, 0])
else:
    initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
```

### `update()` (~line 44): Add optional context parameter
```python
def update(self, index, stoch, deter, context=None):
    # existing stoch/deter update ...
    if context is not None:
        context = context.reshape(-1, *context.shape[2:])
        self._buffer[index[1], index[0]].set_("context", context)
```

---

## 7. Modify `trainer.py` (minimal)

### `begin()` (~line 163): Store context in transitions
```python
trans["stoch"] = agent_state["stoch"]
trans["deter"] = agent_state["deter"]
if "context" in agent_state.keys():
    trans["context"] = agent_state["context"]
```

---

## 8. Modify `train.py` — Conditional agent class

```python
if config.thick.enabled:
    from thick import ThickDreamer
    agent = ThickDreamer(config, obs_space, act_space)
else:
    agent = Dreamer(config, obs_space, act_space)
```

---

## Files Summary

| File | Action | Scope |
|------|--------|-------|
| `gatelord.py` | **CREATE** | GateLord cell + sparse_loss |
| `crssm.py` | **CREATE** | CRSSM class extending RSSM |
| `thick.py` | **CREATE** | ThickDreamer class extending Dreamer |
| `rssm.py` | **MODIFY** | Add `context_dim` to Deter (backward-compatible) |
| `configs/model/_base_.yaml` | **MODIFY** | Add `thick` section + loss scales |
| `buffer.py` | **MODIFY** | Optional context (~5 lines) |
| `trainer.py` | **MODIFY** | Context in transitions (~2 lines) |
| `train.py` | **MODIFY** | Conditional ThickDreamer import (~4 lines) |
| `dreamer.py` | **NO CHANGES** | |
| `networks.py` | **NO CHANGES** | |

---

## Verification

1. **Smoke test without THICK** (`thick.enabled: False`): Run training, verify identical behavior — existing files unchanged
2. **Smoke test with THICK** (`thick.enabled: True`): Run `python train.py model.thick.enabled=True`, verify:
   - Context propagates through observe/imagine/act
   - Sparse, coarse_dyn, coarse_rep, coarse_barlow losses logged
   - feat_size includes context dimension
   - Actor/critic receive larger features
3. **Coarse critic** (`thick.coarse_critic: True`): Verify coarse_value loss logged, mixed target computed
4. **Buffer round-trip**: Context stored and recovered correctly
5. **Variant test**: Run with `rep_loss=dreamer` + `thick.enabled=True` to verify coarse decoder
