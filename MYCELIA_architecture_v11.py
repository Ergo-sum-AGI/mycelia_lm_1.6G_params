# ============================================
# MYCELIA LM Architecture (v10.7) — Fibonacci Geometric Resonance
#
# Changes from v10.3:
#   1. ELIMINATED: Log-space alphas, tanh clamps, sigmoid bounds
#   2. ADOPTED: Raw scalar alphas + Fibonacci geometric substrate
#   3. Fibonacci phase lattice: irrational golden-angle rotation per layer
#   4. Fibonacci word mask: self-similar sign-flip structure
#   5. Depth-dependent geometric capacity: substrate narrows with depth
#   6. LEGACY ALPHA CONVERSION REMOVED — 1.5B is de novo (raw_alpha only)
#   7. ZERO new learned parameters — geometry is pure buffer
#   8. SDPA (no T×T masks), complex Fibonacci rotation, in-place observables
#   9. Consensus residual fixed (no double-count of attn across rounds)
#
# Changes from v10.4 → v10.6 (Surgical Patch):
#   1. POST-MIX governor (ATTN path): governor moved after the residual mix
#      so gradients on raw_alpha_attn are no longer entangled with the
#      geometric substrate.
#   2. POST-MIX governor (FFN path): same treatment for the FFN branch.
#   3. Alpha regularization: explicit quadratic potential well on raw_alpha_*
#      pulling them toward 0 (i.e. alpha → 1.0). Default depth 1e-3.
#   4. Alpha gradient-norm telemetry: L2 norm of grad on raw_alpha_*,
#      returns 0.0 in eval mode, consumed by the meta-governor dashboard.
#   5/6. Compressor-level aggregation helpers expose the per-block loss /
#        grad-norms so the training loop can pick them up with one call.
#
# CORE PRINCIPLE:
#   The substrate (residual stream geometry) prevents runaway, not clamps.
#   Alphas are free scalars. The Fibonacci phase lattice ensures that
#   no two layers ever align in phase, preventing resonant amplification.
#   This is the 1D analog of the Penrose quasicrystal substrate from CQFT.
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List


# ═══════════════════════════════════════════════════════════════════════
# FIBONACCI GEOMETRIC CONSTANTS — Precomputed, cached forever
# ═══════════════════════════════════════════════════════════════════════

_PHI = (1.0 + math.sqrt(5.0)) / 2.0                 # ≈ 1.6180339887...
_PHI_INV = 1.0 / _PHI                                # ≈ 0.6180339887...
_GOLDEN_ANGLE = 2.0 * math.pi * _PHI_INV ** 2        # ≈ 137.507764° (radians)


def _fibonacci_word(n: int) -> str:
    """Generate nth Fibonacci word (binary string). S_0='0', S_1='01', S_2='010', ..."""
    a, b = "0", "01"
    if n == 0:
        return a
    if n == 1:
        return b
    for _ in range(n - 1):
        a, b = b, b + a
    return b


def _make_fibonacci_lattice(n_layers: int, d_model: int) -> torch.Tensor:
    """
    Generate Fibonacci phase lattice: (n_layers, d_model) matrix of angles.
    Each layer rotated by golden angle from previous.
    Each dimension within layer spaced by golden angle / d_model.
    """
    angles = torch.zeros(n_layers, d_model)
    for layer in range(n_layers):
        for dim in range(d_model):
            angles[layer, dim] = ((layer * _GOLDEN_ANGLE) +
                                   (dim * _GOLDEN_ANGLE / max(d_model, 1))) % (2.0 * math.pi)
    return angles


def _make_fibonacci_mask(d_model: int) -> torch.Tensor:
    """
    Fibonacci word mask: self-similar sign-flip pattern.
    '0' -> +1.0, '1' -> -1.0. Precomputed, no recursion at runtime.
    """
    # F_18 = 2584 > 2048, covers d_model=2048 without repetition
    word = _fibonacci_word(18)
    mask = torch.tensor([1.0 if c == '0' else -1.0 for c in word[:d_model]],
                        dtype=torch.float32)
    if len(mask) < d_model:
        # Pad by repeating if needed (shouldn't happen for d_model <= 610)
        repeats = (d_model // len(mask)) + 1
        mask = mask.repeat(repeats)[:d_model]
    return mask


def _make_depth_capacity(n_layers: int) -> torch.Tensor:
    """
    Geometric capacity per layer: C_l = 1 / (1 + (l/L) * (phi - 1))
    Deeper layers have tighter substrate (less capacity for field energy).
    """
    capacities = torch.zeros(n_layers)
    for layer in range(n_layers):
        progress = layer / max(n_layers - 1, 1)
        capacities[layer] = 1.0 / (1.0 + progress * (_PHI - 1.0))
    return capacities


@dataclass
class MyceliaConfig:
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 32
    vocab_size: int = 151643
    max_seq_len: int = 4096
    rope_base: float = 10000.0
    fib_weights: Tuple[int, ...] = (5, 8, 13, 21, 34, 55, 89, 144)
    dissenter_threshold: float = 2.5
    dubito_threshold: float = 7.0
    consensus_rounds: int = 2
    use_compression: bool = True
    compress_ratio: int = 8
    compress_window: int = 128
    compress_freq: int = 999999

    # ── v8.8.1: Governor targets ──────────────────────────────────────
    ffn_norm_target: float = 50.0
    alpha_norm_target: float = 100.0
    soft_cap: float = 400.0

    predictive_scale: bool = True

    expected_curvature: float = 0.5
    curvature_gain: float = 2.0
    coherence_weight: float = 1.0
    forecast_weight: float = 1.0
    instability_target: float = 0.45
    control_gain: float = 1.0

    # ── v8.8.1: control_factor_floor raised 0.4 → 0.7 ────────────────
    control_factor_floor: float = 0.7

    forecast_velocity_weight: float = 1.5
    forecast_accel_weight: float = 2.0

    # ── v8.8.1: Rate governor softened ───────────────────────────────
    use_rate_governor: bool = False
    ffn_growth_ratio_max: float = 2.0
    residual_growth_ratio_max: float = 1.5
    growth_gain: float = 2.0
    rate_governor_floor: float = 0.7

    # ── v8.8.1: Gradual transition config ────────────────────────────
    use_gradual_transition: bool = True
    transition_start_step: int = 0
    transition_duration: int = 10000
    ffn_target_end: float = 150.0
    alpha_target_end: float = 150.0

    # ── v8.8.1: Interaction guard config ────────────────────────────
    max_simultaneous_governors: int = 2
    governor_priority: Tuple[str, ...] = ('cap', 'ffn', 'alpha', 'mpc', 'rate')
    interaction_guard_hysteresis_steps: int = 500

    # ── v10.4: Fibonacci geometric resonance config ───────────────────
    use_fibonacci_governor: bool = True
    fibonacci_pair_rotation: bool = True  # If True, rotate pairs; if False, element-wise

    # ── v10.6: Alpha potential-well depth (quadratic pull on raw_alpha) ─
    # U = well_depth * (raw_alpha_attn^2 + raw_alpha_ffn^2)
    # Larger = stronger pull toward alpha=1.0 equilibrium.
    alpha_well_depth: float = 1e-3

    # ── v10.6: OOM fallback levers for 1.5B on T4 16GB ────────────
    # If OOM during early training, reduce max_seq_len to 2048 first
    # (TCM classical texts rarely need 4096-token context).
    # Second lever: drop n_layers from 24 to 20.
    # Both are hot-swappable without checkpoint invalidation.

    # ── v10.6: Gradient checkpointing for 1.5B scale ────────────────
    use_gradient_checkpointing: bool = True

def get_rotary_embedding(seq_len: int, d_head: int, device: torch.device, base: float = 10000.0):
    half_dim = d_head // 2
    theta = base ** (-torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim)
    positions = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    angles = positions * theta.unsqueeze(0)
    cos = angles.cos()
    sin = angles.sin()
    return cos, sin


def apply_rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, n_heads, T, d_head = x.shape
    half_dim = d_head // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    return torch.cat([rotated_x1, rotated_x2], dim=-1)


class GoldenDropout(nn.Module):
    def __init__(self):
        super().__init__()
        phi = (1 + torch.sqrt(torch.tensor(5.0))) / 2
        self.keep_prob = float(1.0 / phi)
        self.scale = float(phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = torch.rand_like(x) < self.keep_prob
            return x * mask.to(x.dtype) * self.scale
        return x


class MycelialAttention(nn.Module):
    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(0.1)
        self._rope_base = getattr(config, 'rope_base', 10000.0)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, return_heads: bool = True):
        B, T, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in qkv]

        cos, sin = get_rotary_embedding(T, self.d_head, x.device, base=self._rope_base)
        q = apply_rotary_embedding(q, cos, sin)
        k = apply_rotary_embedding(k, cos, sin)

        # ── v10.7: FlashAttention via SDPA ──
        if padding_mask is not None:
            # Causal mask (upper triangle = True)
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            # Padding mask: (B, T) where True = PAD. Broadcast to (B, 1, T, T)
            pad_mask = padding_mask[:, None, None, :].expand(B, 1, T, T)
            
            # Combine: True where we should mask (either causal or pad)
            combined_mask = causal_mask | pad_mask
            
            # Create additive mask (0.0 = attend, -inf = block)
            attn_mask = torch.zeros(B, 1, T, T, device=x.device, dtype=q.dtype)
            attn_mask.masked_fill_(combined_mask, float('-inf'))
            
            head_outputs = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,  # MUST be False when attn_mask is provided
            )
        else:
            head_outputs = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )

        out = head_outputs.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        if return_heads:
            return out, head_outputs
        return out, None


class MycelialConsensus(nn.Module):
    def __init__(self, config: MyceliaConfig, use_dynamic_threshold: bool = True):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.use_dynamic_threshold = use_dynamic_threshold
        self.base_threshold = config.dissenter_threshold
        fib = torch.tensor(config.fib_weights, dtype=torch.float32)
        # v10.6: Pad/truncate fib_weights to match n_heads (handles 1.5B config)
        if len(fib) < self.n_heads:
            repeats = (self.n_heads // len(fib)) + 1
            fib = fib.repeat(repeats)[:self.n_heads]
        elif len(fib) > self.n_heads:
            fib = fib[:self.n_heads]
        self.register_buffer('fib_weights', fib / fib.sum())
        self.register_buffer('_total', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_kept', torch.zeros(1, dtype=torch.long))
        self.cached_stats = {'total': 0, 'kept': 0, 'vetoed': 0}
        self._last_threshold = 0.0

    def reset_stats(self):
        self._total.zero_()
        self._kept.zero_()
        self.cached_stats = {'total': 0, 'kept': 0, 'vetoed': 0}

    def forward(self, head_outputs: torch.Tensor, step: int = 0, layer_idx: int = 0):
        B, n_heads, T, d_head = head_outputs.shape
        w = self.fib_weights.view(1, -1, 1, 1)
        consensus = (head_outputs * w).sum(dim=1, keepdim=True)
        mean_heads = head_outputs.mean(dim=1, keepdim=True)
        variance = (head_outputs - mean_heads).pow(2).mean(dim=1)
        token_variance = variance.mean(dim=-1, keepdim=True)
        flat_var = token_variance.view(-1)
        var_median = flat_var.median()
        var_mad = (flat_var - var_median).abs().median()
        var_scale = var_median + 1.4826 * var_mad

        if self.use_dynamic_threshold:
            layer_factor = 1.0 - (layer_idx / max(self.config.n_layers - 1, 1)) * 0.3
            threshold = 0.8 * var_scale * layer_factor
            threshold = threshold.clamp(min=0.05, max=10.0)
        else:
            threshold = self.base_threshold
        self._last_threshold = float(threshold.mean().item()) if threshold.numel() == 1 else float(threshold.item())

        acclamation_mask = (token_variance < threshold).float().unsqueeze(1)
        variance_veto = acclamation_mask + (1.0 - acclamation_mask) * 0.3
        consensus = consensus * variance_veto
        max_variance = token_variance.max()
        acclamation_rate = (token_variance < threshold).float().mean()
        coherence = acclamation_rate
        veto = (token_variance >= threshold).any()

        with torch.no_grad():
            self._total += acclamation_mask.numel()
            self._kept += acclamation_mask.sum().long()

        flat_var = token_variance.view(-1)
        self._telemetry_stats = {
            'safe_pct': (flat_var <= 2.5).float().mean() * 100,
            'dissenter_pct': ((flat_var > 2.5) & (flat_var <= 7.0)).float().mean() * 100,
            'dubito_pct': (flat_var > 7.0).float().mean() * 100,
        }

        lambda_disagree = token_variance / (threshold + 1e-6)
        lambda_disagree = lambda_disagree.clamp(min=0.0, max=10.0)
        instability_prediction = torch.sigmoid(lambda_disagree - 1.0)

        return consensus.squeeze(1), veto, {
            'coherence': coherence,
            'variance': max_variance,
            'threshold': threshold,
            'mask_kept_ratio': acclamation_mask.mean(),
            'instability_prediction': instability_prediction,
            'lambda_disagree': lambda_disagree,
        }

    def get_stats(self) -> dict:
        total = int(self._total.item())
        kept = int(self._kept.item())
        self.cached_stats = {'total': total, 'kept': kept, 'vetoed': total - kept}
        return self.cached_stats

    def print_stats(self):
        stats = self.get_stats()
        total = stats['total']
        if total == 0:
            print("No tokens processed yet.")
            return
        kept = stats['kept']
        vetoed = stats['vetoed']
        print("="*70)
        print("MYCELIA CONSENSUS TELEMETRY")
        print("="*70)
        print(f" Total elements: {total:,}")
        print(f" Kept (acclaimed): {kept:,} ({kept/total*100:.1f}%)")
        print(f" Vetoed (suppressed): {vetoed:,} ({vetoed/total*100:.1f}%)")
        print("="*70)


class FibonacciGeometricGovernor(nn.Module):
    """
    v10.4/v10.7: Fibonacci geometric resonance governor.
    ZERO learned parameters. Pure geometry.

    Shapes residual updates using:
    1. Fibonacci phase lattice: golden-angle rotation per layer/dimension
    2. Fibonacci word mask: self-similar sign-flip pattern
    3. Depth capacity: substrate tightens with depth

    Pair rotation is a complex multiply z ← z · e^{iθ} per (d, d+1) pair.
    Math runs in float32 (bf16 complex is poorly supported on GPU), then
    the result is cast back to the field dtype — numerically identical to
    a real 2×2 Givens rotation, with fewer intermediate tensors.
    """
    def __init__(self, d_model: int, layer_idx: int, n_layers: int,
                 use_pair_rotation: bool = True):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.use_pair_rotation = use_pair_rotation and (d_model % 2 == 0)

        fib_lattice = _make_fibonacci_lattice(n_layers, d_model)
        fib_mask = _make_fibonacci_mask(d_model)
        depth_cap = _make_depth_capacity(n_layers)

        self.register_buffer('fib_phase', fib_lattice[layer_idx])     # (d_model,)
        self.register_buffer('fib_mask', fib_mask)                    # (d_model,)
        self.register_buffer('depth_capacity', depth_cap[layer_idx])  # scalar

        # Element-wise fallback (also used if pair rotation disabled)
        self.register_buffer('fib_cos', torch.cos(self.fib_phase))
        self.register_buffer('fib_sin', torch.sin(self.fib_phase))

        # Pair rotation: e^{iθ} per pair as complex64 — zero runtime trig
        if self.use_pair_rotation:
            ang = self.fib_phase[::2].float()                         # (D/2,)
            self.register_buffer(
                'rot_c',
                torch.complex(ang.cos(), ang.sin()),                  # complex64 (D/2,)
            )
        else:
            # Empty buffer keeps state_dict shape stable across configs
            self.register_buffer('rot_c', torch.empty(0, dtype=torch.complex64))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        field:  (B, T, d_model)
        returns (B, T, d_model) geometrically shaped field
        """
        dtype = field.dtype

        # 1. Fibonacci word mask
        masked = field * self.fib_mask.to(dtype=dtype)

        # 2. Golden-angle phase rotation
        if self.use_pair_rotation and self.rot_c.numel() > 0:
            # view_as_complex requires a trailing size-2 real dimension and
            # prefers float32; multiply, then cast back to field dtype.
            half = self.d_model // 2
            pairs = masked.reshape(*masked.shape[:-1], half, 2).float().contiguous()
            z = torch.view_as_complex(pairs)              # (B, T, D/2) complex64
            z = z * self.rot_c                            # broadcast over B, T
            rotated = torch.view_as_real(z).reshape_as(field).to(dtype=dtype)
        else:
            cos = self.fib_cos.to(dtype=dtype)
            sin = self.fib_sin.to(dtype=dtype)
            rotated = masked * cos - torch.roll(masked, shifts=1, dims=-1) * sin

        # 3. Depth-dependent capacity
        return rotated * self.depth_capacity.to(dtype=dtype)


class MycelialBlock(nn.Module):
    def __init__(self, config: MyceliaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.attn = MycelialAttention(config)
        self.mycelia = MycelialConsensus(config)
        self.dropout = nn.Dropout(0.1)
        d_ff = int(config.d_model * 4 * 2 / 3)
        self.gate = nn.Linear(config.d_model, d_ff * 2, bias=False)
        self.proj = nn.Linear(d_ff, config.d_model, bias=False)

        # ── v10.5: Centered raw alphas — potential well at α=1.0 ────────
        # raw = 0  →  α = 1.0 (stable equilibrium, bottom of potential well)
        # Hook A quadratic well pulls raw toward 0 → α toward 1.0
        # No clamps, no bounds, no legacy conversion — 1.5B is de novo.
        self.raw_alpha_attn = nn.Parameter(torch.zeros(1))
        self.raw_alpha_ffn = nn.Parameter(torch.zeros(1))

        # ── v10.4: Fibonacci geometric governor ──────────────────────────
        # ZERO parameters — pure geometric substrate
        if config.use_fibonacci_governor:
            self.governor = FibonacciGeometricGovernor(
                config.d_model, layer_idx, config.n_layers,
                use_pair_rotation=config.fibonacci_pair_rotation
            )
        else:
            self.governor = None

        self._hidden_state = None
        self.layer_idx = layer_idx
        self.consensus_rounds = config.consensus_rounds
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.soft_cap = config.soft_cap

        # v8.8.1 governor config (preserved)
        self.ffn_norm_target = config.ffn_norm_target
        self.alpha_norm_target = config.alpha_norm_target
        self.predictive_scale = config.predictive_scale
        self.expected_curvature = config.expected_curvature
        self.curvature_gain = config.curvature_gain
        self.coherence_weight = config.coherence_weight
        self.forecast_weight = config.forecast_weight
        self.instability_target = config.instability_target
        self.control_gain = config.control_gain
        self.control_factor_floor = config.control_factor_floor
        self.forecast_velocity_weight = config.forecast_velocity_weight
        self.forecast_accel_weight = config.forecast_accel_weight

        self.use_rate_governor = config.use_rate_governor
        self.ffn_growth_ratio_max = config.ffn_growth_ratio_max
        self.residual_growth_ratio_max = config.residual_growth_ratio_max
        self.growth_gain = config.growth_gain
        self.rate_governor_floor = config.rate_governor_floor

        self._governor_hysteresis: Dict[str, int] = {}

        # Buffers for geometric observables and MPC
        self.register_buffer('instability_forecast', torch.zeros(1))
        self.register_buffer('forecast_confidence', torch.ones(1))
        self.register_buffer('instability_velocity', torch.zeros(1))
        self.register_buffer('instability_acceleration', torch.zeros(1))
        self.register_buffer('_predicted_instability', torch.zeros(1))
        self.register_buffer('_actual_intervention', torch.zeros(1))
        self.register_buffer('_forecast_error', torch.zeros(1))
        # Geometric observable state — allocated lazily with zeros_like (dtype-safe)
        self.register_buffer('_prev_hidden', torch.zeros(1, 1, config.d_model))
        self.register_buffer('_prev_delta', torch.zeros(1, 1, config.d_model))
        self.register_buffer('_prev_curvature', torch.zeros(()))  # scalar buffer

        self.register_buffer('_rate_governor_hits', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_rate_governor_total', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_ffn_veto_hits', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_ffn_total', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_soft_cap_hits', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_soft_cap_total', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_alpha_scale_hits', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_alpha_total', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_mpc_interventions', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_mpc_total', torch.zeros(1, dtype=torch.long))

    def _compute_geometric_observables(self, x: torch.Tensor):
        """In-place buffer updates — no per-step reallocation, dtype-matched."""
        B, T, D = x.shape
        if (self._prev_hidden.shape[0] != B or self._prev_hidden.shape[1] != T
                or self._prev_hidden.dtype != x.dtype
                or self._prev_hidden.device != x.device):
            self._prev_hidden = torch.zeros_like(x)
            self._prev_delta = torch.zeros_like(x)
            self._prev_curvature = torch.zeros((), device=x.device, dtype=x.dtype)

        velocity = x - self._prev_hidden
        velocity_norm = velocity.norm(p=2, dim=-1)
        acceleration = velocity - self._prev_delta
        acceleration_norm = acceleration.norm(p=2, dim=-1)
        curvature = (acceleration_norm / (velocity_norm.pow(2) + 1e-6)).clamp_(max=10.0)
        jerk = (curvature - self._prev_curvature).abs()

        # In-place updates — no orphaned buffers
        self._prev_hidden.copy_(x.detach())
        self._prev_delta.copy_(velocity.detach())
        self._prev_curvature.copy_(curvature.detach().mean())
        return velocity_norm, acceleration_norm, curvature, jerk

    def _rate_governor(self, ffn_out: torch.Tensor, ffn_in: torch.Tensor,
                       x_out: torch.Tensor, x_in: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        diag = {
            'ffn_growth_ratio': 1.0,
            'residual_growth_ratio': 1.0,
            'rate_scale': 1.0,
            'rate_governor_hit': 0.0,
        }
        if not self.use_rate_governor:
            return torch.ones(ffn_out.shape[:-1] + (1,), device=ffn_out.device), diag

        ffn_in_norm = ffn_in.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ffn_out_norm = ffn_out.norm(dim=-1, keepdim=True)
        ffn_growth = (ffn_out_norm / ffn_in_norm).clamp(min=1.0)
        diag['ffn_growth_ratio'] = float(ffn_growth.mean().item())

        x_in_norm = x_in.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_out_norm = x_out.norm(dim=-1, keepdim=True)
        residual_growth = (x_out_norm / x_in_norm).clamp(min=1.0)
        diag['residual_growth_ratio'] = float(residual_growth.mean().item())

        ffn_excess = F.relu(ffn_growth - self.ffn_growth_ratio_max)
        res_excess = F.relu(residual_growth - self.residual_growth_ratio_max)
        total_excess = self.growth_gain * (ffn_excess + res_excess)
        scale = torch.exp(-total_excess).clamp(min=self.rate_governor_floor, max=1.0)

        diag['rate_scale'] = float(scale.mean().item())
        diag['rate_governor_hit'] = float((ffn_growth > self.ffn_growth_ratio_max).float().mean().item())

        with torch.no_grad():
            self._rate_governor_hits += (ffn_growth > self.ffn_growth_ratio_max).long().sum()
            self._rate_governor_total += ffn_growth.numel()

        return scale, diag

    def _recursive_instability_forecast(self, instability_prediction, coherence, curvature,
                                        prev_forecast, prev_confidence):
        B, T = curvature.shape
        device = curvature.device

        if instability_prediction.dim() == 3:
            instability_prediction = instability_prediction.squeeze(-1)
        if instability_prediction.dim() == 1:
            instability_prediction = instability_prediction.unsqueeze(0).expand(B, T)
        instability_prediction = instability_prediction.view(B, T)

        if isinstance(coherence, torch.Tensor):
            if coherence.dim() == 0:
                coherence = coherence.view(1, 1).expand(B, T)
            elif coherence.numel() == 1:
                coherence = coherence.view(1, 1).expand(B, T)
            else:
                coherence = coherence.view(B, T)
        else:
            coherence = torch.full((B, T), float(coherence), device=device)

        P_current = instability_prediction

        if prev_forecast is not None:
            P_persist = prev_forecast.view(1, 1).expand(B, T)
        else:
            P_persist = torch.zeros(B, T, device=device)

        if prev_confidence is not None:
            c_persist = prev_confidence.view(1, 1).expand(B, T)
        else:
            c_persist = torch.ones(B, T, device=device)

        base_confidence = torch.sigmoid(
            self.coherence_weight * coherence +
            self.forecast_weight * c_persist
        )
        delta_curvature = F.relu(curvature - self.expected_curvature)
        curvature_damping = torch.exp(-self.curvature_gain * delta_curvature)
        confidence = base_confidence * curvature_damping

        I = confidence * P_current + (1.0 - confidence) * P_persist
        I = I.clamp(min=0.0, max=1.0)

        prev_I = self.instability_forecast.detach().squeeze()
        prev_v = self.instability_velocity.detach().squeeze()
        current_I_mean = I.mean().detach()
        v_I = current_I_mean - prev_I
        a_I = v_I - prev_v

        self.instability_forecast = torch.tensor([current_I_mean], device=device)
        self.instability_velocity = torch.tensor([v_I], device=device)
        self.instability_acceleration = torch.tensor([a_I], device=device)

        I_raw = I
        I_combined = I_raw + (
            self.forecast_velocity_weight * F.relu(torch.tensor(v_I, device=device)) +
            self.forecast_accel_weight * F.relu(torch.tensor(a_I, device=device))
        )
        I_combined = I_combined.clamp(max=2.0)

        predicted_from_prev = self._predicted_instability.detach().squeeze()
        actual_intervention = (I_combined > self.instability_target).float().mean().detach()
        forecast_error = abs(predicted_from_prev - actual_intervention)

        self._predicted_instability = I_raw.mean().detach().reshape(1).to(device)
        self._actual_intervention = torch.tensor([actual_intervention], device=device)
        self._forecast_error = torch.tensor([forecast_error], device=device)

        next_forecast = torch.tensor([current_I_mean], device=device)
        next_confidence = confidence.mean().detach().reshape(1).to(device)

        diagnostics = {
            'confidence_mean': float(confidence.mean().item()),
            'confidence_min': float(confidence.min().item()),
            'curvature_damping_mean': float(curvature_damping.mean().item()),
            'delta_curvature_mean': float(delta_curvature.mean().item()),
            'instability_prediction': float(P_current.mean().item()),
            'instability_persistence': float(P_persist.mean().item()),
            'instability_posterior': float(I.mean().item()),
            'instability_velocity': float(v_I),
            'instability_acceleration': float(a_I),
            'instability_combined': float(I_combined.mean().item()),
            'forecast_error': float(forecast_error),
            'predicted_from_prev': float(predicted_from_prev),
            'actual_intervention': float(actual_intervention),
        }

        return I_combined, next_forecast, next_confidence, torch.tensor(forecast_error, device=device), diagnostics

    def _control_policy(self, I_combined, confidence, alpha_native):
        prediction = I_combined.clamp(max=1.0)
        control_signal = confidence * prediction
        control_factor = torch.exp(-self.control_gain * control_signal)
        control_factor = control_factor.clamp(min=self.control_factor_floor, max=1.0)
        return (alpha_native * control_factor).unsqueeze(-1)

    def _apply_interaction_guard(self, scales: Dict[str, torch.Tensor],
                                  step: int) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        max_govs = self.config.max_simultaneous_governors
        hysteresis = self.config.interaction_guard_hysteresis_steps
        priority = {name: i for i, name in enumerate(self.config.governor_priority)}

        active = []
        for name, scale in scales.items():
            mean_scale = float(scale.mean().item())
            if mean_scale < 0.99:
                if name not in self._governor_hysteresis:
                    self._governor_hysteresis[name] = 0
                self._governor_hysteresis[name] += 1
                active.append((name, mean_scale, self._governor_hysteresis[name]))
            else:
                if name in self._governor_hysteresis:
                    self._governor_hysteresis[name] = 0

        disabled: List[str] = []
        if len(active) <= max_govs:
            return scales, disabled

        active.sort(key=lambda x: priority.get(x[0], 99))

        for name, scale_val, hyst_count in active[max_govs:]:
            if hyst_count >= hysteresis:
                disabled.append(name)
                scales[name] = torch.ones_like(scales[name])
                self._governor_hysteresis[name] = 0

        return scales, disabled

    # ── v10.4: Forward pass with Fibonacci geometric resonance ──────────
    def forward(self, x: torch.Tensor, step: int = 0,
                padding_mask: Optional[torch.Tensor] = None,
                prev_forecast: Optional[torch.Tensor] = None,
                prev_confidence: Optional[torch.Tensor] = None):
        B, T, D = x.shape
        assert T <= 4096, f"Sequence length {T} exceeds max 4096"
        assert D == self.norm1.normalized_shape[0], f"Feature dim mismatch: got {D}"

        residual = x
        x_input_for_rate = x
        final_round_info = {}

        # ── v8.8.1: Compute current targets via smoothstep transition ────
        if self.config.use_gradual_transition:
            progress = min(1.0, (step - self.config.transition_start_step) / self.config.transition_duration)
            ease = 0.5 - 0.5 * math.cos(math.pi * progress)
            current_ffn_target = self.config.ffn_norm_target + ease * (self.config.ffn_target_end - self.config.ffn_norm_target)
            current_alpha_target = self.config.alpha_norm_target + ease * (self.config.alpha_target_end - self.config.alpha_norm_target)
            use_rate = progress > 0.5
        else:
            current_ffn_target = self.ffn_norm_target
            current_alpha_target = self.alpha_norm_target
            use_rate = self.use_rate_governor

        velocity_norm, acceleration_norm, curvature, jerk = self._compute_geometric_observables(x)

        # v10.5: Centered raw alphas — potential well at α=1.0
        # α = 1 + raw. Hook A well pulls raw toward 0 → α toward 1.0
        alpha_attn = 1.0 + self.raw_alpha_attn
        alpha_ffn = 1.0 + self.raw_alpha_ffn

        # Consensus rounds refine attention against the ORIGINAL residual.
        # Consensus rounds iteratively refine the hidden state.
        x = residual  # Start from the original block input
        for round_idx in range(self.consensus_rounds):
            attn_out, head_outputs = self.attn(
                self.norm1(x),  # Attend to the progressively refined state
                padding_mask=padding_mask,
                return_heads=True,
            )
            consensus, veto, info = self.mycelia(head_outputs, step=step, layer_idx=self.layer_idx)
            final_round_info = info
            
            consensus_expanded = (
                consensus
                .unsqueeze(2)
                .expand(B, T, self.n_heads, self.d_head)
                .reshape(B, T, -1)
            )
            mix_ratio = 0.9 - (round_idx * 0.05)
            mix_ratio = max(0.5, mix_ratio)
            attn_out = mix_ratio * attn_out + (1.0 - mix_ratio) * consensus_expanded
            
            # Accumulate onto the current state (Pre-LN style)
            x = x + alpha_attn * attn_out
            if self.governor is not None:
                x = self.governor(x)
            x = self.dropout(x)
            # residual stays the original block input for every round

        g, h = self.gate(self.norm2(x)).chunk(2, dim=-1)
        ffn_in_for_rate = x
        ffn_out = self.proj(F.silu(g) * h)

        # ── v10.6: Governor moved to POST-mix for clean alpha gradients ──
        # Pre-mix shaping would entangle the geometric substrate with the
        # alpha multiplier, polluting its gradient. Pass raw FFN through to
        # the residual-mix; governor is applied to the combined x below.
        ffn_shaped = ffn_out

        # ── v8.8.1: FFN VETO with current target ─────────────────────────
        ffn_norms = torch.norm(ffn_shaped, p=2, dim=-1, keepdim=True)
        ffn_veto = torch.clamp(current_ffn_target / (ffn_norms + 1e-6), max=1.0)
        ffn_shaped = ffn_shaped * ffn_veto
        ffn_veto_factor = ffn_veto.mean().item()
        ffn_work = 1.0 - ffn_veto_factor

        with torch.no_grad():
            ffn_veto_hits = (ffn_norms > current_ffn_target).float()
            self._ffn_veto_hits += ffn_veto_hits.sum().long()
            self._ffn_total += ffn_veto_hits.numel()

        if self.predictive_scale:
            instability_prediction = info.get('instability_prediction',
                                             torch.zeros(B, T, device=x.device))
            coherence = info.get('coherence', torch.tensor(0.5, device=x.device))

            I_combined, next_forecast, next_confidence, forecast_error, mpc_diag = self._recursive_instability_forecast(
                instability_prediction, coherence, curvature, prev_forecast, prev_confidence)

            if isinstance(coherence, torch.Tensor):
                if coherence.dim() == 0:
                    coherence_bt = coherence.view(1, 1).expand(B, T)
                elif coherence.numel() == 1:
                    coherence_bt = coherence.view(1, 1).expand(B, T)
                else:
                    coherence_bt = coherence.view(B, T)
            else:
                coherence_bt = torch.full((B, T), float(coherence), device=x.device)

            if prev_confidence is not None:
                prev_conf_bt = prev_confidence.view(1, 1).expand(B, T)
            else:
                prev_conf_bt = torch.ones(B, T, device=x.device)

            confidence_for_control = torch.sigmoid(
                self.coherence_weight * coherence_bt +
                self.forecast_weight * prev_conf_bt
            ) * torch.exp(-self.curvature_gain * F.relu(curvature - self.expected_curvature))

            effective_alpha_attn = self._control_policy(I_combined, confidence_for_control, alpha_attn)
            effective_alpha_ffn = self._control_policy(I_combined, confidence_for_control, alpha_ffn)

            assert effective_alpha_attn.shape == (B, T, 1)
            assert effective_alpha_ffn.shape == (B, T, 1)
        else:
            effective_alpha_attn = alpha_attn.view(1, 1, 1).expand(B, T, 1)
            effective_alpha_ffn = alpha_ffn.view(1, 1, 1).expand(B, T, 1)
            mpc_diag = {}
            I_combined = torch.zeros(B, T, device=x.device)
            next_forecast = torch.zeros(1, device=x.device)
            next_confidence = torch.ones(1, device=x.device)
            forecast_error = torch.zeros(1, device=x.device)

        # ── v8.8.1: ALPHA SCALE with current target ────────────────────
        attn_norms = torch.norm(attn_out, p=2, dim=-1, keepdim=True)
        ffn_norms_post = torch.norm(ffn_shaped, p=2, dim=-1, keepdim=True)
        contrib_norm = torch.sqrt(
            (effective_alpha_attn * attn_norms).pow(2) +
            (effective_alpha_ffn * ffn_norms_post).pow(2) + 1e-6
        )
        alpha_scale = torch.clamp(current_alpha_target / (contrib_norm + 1e-6), max=1.0)
        effective_alpha_attn = effective_alpha_attn * alpha_scale
        effective_alpha_ffn = effective_alpha_ffn * alpha_scale
        alpha_scale_factor = alpha_scale.mean().item()
        alpha_work = 1.0 - alpha_scale_factor

        with torch.no_grad():
            alpha_hits = (contrib_norm > current_alpha_target).float()
            self._alpha_scale_hits += alpha_hits.sum().long()
            self._alpha_total += alpha_hits.numel()

        x = residual + effective_alpha_attn * attn_out + effective_alpha_ffn * ffn_shaped
        # ── v10.6: Post-mix governor on the combined residual update ────
        if self.governor is not None:
            x = self.governor(x)

        # ── v8.8.1: RATE GOVERNOR ──────────────────────────────────────
        rate_scale, rate_diag = self._rate_governor(
            ffn_out=ffn_shaped, ffn_in=ffn_in_for_rate,
            x_out=x, x_in=x_input_for_rate,
        )
        if use_rate:
            x = x * rate_scale

        # ── SOFT NORM CAP ──────────────────────────────────────────────
        token_norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        excess = F.softplus(token_norms - self.soft_cap)
        soft_scale = 1.0 + excess / (token_norms + 1e-6)
        x = x / soft_scale
        soft_cap_factor = 1.0 / soft_scale.mean().item()
        cap_work = 1.0 - soft_cap_factor

        with torch.no_grad():
            cap_engaged = (token_norms > self.soft_cap).float()
            self._soft_cap_hits += cap_engaged.sum().long()
            self._soft_cap_total += cap_engaged.numel()

        # ── v8.8.1: GOVERNOR INTERACTION GUARD ──────────────────────────
        scales = {
            'ffn': ffn_veto,
            'alpha': alpha_scale,
            'cap': soft_scale,
            'rate': rate_scale if use_rate else torch.ones_like(rate_scale),
            'mpc': torch.exp(-self.control_gain * (confidence_for_control * I_combined.clamp(max=1.0))).unsqueeze(-1) if I_combined.numel() > 0 else torch.ones(B, T, 1, device=x.device),
        }
        scales, disabled_governors = self._apply_interaction_guard(scales, step)

        if 'ffn' in disabled_governors:
            ffn_shaped = ffn_shaped / (ffn_veto + 1e-8)
            ffn_work = 0.0
        if 'alpha' in disabled_governors:
            alpha_work = 0.0
        if 'cap' in disabled_governors:
            cap_work = 0.0

        with torch.no_grad():
            self.instability_forecast = next_forecast
            self.forecast_confidence = next_confidence

        # ── TELEMETRY ────────────────────────────────────────────────────
        final_round_info['ffn_veto_ratio'] = float(ffn_veto_hits.mean().item())
        final_round_info['mean_ffn_norm'] = float(ffn_norms.mean().item())
        final_round_info['max_ffn_norm'] = float(ffn_norms.max().item())
        final_round_info['ffn_veto_factor'] = float(ffn_veto_factor)
        final_round_info['ffn_work'] = float(ffn_work)
        final_round_info['ffn_target_current'] = float(current_ffn_target)

        final_round_info['alpha_scale_ratio'] = float(alpha_hits.mean().item())
        final_round_info['mean_alpha_scale'] = float(alpha_scale.mean().item())
        final_round_info['mean_contrib_norm'] = float(contrib_norm.mean().item())
        final_round_info['alpha_scale_factor'] = float(alpha_scale_factor)
        final_round_info['alpha_work'] = float(alpha_work)
        final_round_info['alpha_target_current'] = float(current_alpha_target)

        # ── v10.4: Geometric telemetry ───────────────────────────────────
        if self.governor is not None:
            final_round_info['geometric_capacity'] = float(self.governor.depth_capacity.item())
            final_round_info['geometric_phase_mean'] = float(self.governor.fib_phase.mean().item())
            final_round_info['geometric_shell'] = self.layer_idx + 1
        else:
            final_round_info['geometric_capacity'] = 1.0
            final_round_info['geometric_phase_mean'] = 0.0
            final_round_info['geometric_shell'] = 0

        final_round_info['alpha_attn_raw'] = float((1.0 + self.raw_alpha_attn).item())
        final_round_info['alpha_ffn_raw'] = float((1.0 + self.raw_alpha_ffn).item())

        final_round_info['mpc_intervention_ratio'] = float(
            (I_combined > self.instability_target).float().mean().item()
        ) if I_combined.numel() > 0 else 0.0
        final_round_info['mean_control_factor'] = float(
            torch.exp(-self.control_gain * (confidence_for_control * I_combined.clamp(max=1.0))).mean().item()
        ) if I_combined.numel() > 0 else 1.0
        final_round_info['mean_instability_field'] = float(I_combined.mean().item())
        final_round_info['instability_velocity'] = mpc_diag.get('instability_velocity', 0.0)
        final_round_info['instability_acceleration'] = mpc_diag.get('instability_acceleration', 0.0)
        mpc_control_factor = final_round_info['mean_control_factor']
        mpc_work = 1.0 - mpc_control_factor
        final_round_info['mpc_work'] = float(mpc_work)
        final_round_info['mpc_control_factor'] = float(mpc_control_factor)

        final_round_info['mean_prediction'] = mpc_diag.get('instability_prediction', 0.0)
        final_round_info['mean_confidence'] = mpc_diag.get('confidence_mean', 1.0)
        final_round_info['confidence_min'] = mpc_diag.get('confidence_min', 1.0)
        final_round_info['curvature_damping'] = mpc_diag.get('curvature_damping_mean', 1.0)
        final_round_info['delta_curvature'] = mpc_diag.get('delta_curvature_mean', 0.0)

        final_round_info['forecast_error'] = mpc_diag.get('forecast_error', 0.0)
        final_round_info['predicted_from_prev'] = mpc_diag.get('predicted_from_prev', 0.0)
        final_round_info['actual_intervention'] = mpc_diag.get('actual_intervention', 0.0)

        final_round_info['mean_velocity'] = float(velocity_norm.mean().item())
        final_round_info['mean_acceleration'] = float(acceleration_norm.mean().item())
        final_round_info['mean_curvature'] = float(curvature.mean().item())
        final_round_info['max_curvature'] = float(curvature.max().item())
        final_round_info['mean_jerk'] = float(jerk.mean().item())
        final_round_info['soft_cap_hit_ratio'] = float(cap_engaged.mean().item())
        final_round_info['mean_soft_scale'] = float(soft_scale.mean().item())
        final_round_info['max_raw_norm'] = float(token_norms.max().item())
        final_round_info['mean_raw_norm'] = float(token_norms.mean().item())
        final_round_info['soft_cap_factor'] = float(soft_cap_factor)
        final_round_info['cap_work'] = float(cap_work)

        final_round_info['rate_governor_hit'] = rate_diag['rate_governor_hit']
        final_round_info['rate_scale_mean'] = rate_diag['rate_scale']
        final_round_info['ffn_growth_ratio'] = rate_diag['ffn_growth_ratio']
        final_round_info['residual_growth_ratio'] = rate_diag['residual_growth_ratio']
        final_round_info['rate_governor_enabled'] = float(use_rate)

        final_round_info['transition_progress'] = float(progress) if self.config.use_gradual_transition else 1.0
        final_round_info['transition_ease'] = float(ease) if self.config.use_gradual_transition else 1.0
        final_round_info['disabled_governors'] = disabled_governors

        layer_pressure = (
            ffn_work * ffn_norms.mean().item() +
            alpha_work * contrib_norm.mean().item() +
            cap_work * token_norms.mean().item() +
            mpc_work * I_combined.mean().item()
        )
        final_round_info['layer_pressure'] = float(layer_pressure)

        x = self.dropout(x)
        self._hidden_state = x.detach()
        self._last_info = final_round_info
        return x, final_round_info

    # ── v10.6: Alpha potential-well regularization ──────────────────────
    def alpha_regularization(self):
        """
        v10.6: Explicit potential well in the loss landscape.
        Creates direct, unambiguous gradient on raw_alpha_*.

        U = well_depth * (raw_alpha_attn^2 + raw_alpha_ffn^2)
        raw_alpha=0  =>  alpha=1.0 (equilibrium / well bottom)

        well_depth is controlled by the meta-governor (default 1e-3).
        """
        well_depth = getattr(self.config, 'alpha_well_depth', 1e-3)
        return (self.raw_alpha_attn ** 2 + self.raw_alpha_ffn ** 2) * well_depth

    # ── v10.6: Alpha gradient-norm telemetry ──────────────────────────
    def alpha_gradient_norm(self):
        """
        v10.6: Telemetry sensor for the meta-governor.
        Returns L2 norm of gradients on raw_alpha_*.
        If grad is None (e.g. eval mode), returns 0.0.
        """
        grad_norm_sq = 0.0
        if self.raw_alpha_attn.grad is not None:
            grad_norm_sq += self.raw_alpha_attn.grad.norm().item() ** 2
        if self.raw_alpha_ffn.grad is not None:
            grad_norm_sq += self.raw_alpha_ffn.grad.norm().item() ** 2
        return grad_norm_sq ** 0.5


class MycelialCompressor(nn.Module):
    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.config = config
        self.window = config.compress_window
        self.ratio = config.compress_ratio
        self.latent_dim = config.d_model
        self.encoder_blocks = nn.ModuleList([MycelialBlock(config, i) for i in range(2)])
        self.latent_proj = nn.Linear(config.d_model, config.d_model)
        self.input_pos = nn.Parameter(torch.randn(1, config.compress_window, config.d_model) * 0.02)
        self.latent_pos = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, D = x.shape
        assert W == self.window, f"Expected window {self.window}, got {W}"
        x = x + self.input_pos
        h = x
        for block in self.encoder_blocks:
            h, _ = block(h)
        h = h.view(B, W // self.ratio, self.ratio, D)
        latent = h.mean(dim=2)
        latent = self.latent_proj(latent)
        seq_len = latent.shape[1]
        latent = latent + self.latent_pos[:, :seq_len, :]
        return latent

    # ── v10.6: Training-loop hooks (aggregated across encoder blocks) ───
    def alpha_regularization_loss(self):
        """
        v10.6: Sum of alpha potential-well losses across compressor encoder
        blocks. Call this from the training step:

            alpha_loss = model.compressor.alpha_regularization_loss()
            total_loss = ce_loss + alpha_loss
        """
        return sum(block.alpha_regularization() for block in self.encoder_blocks)

    def alpha_gradient_norms(self):
        """
        v10.6: Per-block L2 norm of the alpha gradient. Call this AFTER
        loss.backward() and BEFORE optimizer.step():

            norms = model.compressor.alpha_gradient_norms()
            logs["alpha_grad_norm/mean"] = float(sum(norms) / len(norms))
        """
        return [block.alpha_gradient_norm() for block in self.encoder_blocks]


class DubitoMonitor(nn.Module):
    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.config = config

    def forward(self, hidden_states: torch.Tensor, depth: int) -> float:
        if hidden_states is None or hidden_states.shape[0] < 5:
            return 0.0
        eps = 1e-8
        h_norm = hidden_states / (hidden_states.norm(dim=-1, keepdim=True) + eps)
        v = h_norm[1:] - h_norm[:-1]
        v_unit = v / (v.norm(dim=-1, keepdim=True) + eps)
        persistence = (v_unit[1:] * v_unit[:-1]).sum(dim=-1)
        paradox_ratio = 1 - abs(persistence.mean().item())
        dubito = paradox_ratio * (1 + math.log(depth + 1))
        return max(0.0, min(15.0, dubito))


class FibonacciGuardrails(nn.Module):
    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.config = config

    def should_continue(self, depth: int, dubito: float):
        if depth <= 5:
            ring = 0
        elif depth <= 8:
            ring = 1
        elif depth <= 13:
            ring = 2
        else:
            ring = 3
        if dubito > self.config.dubito_threshold and ring >= 2:
            return False, f"Stop: Dubito={dubito:.2f}"
        if depth > [5, 8, 13, 21][ring]:
            return False, f"Stop: Depth {depth} exceeds ring {ring} limit"
        return True, "Continue"


class MyceliaLM(nn.Module):
    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.compressor = MycelialCompressor(config)
        self.blocks = nn.ModuleList([MycelialBlock(config, i) for i in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model, eps=1e-6)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # v10.6: tied weights
        self.guardrails = FibonacciGuardrails(config)
        self.dubito_monitor = DubitoMonitor(config)
        self.depth = 0
        self.consensus_stats = []
        self.dubito_history = []
        self.register_buffer("cumulative_saved_bytes", torch.tensor(0, dtype=torch.int64))
        self._last_info = {}
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    # ── v10.7: Global alpha regularization + gradient telemetry ─────────
    def alpha_regularization_loss(self):
        """Sum of alpha potential-well losses across ALL blocks + compressor."""
        main_loss = sum(block.alpha_regularization() for block in self.blocks)
        comp_loss = self.compressor.alpha_regularization_loss() if hasattr(self, 'compressor') else 0.0
        return main_loss + comp_loss

    def alpha_gradient_norms(self):
        """L2 norm of alpha gradients across ALL blocks + compressor."""
        norms = [block.alpha_gradient_norm() for block in self.blocks]
        if hasattr(self, 'compressor'):
            norms.extend(self.compressor.alpha_gradient_norms())
        return norms                

    def forward(self, input_ids: torch.Tensor, use_compression: bool = False,
                log_during_train: bool = False, padding_mask: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        input_ids = torch.clamp(input_ids, 0, self.config.vocab_size - 1)
        x = self.embedding(input_ids)

        bytes_per_element = 2
        uncompressed_bytes = B * T * self.config.d_model * bytes_per_element
        compression_applied = False
        vram_saved_mb = 0.0

        if use_compression and T > self.config.compress_window:
            prefix_len = self.config.compress_window
            prefix = x[:, :prefix_len, :]
            suffix = x[:, prefix_len:, :]
            latent = self.compressor(prefix)
            x = torch.cat([latent, suffix], dim=1)
            compression_applied = True
            compressed_len = self.config.compress_window // self.config.compress_ratio
            compressed_bytes = (
                B * compressed_len * self.config.d_model * bytes_per_element
                + B * (T - prefix_len) * self.config.d_model * bytes_per_element
            )
            step_saved_bytes = uncompressed_bytes - compressed_bytes
            vram_saved_mb = step_saved_bytes / (1024 ** 2)
            self.cumulative_saved_bytes += step_saved_bytes
            if padding_mask is not None:
                compressed_pad = padding_mask[:, :prefix_len].any(dim=1, keepdim=True)
                compressed_pad = compressed_pad.expand(B, compressed_len)
                suffix_pad = padding_mask[:, prefix_len:]
                padding_mask = torch.cat([compressed_pad, suffix_pad], dim=1)

        all_layer_coherence = []
        layer_variances = []
        max_variance_tracked = 0.0
        last_info = {}
        prev_forecast = None
        prev_confidence = None
        instability_field_history = []
        confidence_history = []

        for block_idx, block in enumerate(self.blocks):
            if block_idx > 0:
                prev_forecast = self.blocks[block_idx - 1].instability_forecast
                prev_confidence = self.blocks[block_idx - 1].forecast_confidence

            if self.config.use_gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x, info = checkpoint(
                    block, x, self.depth, padding_mask, prev_forecast, prev_confidence,
                    use_reentrant=False
                )
            else:
                x, info = block(x, step=self.depth, padding_mask=padding_mask,
                                prev_forecast=prev_forecast, prev_confidence=prev_confidence)
            last_info = info

            if info and 'coherence' in info:
                all_layer_coherence.append(info['coherence'])
            layer_variances.append(info.get('variance', 0.0))
            instability_field_history.append(info.get('mean_instability_field', 0.0))
            confidence_history.append(info.get('mean_confidence', 1.0))

            if info.get('variance', 0.0) > max_variance_tracked:
                max_variance_tracked = info.get('variance', 0.0)

            if log_during_train and 'coherence' in info:
                self.consensus_stats.append(info['coherence'])

        n_layers = len(layer_variances)
        if n_layers >= 2:
            mid = n_layers // 2
            early_variance = sum(layer_variances[:mid]) / mid
            late_variance = sum(layer_variances[mid:]) / (n_layers - mid)
        else:
            early_variance = 0.0
            late_variance = 0.0

        total_pressure = 0.0
        pressure_by_governor = {'ffn': 0.0, 'alpha': 0.0, 'cap': 0.0, 'mpc': 0.0}

        for block_idx, block in enumerate(self.blocks):
            if hasattr(block, '_last_info') and block._last_info:
                info = block._last_info
                for gov in ['ffn', 'alpha', 'cap', 'mpc']:
                    work = info.get(f'{gov}_work', 0.0)
                    norm = info.get({
                        'ffn': 'mean_ffn_norm',
                        'alpha': 'mean_contrib_norm',
                        'cap': 'mean_raw_norm',
                        'mpc': 'mean_instability_field'
                    }[gov], 0.0)
                    pressure_by_governor[gov] += work * norm
                    total_pressure += work * norm

        concentration = max(pressure_by_governor.values()) / (total_pressure + 1e-8) if total_pressure > 0 else 0.0

        # Then chunked lm_head...
        # NEW (chunked — never allocates more than (B, 128, V) at once):
        x = self.final_norm(x)
        
        # ── v10.7: Chunked lm_head (Zero-Copy Logits) ──
        chunk_size = 128
        B, T, D = x.shape
        logits_chunks = []
        for i in range(0, T, chunk_size):
            chunk = x[:, i:i+chunk_size, :]
            logits_chunks.append(self.lm_head(chunk))
            
        # DO NOT torch.cat here unconditionally! 
        
        mean_coherence = sum(all_layer_coherence) / len(all_layer_coherence) if all_layer_coherence else 0.0
        self._last_info = {
            **last_info,
            'total_pressure': float(total_pressure),
            'pressure_concentration': float(concentration),
            'pressure_by_governor': {k: float(v) for k, v in pressure_by_governor.items()},
            'dominant_governor': max(pressure_by_governor, key=pressure_by_governor.get) if total_pressure > 0 else None,
            'coherence': mean_coherence,
            'avg_coherence': mean_coherence,
            'num_layers': len(all_layer_coherence),
            'layer_coherences': all_layer_coherence,
            'layer_variances': layer_variances,
            'instability_field_history': instability_field_history,
            'confidence_history': confidence_history,
            'early_var': early_variance,
            'late_var': late_variance,
            'variance_delta': early_variance - late_variance,
            'max_variance': max_variance_tracked,
            'compression_applied': compression_applied,
            'compress_ratio': self.config.compress_ratio if compression_applied else 1,
            'vram_saved': vram_saved_mb,
            'cumulative_gb': float(self.cumulative_saved_bytes.item()) / (1024 ** 3),
            'effective_seq_len': x.shape[1],
        }
        # During training, return the list to prevent torch.cat from allocating 594 MiB
        if self.training:
            return logits_chunks
            
        # During inference/generation, we need the full tensor
        return torch.cat(logits_chunks, dim=1)

    def get_hidden_states(self) -> Optional[torch.Tensor]:
        if self.blocks and hasattr(self.blocks[-1], '_hidden_state'):
            return self.blocks[-1]._hidden_state
        return None

    @torch.no_grad()
    def generate(self, prompt: str, tokenizer, max_new_tokens: int = 30, temperature: float = 0.7):
        self.eval()
        self.depth = 0
        device = next(self.parameters()).device
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        generated = input_ids.clone()
        for step in range(max_new_tokens):
            self.depth = step
            logits = self(generated, use_compression=False, log_during_train=False, padding_mask=None)
            hidden = self.get_hidden_states()
            dubito = self.dubito_monitor(hidden, self.depth) if hidden is not None else 0
            should_continue, _ = self.guardrails.should_continue(self.depth, dubito)
            if not should_continue:
                break
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
        return tokenizer.decode(generated[0], skip_special_tokens=True)


if __name__ == "__main__":
    config = MyceliaConfig()
    model = MyceliaLM(config)
    n = sum(p.numel() for p in model.parameters())
    print(f"MyceliaLM v10.6 (1.5B config): {n:,} parameters (~1.46B active with tied weights)")
    print(f"Fibonacci Geometric Resonance")
    print(f"  Golden ratio: φ = {_PHI:.10f}")
    print(f"  Golden angle: {_GOLDEN_ANGLE * 180 / math.pi:.6f}°")
    print(f"  Pair rotation: {config.fibonacci_pair_rotation}")
    print(f"  Governor: {'ACTIVE' if config.use_fibonacci_governor else 'DISABLED'}")
    print(f"  Raw scalar alphas — NO log-space, NO tanh, NO sigmoid")
    print(f"  Tied weights: embedding ↔ lm_head (saves ~310M params)")
    print(f"  Scale: d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")
    print(f"  Gradual transition: FFN 50→150 | α 100→150 over 10K steps")
    print(f"  Interaction guard: max 2 simultaneous, hysteresis 500 steps")
    print(f"  Control floor: 0.7")
    print(f"  ── v10.6 surgical patch ──")
    print(f"  Governor: PRE-mix → POST-mix (ATTN + FFN, clean α gradients)")
    print(f"  Alpha potential well: depth={config.alpha_well_depth}  (raw→0, α→1.0)")
    print(f"  Alpha grad-norm telemetry: active (meta-governor dashboard)")
    print(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
    print(f"  Compressor helpers: alpha_regularization_loss(), alpha_gradient_norms()")