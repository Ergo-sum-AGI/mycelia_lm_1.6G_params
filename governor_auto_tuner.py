"""
governor_auto_tuner.py — Governor Auto-Tuner v4.3
====================================================

Refactored auto-tuner that owns:
    - Phase computation and phase-aware suppression
    - R-guarded tuning decisions (constructive/marginal/compensatory)
    - EMA telemetry tracking
    - Structured TuningDecision output

The training loop becomes dumb: it calls tune(), receives a TuningDecision,
and applies it. No inline regime logic in the training loop.

Usage:
    from governor_auto_tuner import GovernorAutoTuner, TuningDecision

    auto_tuner = GovernorAutoTuner(model)
    decision = auto_tuner.tune(step, info)

    # Apply in training loop:
    for action in decision.actions:
        print(action)
    if decision.lr_multiplier is not None:
        for pg in opt.param_groups:
            pg['lr'] *= decision.lr_multiplier
    if decision.new_ffn_target is not None:
        for block in model.blocks:
            block.ffn_norm_target = decision.new_ffn_target
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

from optimization_state import PressureState, OptimizationRegime


# ─── CONSTANTS ────────────────────────────────────────────────────────────

AUTO_TUNE_EVERY = 5000
CONTROL_GAIN_MIN = 0.5
CONTROL_GAIN_MAX = 1.5
CONTROL_GAIN_DEFAULT = 1.0

# Phase-aware scheduling
DESTRUCTIVE_WINDOW_MIN = 0.12
DESTRUCTIVE_WINDOW_MAX = 0.25

# R-guarded thresholds
R_CONSERVATIVE_THRESHOLD = 1.5
R_COASTING_THRESHOLD = 3.0
R_CONFIDENCE_THRESHOLD = 0.8
R_SWEET_SPOT_MIN = 2.0
R_SWEET_SPOT_MAX = 3.0

# LR noise for coasting break
COASTING_NOISE_SCALE = 0.01


@dataclass
class TuningDecision:
    """Structured output from GovernorAutoTuner.tune().

    The training loop applies these fields directly. No logic, no conditionals.
    """
    actions: List[str] = None
    lr_multiplier: Optional[float] = None
    new_ffn_target: Optional[float] = None
    new_alpha_target: Optional[float] = None
    suppress_tuning: bool = False
    phase_action: Optional[str] = None
    r_action: Optional[str] = None
    notes: str = ""

    def __post_init__(self):
        if self.actions is None:
            self.actions = []


class GovernorAutoTuner:
    """Phase-aware, R-guarded auto-tuner with structured output."""

    def __init__(self, model, interval: int = AUTO_TUNE_EVERY):
        self.model = model
        self._interval = interval
        self.control_gain = CONTROL_GAIN_DEFAULT
        self.instability_target = 0.45
        self.coherence_weight = 1.0
        self.forecast_weight = 1.0
        self.mpc_intervention_ema = 0.5
        self.forecast_error_ema = 0.15
        self.cap_hit_ema = 0.10
        self.curvature_damping_ema = 1.0
        self._aggressive_count = 0
        self._last_tune_step = 0
        self._suppress_tuning = False
        self.scheduler_resurrected_count = 0
        # Phase tracking
        self._event_count = 0
        self._phase_history: deque = deque(maxlen=10)

    # ─── PHASE COMPUTATION ────────────────────────────────────────────────

    @property
    def phase(self) -> float:
        """Auto-tune phase (0.0–1.0) within the current interval.

        Phase 0.0 = just fired (post-fire, destructive window start)
        Phase 0.75–1.0 = pre-fire, constructive window
        """
        if self._interval <= 0:
            return 0.0
        return (self._last_tune_step % self._interval) / self._interval

    @property
    def event_count(self) -> int:
        """How many auto-tune events have fired so far."""
        return self._event_count

    def should_suppress(self, step: int, auto_tune_phase: Optional[float] = None) -> bool:
        """Should parameter mutations be suppressed in this phase window?

        Destructive window: 0.12–0.25 (post-fire transient ringing).
        Auto-clear after 2x interval to prevent stuck flags.
        """
        phase = auto_tune_phase if auto_tune_phase is not None else self.phase

        # Auto-clear stuck suppression
        if self._suppress_tuning and step - self._last_tune_step > 2 * self._interval:
            self._suppress_tuning = False

        if DESTRUCTIVE_WINDOW_MIN <= phase <= DESTRUCTIVE_WINDOW_MAX:
            self._suppress_tuning = True
            return True
        else:
            self._suppress_tuning = False
            return False

    # ─── TELEMETRY EMAs ───────────────────────────────────────────────────

    def update_telemetry_emas(self, info: Dict, alpha: float = 0.05):
        mpc = info.get('mpc_intervention_ratio', 0.0)
        fe = info.get('forecast_error', 0.0)
        ch = info.get('soft_cap_hit_ratio', 0.0)
        cd = info.get('curvature_damping', 1.0)
        self.mpc_intervention_ema = (1 - alpha) * self.mpc_intervention_ema + alpha * mpc
        self.forecast_error_ema = (1 - alpha) * self.forecast_error_ema + alpha * fe
        self.cap_hit_ema = (1 - alpha) * self.cap_hit_ema + alpha * ch
        self.curvature_damping_ema = (1 - alpha) * self.curvature_damping_ema + alpha * cd

    # ─── R-GUARDED TUNING ─────────────────────────────────────────────────

    def _r_guarded_decision(self, pressure: PressureState, info: Dict, step: int) -> TuningDecision:
        """Compute R-guarded tuning decision from PressureState.

        v11.9: Distinguish alpha dormancy (R=0 because alphas asleep) from
        genuine compensatory instability (R=0 because system is collapsing).
        Alpha dormancy requires wake-up, NOT conservative suppression.
        """
        R = pressure.ccr
        conf = pressure.confidence
        decision = TuningDecision()

        # ── v11.9: ALPHA DORMANCY DETECTION ─────────────────────────────
        # If R is pinned near zero AND AlphaScale has never fired AND
        # observed contrib_norm is far below alpha_norm_target, the alpha
        # channel is asleep. Conservative suppression (LR↓, FFN↓) makes it
        # worse by starving the model of the variance alphas need to activate.
        alpha_scale_ratio = info.get('alpha_scale_ratio', 0.0)
        contrib_norm = info.get('mean_contrib_norm', 0.0)
        alpha_target = info.get('alpha_target_current', 150.0)

        if R < 0.1 and alpha_scale_ratio < 0.01 and contrib_norm < alpha_target * 0.5:
            decision.r_action = "ALPHA_DORMANT"
            decision.new_alpha_target = max(30.0, contrib_norm * 0.8)
            decision.notes = (
                f"R={R:.3f}, alpha_scale={alpha_scale_ratio*100:.1f}%, "
                f"contrib_norm={contrib_norm:.1f} << target={alpha_target:.1f} — "
                f"ALPHAS ASLEEP. Wake-up target={decision.new_alpha_target:.0f}. "
                f"NO LR/FFN suppression."
            )
            decision.actions.append(
                f"ALPHA_WAKE: target→{decision.new_alpha_target:.0f} "
                f"(R={R:.3f})"
            )
            # Explicitly do NOT set lr_multiplier or new_ffn_target
            return decision

        # ── Standard R-guarded logic (unchanged below) ──────────────────
        if R < R_CONSERVATIVE_THRESHOLD:
            decision.r_action = "CONSERVATIVE_MODE"
            decision.lr_multiplier = 0.9
            decision.new_ffn_target = getattr(self.model.blocks[0], 'ffn_norm_target', 150.0) * 0.95
            decision.notes = f"R={R:.3f} < {R_CONSERVATIVE_THRESHOLD} — conservative mode"
            decision.actions.append(f"LR↓10% (R={R:.3f})")
            decision.actions.append(f"FFN↓5% (R={R:.3f})")

        elif R > R_COASTING_THRESHOLD and conf > R_CONFIDENCE_THRESHOLD:
            decision.r_action = "BREAK_COASTING"
            noise = np.random.normal(0, COASTING_NOISE_SCALE)
            decision.lr_multiplier = 1.0 + noise
            decision.notes = f"R={R:.3f}, conf={conf:.3f} — injecting LR noise"
            decision.actions.append(f"LR noise ±{abs(noise)*100:.1f}% (R={R:.3f})")

        elif R_SWEET_SPOT_MIN <= R <= R_SWEET_SPOT_MAX and conf < R_CONFIDENCE_THRESHOLD:
            decision.r_action = "CONSTRUCTIVE_LEARNING"
            decision.notes = f"R={R:.3f} in sweet spot — no intervention"

        else:
            decision.r_action = None
            decision.notes = f"R={R:.3f} — no R-guard action"

        return decision

    # ─── MAIN TUNE METHOD ─────────────────────────────────────────────────

    def tune(self, step: int, info: Dict) -> TuningDecision:
        """Main tuning entry point. Returns structured TuningDecision.

        The training loop should:
            1. Call this every LOG_EVERY steps (or as desired)
            2. Apply decision.lr_multiplier to optimizer LR
            3. Apply decision.new_ffn_target to all blocks
            4. Print decision.actions for telemetry
        """
        # Check cooldown
        if step - self._last_tune_step < AUTO_TUNE_EVERY:
            return TuningDecision(notes="Cooldown active")

        # Phase-aware suppression
        if self.should_suppress(step):
            return TuningDecision(
                suppress_tuning=True,
                phase_action="DESTRUCTIVE_WINDOW",
                notes=f"Phase={self.phase:.3f} in destructive window — suppression active"
            )

        self._last_tune_step = step
        self._event_count += 1
        self._phase_history.append(self.phase)

        # Build PressureState from telemetry
        pressure = PressureState.from_telemetry(info)

        # R-guarded decision
        r_decision = self._r_guarded_decision(pressure, step)

        # Classic auto-tune parameter mutations (EMA-based)
        actions = []
        mpc = self.mpc_intervention_ema
        fe = self.forecast_error_ema
        ch = self.cap_hit_ema
        cd = self.curvature_damping_ema

        if mpc > 0.70:
            self.instability_target = min(0.7, self.instability_target * 1.10)
            actions.append(f"instability_target↑{self.instability_target:.3f}")
        elif mpc < 0.05:
            self.instability_target = max(0.1, self.instability_target * 0.90)
            actions.append(f"instability_target↓{self.instability_target:.3f}")

        if ch > 0.50 and self.control_gain < CONTROL_GAIN_MAX:
            self.control_gain = min(CONTROL_GAIN_MAX, self.control_gain * 1.05)
            actions.append(f"control_gain↑{self.control_gain:.3f}")
        elif ch < 0.01 and mpc < 0.10 and self.control_gain > CONTROL_GAIN_MIN:
            self.control_gain = max(CONTROL_GAIN_MIN, self.control_gain * 0.95)
            actions.append(f"control_gain↓{self.control_gain:.3f}")

        if fe > 0.30:
            self.coherence_weight = min(2.0, self.coherence_weight * 1.02)
            self.forecast_weight = min(2.0, self.forecast_weight * 1.02)
            actions.append(f"coh/fcst weights↑{self.coherence_weight:.2f}/{self.forecast_weight:.2f}")
        elif fe < 0.05:
            self.coherence_weight = max(0.5, self.coherence_weight * 0.99)
            self.forecast_weight = max(0.5, self.forecast_weight * 0.99)

        # Propagate to blocks
        for block in self.model.blocks:
            block.instability_target = self.instability_target
            block.control_gain = self.control_gain
            block.coherence_weight = self.coherence_weight
            block.forecast_weight = self.forecast_weight
            # Apply R-guarded FFN target if set
            if r_decision.new_ffn_target is not None:
                block.ffn_norm_target = r_decision.new_ffn_target
            # v11.9: Propagate alpha wake-up target if set
            if r_decision.new_alpha_target is not None:
                block.alpha_norm_target = r_decision.new_alpha_target
        # Merge actions
        all_actions = actions + r_decision.actions

        return TuningDecision(
            actions=all_actions,
            lr_multiplier=r_decision.lr_multiplier,
            new_ffn_target=r_decision.new_ffn_target,
            suppress_tuning=False,
            phase_action=None,
            r_action=r_decision.r_action,
            notes=r_decision.notes,
        )

    def get_state(self) -> Dict:
        """Serialize state for checkpointing."""
        return {
            'control_gain': self.control_gain,
            'instability_target': self.instability_target,
            'coherence_weight': self.coherence_weight,
            'forecast_weight': self.forecast_weight,
            'last_tune_step': self._last_tune_step,
            'event_count': self._event_count,
        }

    def load_state(self, state: Dict):
        """Restore state from checkpoint."""
        self.control_gain = state.get('control_gain', CONTROL_GAIN_DEFAULT)
        self.instability_target = state.get('instability_target', 0.45)
        self.coherence_weight = state.get('coherence_weight', 1.0)
        self.forecast_weight = state.get('forecast_weight', 1.0)
        self._last_tune_step = state.get('last_tune_step', 0)
        self._event_count = state.get('event_count', 0)
