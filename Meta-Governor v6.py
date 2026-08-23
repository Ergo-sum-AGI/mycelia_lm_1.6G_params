"""
Meta-Governor v6 — All agents on rotation + Scheduler-Aware + Active Teaching
==================================================================
Dual-agent consensus system with Groq LPU inference.
Scheduler health telemetry, frequent rounds (250 steps), verbose teaching.

All 7 agents rotating:
  - qwen/qwen3.6-27b (macro): Trajectory analysis, pattern detection, rpm_limit=60
  - gpt-oss-20b (micro): Fast numeric parameter tuning, rpm_limit=30
  - deepseek-v4-pro" rpm_limit=60
  - gemini-2.5-flash-lite" rpm_limit=15
  - moonshot-v1-8k" rpm_limit=20
  - gpt-oss-120b" rpm_limit=30
  - grok-4.5" rpm_limit=60
Teaching Protocol:
  Agents emit "lesson" rationales that are stored in model._teacher_rationales.
  These are physics/computer-science insights about hidden state dynamics.
  The model can later reflect on these (e.g., via auxiliary loss or prompt
  injection). For now, they are logged and stored as structured metadata.

Consensus Principle:
  Both agents must AGREE on variable AND direction
  OR one agent abstains (confidence < threshold). Disagreement = no action.

Usage:
    from meta_governor import integrate_meta_governor, TelemetryPacket

    # In training loop:
    if step % LOG_EVERY == 0:
        actions = integrate_meta_governor(model, auto_tuner, step, loss, lr)
"""

import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, List, Literal, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import deque
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError, ConfigDict, field_validator

# =============================================================================
# GROQ CLIENT
# =============================================================================

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    print("⚠️  groq not installed, using requests fallback")

# API key: env var first, then legacy fallback
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
os.environ["XAI_API_KEY"]
os.environ["MINIMAX_API_KEY"]
# ── v4.1: Alibaba DashScope (Native OpenAI compat, native JSON mode) ──
os.environ["DASHSCOPE_API_KEY"]
# Free tier models on Groq
GROQ_MICRO_MODEL = "openai/gpt-oss-20b"         # Groq replacement for llama-3.1-8b-instant      # Fast numeric tuning, 30 RPM
GROQ_MACRO_MODEL = "qwen/qwen3.6-27b"           # Trajectory analysis, 60 RPM

# Rate limit tracking (RPM guards)
MICRO_RPM_LIMIT = 30
MACRO_RPM_LIMIT = 60
RATE_LIMIT_SAFETY_FACTOR = 0.8  # Use only 80% of RPM to avoid edge 429s

# Circuit breaker thresholds
CIRCUIT_BREAKER_FAILURE_WINDOW = 10
CIRCUIT_BREAKER_TRIP_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_THRESHOLD = 3

# Expert confidence bounds
EXPERT_CONFIDENCE_MIN = 0.05
EXPERT_CONFIDENCE_MAX = 0.95
EXPERT_CONFIDENCE_DEFAULT = 0.50

# Consensus threshold — LOWERED per user request to see more agent work
CONSENSUS_CONFIDENCE_THRESHOLD = 0.20

# API timeout per call (seconds) — Groq LPU is fast
API_TIMEOUT = 6

# Async worker pool
MAX_WORKERS = 4


# =============================================================================
# PYDANTIC SCHEMAS — Strict Validation
# =============================================================================

class VariableName(str, Enum):
    """Valid variables the meta-governor can adjust."""
    FFN_NORM_TARGET = "ffn_norm_target"
    ALPHA_NORM_TARGET = "alpha_norm_target"
    CONTROL_GAIN = "control_gain"
    INSTABILITY_TARGET = "instability_target"
    SOFT_CAP = "soft_cap"
    WEIGHT_FLOOR = "weight_floor"
    TEMPERATURE = "temperature"
    BURN_IN_STEPS = "burn_in_steps"
    EXPECTED_CURVATURE = "expected_curvature"


class Direction(str, Enum):
    """Valid directions."""
    RAISE = "raise"
    LOWER = "lower"
    SET = "set"
    PAUSE = "pause"
    INVESTIGATE = "investigate"


class Magnitude(str, Enum):
    """Adjustment magnitude."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EMERGENCY = "emergency"


class Enactable(str, Enum):
    """Whether suggestion can be applied automatically."""
    RUNTIME = "runtime"
    REQUIRES_RESTART = "requires_restart"
    REQUIRES_HUMAN = "requires_human"


class ExpectedOutcome(str, Enum):
    """Valid expected outcomes."""
    LOSS_DECREASE = "loss_decrease"
    PRESSURE_REDISTRIBUTE = "pressure_redistribute"
    COHERENCE_INCREASE = "coherence_increase"
    REGIME_TRANSITION = "regime_transition"
    STABILITY_IMPROVEMENT = "stability_improvement"


class MetaGovernorAction(BaseModel):
    """Structured, machine-comparable action from any agent."""
    model_config = ConfigDict(extra="ignore")  # Pydantic v2

# ═════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS — v3.42: Permissive Coercion
# ═════════════════════════════════════════════════════════════════════

class VariableName(str, Enum):
    """Valid variables the meta-governor can adjust."""
    FFN_NORM_TARGET = "ffn_norm_target"
    ALPHA_NORM_TARGET = "alpha_norm_target"
    CONTROL_GAIN = "control_gain"
    INSTABILITY_TARGET = "instability_target"
    SOFT_CAP = "soft_cap"
    WEIGHT_FLOOR = "weight_floor"
    TEMPERATURE = "temperature"
    BURN_IN_STEPS = "burn_in_steps"
    EXPECTED_CURVATURE = "expected_curvature"
    ALPHA_WELL_DEPTH = "alpha_well_depth"
    ALPHA_WELL_INVERT = "alpha_well_invert"
    AUTO_TUNE_PHASE = "auto_tune_phase"  # NEW: Moonshot-KIMI keeps suggesting this


class Direction(str, Enum):
    """Valid directions."""
    RAISE = "raise"
    LOWER = "lower"
    SET = "set"
    PAUSE = "pause"
    INVESTIGATE = "investigate"


class Magnitude(str, Enum):
    """Adjustment magnitude."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EMERGENCY = "emergency"


class Enactable(str, Enum):
    """Whether suggestion can be applied automatically."""
    RUNTIME = "runtime"
    REQUIRES_RESTART = "requires_restart"
    REQUIRES_HUMAN = "requires_human"


class ExpectedOutcome(str, Enum):
    """Valid expected outcomes."""
    LOSS_DECREASE = "loss_decrease"
    PRESSURE_REDISTRIBUTE = "pressure_redistribute"
    COHERENCE_INCREASE = "coherence_increase"
    REGIME_TRANSITION = "regime_transition"
    STABILITY_IMPROVEMENT = "stability_improvement"


class MetaGovernorAction(BaseModel):
    """Structured, machine-comparable action from any agent.
    
    v3.42: Added mode='before' validators to coerce common LLM outputs
    into valid enum values BEFORE Pydantic rejects them.
    """
    model_config = ConfigDict(extra="ignore")  # Pydantic v2

    variable: VariableName
    direction: Direction
    magnitude: Magnitude = Field(default=Magnitude.SMALL)
    value: Optional[float] = Field(default=None, description="Quantified value for SET")
    enactable: Enactable = Field(default=Enactable.RUNTIME)
    duration_steps: int = Field(default=500, ge=10, le=5000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = Field(default="", description="Single string OR array of <500-char chunks; reassembled, never truncated")
    lesson: str = Field(default="", description="Teaching insight; single string OR array of <500-char chunks; reassembled, never truncated")

    expected_outcome: Optional[ExpectedOutcome] = Field(
        default=None,
        description="Predicted result for later verification"
    )

    # ── v3.42: COERCION VALIDATORS ──────────────────────────────────
    
    @field_validator("variable", mode="before")
    @classmethod
    def coerce_variable(cls, v):
        """Map common LLM variable names to valid enum values."""
        if isinstance(v, str):
            v = v.lower().strip().replace(" ", "_").replace("-", "_")
            # Common synonyms agents produce
            synonyms = {
                "ffn_target": "ffn_norm_target",
                "ffn_norm": "ffn_norm_target",
                "alpha_target": "alpha_norm_target",
                "alpha_norm": "alpha_norm_target",
                "gain": "control_gain",
                "control": "control_gain",
                "instability": "instability_target",
                "inst_target": "instability_target",
                "cap": "soft_cap",
                "norm_cap": "soft_cap",
                "curvature": "expected_curvature",
                "exp_curvature": "expected_curvature",
                "well_depth": "alpha_well_depth",
                "alpha_well": "alpha_well_depth",
                "phase": "auto_tune_phase",
                "auto_tune": "auto_tune_phase",
                "tune_phase": "auto_tune_phase",
            }
            if v in synonyms:
                return synonyms[v]
        return v

    @field_validator("direction", mode="before")
    @classmethod
    def coerce_direction(cls, v):
        """Map common LLM direction words to valid enum values."""
        if isinstance(v, str):
            v = v.lower().strip()
            synonyms = {
                "up": "raise",
                "increase": "raise",
                "increment": "raise",
                "boost": "raise",
                "deepen": "raise",
                "widen": "raise",
                "down": "lower",
                "decrease": "lower",
                "reduce": "lower",
                "shrink": "lower",
                "flatten": "lower",
                "narrow": "lower",
                "fix": "set",
                "assign": "set",
                "stop": "pause",
                "halt": "pause",
                "freeze": "pause",
                "check": "investigate",
                "inspect": "investigate",
                "analyze": "investigate",
                "monitor": "investigate",
            }
            if v in synonyms:
                return synonyms[v]
        return v

    @field_validator("magnitude", mode="before")
    @classmethod
    def coerce_magnitude(cls, v):
        """Coerce floats, numeric strings, and synonym words to valid enum."""
        # Handle numeric inputs (0.05, 0.2, 0.5, etc.)
        if isinstance(v, (int, float)):
            if v <= 0.1:
                return "small"
            elif v <= 0.3:
                return "medium"
            elif v <= 0.7:
                return "large"
            else:
                return "emergency"
        
        if isinstance(v, str):
            v_lower = v.lower().strip()
            
            # Numeric strings: "0.05", "0.2", etc.
            try:
                num = float(v_lower)
                if num <= 0.1:
                    return "small"
                elif num <= 0.3:
                    return "medium"
                elif num <= 0.7:
                    return "large"
                else:
                    return "emergency"
            except ValueError:
                pass
            
            # Word synonyms
            synonyms = {
                "tiny": "small",
                "minor": "small",
                "gentle": "small",
                "slight": "small",
                "conservative": "small",
                "cautious": "small",
                "moderate": "medium",
                "medium": "medium",
                "normal": "medium",
                "standard": "medium",
                "significant": "large",
                "major": "large",
                "substantial": "large",
                "strong": "large",
                "aggressive": "large",
                "critical": "emergency",
                "urgent": "emergency",
                "extreme": "emergency",
                "max": "emergency",
                # Direction words mistakenly used as magnitude
                "raise": "small",
                "lower": "small",
                "increase": "small",
                "decrease": "small",
                "up": "small",
                "down": "small",
                "pause": "small",
                "investigate": "small",
            }
            if v_lower in synonyms:
                return synonyms[v_lower]
        
        # Fallback: default to small rather than rejecting
        return "small"

    @field_validator("enactable", mode="before")
    @classmethod
    def coerce_enactable(cls, v):
        """Coerce common enactable synonyms."""
        if isinstance(v, str):
            v = v.lower().strip().replace(" ", "_")
            synonyms = {
                "auto": "runtime",
                "automatic": "runtime",
                "now": "runtime",
                "immediate": "runtime",
                "restart": "requires_restart",
                "reboot": "requires_restart",
                "human": "requires_human",
                "manual": "requires_human",
                "review": "requires_human",
            }
            if v in synonyms:
                return synonyms[v]
        return v

    @field_validator("expected_outcome", mode="before")
    @classmethod
    def coerce_expected_outcome(cls, v):
        """Map common LLM outcome words to valid enum values."""
        if isinstance(v, str):
            v = v.lower().strip().replace(" ", "_").replace("-", "_")
            synonyms = {
                "loss_down": "loss_decrease",
                "loss_drop": "loss_decrease",
                "lower_loss": "loss_decrease",
                "pressure_drop": "pressure_redistribute",
                "pressure_relief": "pressure_redistribute",
                "load_balance": "pressure_redistribute",
                "coherence_up": "coherence_increase",
                "more_coherent": "coherence_increase",
                "regime_change": "regime_transition",
                "phase_change": "regime_transition",
                "stability_up": "stability_improvement",
                "more_stable": "stability_improvement",
            }
            if v in synonyms:
                return synonyms[v]
        return v

    @field_validator("value")
    @classmethod
    def validate_value_bounds(cls, v, info):
        """Sanity checks on values based on variable."""
        if v is None:
            return v
        var = info.data.get("variable")
        bounds = {
            VariableName.FFN_NORM_TARGET: (10, 1000),
            VariableName.ALPHA_NORM_TARGET: (10, 1000),
            VariableName.CONTROL_GAIN: (0.01, 10.0),
            VariableName.INSTABILITY_TARGET: (0.01, 1.0),
            VariableName.SOFT_CAP: (100, 2000),
            VariableName.WEIGHT_FLOOR: (3, 8),
            VariableName.TEMPERATURE: (0.3, 2.0),
            VariableName.BURN_IN_STEPS: (100, 10000),
            VariableName.EXPECTED_CURVATURE: (0.1, 2.0),
            VariableName.ALPHA_WELL_DEPTH: (-0.2, 0.6),
            VariableName.ALPHA_WELL_INVERT: (-0.2, 0.6),
            VariableName.AUTO_TUNE_PHASE: (0.0, 1.0),
        }
        if var in bounds:
            lo, hi = bounds[var]
            if not (lo <= v <= hi):
                # Clamp instead of rejecting — preserve the agent's intent
                return max(lo, min(hi, v))
        return v

    @field_validator("duration_steps", mode="before")
    @classmethod
    def coerce_duration(cls, v):
        """Coerce string durations to int."""
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return 500
        if v is None:
            return 500
        return max(10, min(5000, int(v)))

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v):
        """Coerce string confidence to float, clamp to [0, 1]."""
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                return 0.5
        if v is None:
            return 0.5
        return max(0.0, min(1.0, float(v)))

    # ── v3.45: Reassemble chunked text — never truncate, never reject ──
    @field_validator("rationale", "lesson", mode="before")
    @classmethod
    def reassemble_chunked_text(cls, v):
        """Macro strategists often need >500 chars for physics reasoning.
        Instead of truncating (losing content) or rejecting the action,
        accept EITHER a single string of any length OR an array of chunks,
        and reassemble chunks into the full text. No valid decision is ever
        discarded for being verbose."""
        if isinstance(v, list):
            chunks = [c.strip() for c in v if isinstance(c, str) and c.strip()]
            return " ".join(chunks)
        if isinstance(v, str):
            return v.strip()
        return ""

# ═════════════════════════════════════════════════════════════════════
# PARAMETER REGISTRY / VERIFIER — v3.5
# Every valid VariableName MUST have an entry. Unmapped variables are
# explicitly rejected with a log line, never silently dropped.
# ═════════════════════════════════════════════════════════════════════

class ParameterTarget(str, Enum):
    BLOCK = "block"
    GLOBAL = "global"
    MODEL = "model"


PARAMETER_REGISTRY: Dict[VariableName, Dict[str, Any]] = {
    VariableName.FFN_NORM_TARGET: {
        "target": ParameterTarget.BLOCK,
        "bounds": (10, 1000),
        "default_expected": ExpectedOutcome.PRESSURE_REDISTRIBUTE,
    },
    VariableName.ALPHA_NORM_TARGET: {
        "target": ParameterTarget.BLOCK,
        "bounds": (10, 1000),
        "default_expected": ExpectedOutcome.PRESSURE_REDISTRIBUTE,
    },
    VariableName.SOFT_CAP: {
        "target": ParameterTarget.BLOCK,
        "bounds": (100, 2000),
        "default_expected": ExpectedOutcome.STABILITY_IMPROVEMENT,
    },
    VariableName.EXPECTED_CURVATURE: {
        "target": ParameterTarget.BLOCK,
        "bounds": (0.1, 2.0),
        "default_expected": ExpectedOutcome.COHERENCE_INCREASE,
    },
    VariableName.ALPHA_WELL_DEPTH: {
        "target": ParameterTarget.BLOCK,
        "bounds": (-0.2, 0.6),
        "default_expected": ExpectedOutcome.REGIME_TRANSITION,
    },
    VariableName.ALPHA_WELL_INVERT: {
        "target": ParameterTarget.BLOCK,
        "bounds": (-0.2, 0.6),
        "default_expected": ExpectedOutcome.REGIME_TRANSITION,
    },
    VariableName.WEIGHT_FLOOR: {
        "target": ParameterTarget.BLOCK,
        "bounds": (3, 8),
        "default_expected": ExpectedOutcome.STABILITY_IMPROVEMENT,
    },
    VariableName.TEMPERATURE: {
        "target": ParameterTarget.MODEL,
        "bounds": (0.3, 2.0),
        "default_expected": ExpectedOutcome.STABILITY_IMPROVEMENT,
    },
    VariableName.BURN_IN_STEPS: {
        "target": ParameterTarget.MODEL,
        "bounds": (100, 10000),
        "default_expected": ExpectedOutcome.STABILITY_IMPROVEMENT,
    },
    VariableName.AUTO_TUNE_PHASE: {
        "target": ParameterTarget.MODEL,
        "bounds": (0.0, 1.0),
        "default_expected": ExpectedOutcome.REGIME_TRANSITION,
    },
    VariableName.CONTROL_GAIN: {
        "target": ParameterTarget.GLOBAL,
        "bounds": (0.01, 10.0),
        "default_expected": ExpectedOutcome.STABILITY_IMPROVEMENT,
    },
    VariableName.INSTABILITY_TARGET: {
        "target": ParameterTarget.GLOBAL,
        "bounds": (0.01, 1.0),
        "default_expected": ExpectedOutcome.STABILITY_IMPROVEMENT,
    },
}

class AgentResponse(BaseModel):
    """Wrapper for agent response with metadata."""
    model_config = ConfigDict(extra="ignore")

    agent_name: str
    action: Optional[MetaGovernorAction] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ConsensusDecision(BaseModel):
    """The resolved consensus decision from multiple agents."""
    model_config = ConfigDict(extra="ignore")

    variable: Optional[VariableName] = None
    direction: Optional[Direction] = None
    value: Optional[float] = None
    confidence: float = 0.0
    duration_steps: int = 500
    expected_outcome: Optional[ExpectedOutcome] = None
    rationale: str = ""
    lesson: str = ""  # NEW v3.1: Aggregated teaching
    enactable: Enactable = Field(default=Enactable.RUNTIME)
    participating_agents: List[str] = Field(default_factory=list)
    dissenting_agents: List[str] = Field(default_factory=list)
    consensus_type: Literal["unanimous", "majority", "single", "none", "conflict"] = "none"


# =============================================================================
# TELEMETRY COMPRESSOR v1.0 — SVD + TDA for Groq TPM efficiency
# =============================================================================

import numpy as np


class TelemetryCompressor:
    """
    Compresses high-dimensional telemetry into compact text descriptors.

    Replaces raw layer vectors (12 floats each) and history arrays with:
    - SVD-based dominant mode descriptors (trend + explained variance)
    - TDA-inspired topology invariants (entropy, persistence, Betti-0)
    - Pressure tensor ratio signatures

    Compression ratio: ~3-5:1 on telemetry payload
    """

    def __init__(self, n_layers: int = 24):
        self.n_layers = n_layers

    def _svd_compress(self, vector: List[float]) -> Tuple[str, float]:
        """Compress layer-wise vector into dominant trend descriptor."""
        if len(vector) < 3:
            return "insufficient", 0.0

        arr = np.array(vector, dtype=np.float64)
        x = np.arange(len(arr), dtype=np.float64)

        # Linear trend = first principal component for 1D
        slope, intercept = np.polyfit(x, arr, 1)
        trend = slope * x + intercept

        ss_res = np.sum((arr - trend) ** 2)
        ss_tot = np.sum((arr - np.mean(arr)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

        # Classify mode
        std = np.std(arr)
        if std < 1e-10:
            mode = "flat"
        elif abs(slope) < 0.01 * std:
            mode = "uniform"
        elif slope > 0:
            mode = "late↑"
        else:
            mode = "early↑"

        # Check for U-shape (quadratic dominance)
        if len(arr) >= 4 and std > 1e-10:
            quad = np.polyfit(x, arr, 2)[0]
            if abs(quad) > 0.05 * abs(slope) and quad > 0:
                mode = "mid↓"

        return f"{mode}|R²={r2:.2f}", r2

    def _tda_compress(self, history: List[float]) -> str:
        """Compress time-series into topological invariants."""
        if len(history) < 3:
            return "N/A"

        arr = np.array(history, dtype=np.float64)

        # Shannon entropy (normalized)
        hist, _ = np.histogram(arr, bins=min(5, len(arr)), density=True)
        hist = hist[hist > 1e-10]
        entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
        entropy_norm = entropy / np.log2(len(hist)) if len(hist) > 1 else 0.0

        # Persistence: oscillation count via second derivative sign changes
        if len(arr) >= 3:
            diff2 = np.diff(arr, n=2)
            persistence = int(np.sum(np.diff(np.sign(diff2)) != 0))
        else:
            persistence = 0

        # Betti-0: connected components above median
        median = np.median(arr)
        above = arr > median
        if len(above) > 1:
            transitions = int(np.sum(np.diff(above.astype(int)) != 0))
            components = transitions + 1
        else:
            components = 1

        # Trend shift: second half vs first half
        if len(arr) >= 4:
            mid = len(arr) // 2
            shift = float(np.mean(arr[mid:]) - np.mean(arr[:mid]))
            shift_str = f"{'+' if shift >= 0 else ''}{shift:.2f}"
        else:
            shift_str = "N/A"

        return f"H={entropy_norm:.2f}|P={persistence}|β₀={components}|Δ={shift_str}"

    def _pressure_compress(self, packet: "TelemetryPacket") -> str:
        """Compress pressure tensor into compact ratio signature."""
        pi = packet.pressure_total
        chi = packet.pressure_concentration
        breakdown = packet.pressure_by_governor

        if not breakdown or pi <= 0:
            return "N/A"

        # Normalize
        ffn = breakdown.get("ffn", 0.0) / pi
        alpha = breakdown.get("alpha", 0.0) / pi
        cap = breakdown.get("cap", 0.0) / pi
        mpc = breakdown.get("mpc", 0.0) / pi

        # Dominant
        max_val = max(ffn, alpha, cap, mpc)
        dominant = "ffn" if ffn == max_val else "α" if alpha == max_val else "cap" if cap == max_val else "mpc"

        # Ratios relative to smallest non-zero
        vals = [v for v in [ffn, alpha, cap, mpc] if v > 0.001]
        if not vals:
            return f"{dominant}|χ={chi:.2f}|Π={pi:.0f}"

        min_v = min(vals)
        r_ffn, r_alpha, r_cap, r_mpc = ffn/min_v, alpha/min_v, cap/min_v, mpc/min_v

        return f"{dominant}-dom|χ={chi:.2f}|Π={pi:.0f}|f:α:c:m={r_ffn:.1f}:{r_alpha:.1f}:{r_cap:.1f}:{r_mpc:.2f}"

    def compress(self, packet: "TelemetryPacket") -> Dict:
        """Compress full telemetry into compact dict for LLM payload."""

        # SVD on layer metrics
        coh_mode, coh_r2 = self._svd_compress(packet.layer_coherences)
        inst_mode, inst_r2 = self._svd_compress(packet.layer_instability)

        # Pick the more structured signal
        if inst_r2 > coh_r2:
            layer_desc = f"Inst:{inst_mode}"
            layer_var = inst_r2
        else:
            layer_desc = f"Coh:{coh_mode}"
            layer_var = coh_r2

        # TDA on histories
        i_topo = self._tda_compress(packet.instability_history)
        conf_topo = self._tda_compress(packet.confidence_history)

        # Pressure
        pressure_sig = self._pressure_compress(packet)

        return {
            "loss": round(packet.loss, 4),
            "lr": f"{packet.lr:.2e}",
            "coh": round(packet.coherence, 3),
            "friction": packet.friction_regime,
            "layer": layer_desc,
            "layer_var": round(layer_var, 2),
            "i_topo": i_topo,
            "conf_topo": conf_topo,
            "pressure": pressure_sig,
            "mpc": round(packet.mpc_intervention, 3),
            "forecast_err": round(packet.forecast_error, 3),
            "vel": round(packet.mean_velocity, 4),
            "acc": round(packet.mean_acceleration, 4),
            "curv": round(packet.mean_curvature, 4),
            "scheduler_alive": packet.scheduler_alive,
            "scheduler_resurrected": packet.scheduler_resurrected_count,
        }


# =============================================================================
# TELEMETRY PACKET — v3.1: Scheduler-Aware
# =============================================================================

@dataclass
class TelemetryPacket:
    """Structured telemetry from Mycelia training loop."""
    step: int
    loss: float
    lr: float
    coherence: float = 0.0
    friction_regime: str = "UNKNOWN"
    delta: float = 0.0
    early_var: float = 0.0
    late_var: float = 0.0
    # RoPE-specific metrics
    rope_stability: float = 0.0
    positional_coherence: float = 0.0

    # Layer-wise for all layers
    layer_coherences: List[float] = field(default_factory=list)
    layer_instability: List[float] = field(default_factory=list)

    # Dynamic hidden state metrics
    mean_velocity: float = 0.0
    mean_acceleration: float = 0.0
    mean_curvature: float = 0.0
    mean_jerk: float = 0.0

    # Governor metrics
    ffn_veto_ratio: float = 0.0
    ffn_norm_mean: float = 0.0
    ffn_norm_max: float = 0.0
    ffn_target: float = 0.0
    alpha_scale_ratio: float = 0.0
    alpha_contrib_norm: float = 0.0
    alpha_target: float = 0.0
    soft_cap_hit: float = 0.0
    soft_cap_max_raw: float = 0.0
    mpc_intervention: float = 0.0
    mpc_control_factor: float = 1.0
    instability_field_mean: float = 0.0
    forecast_error: float = 0.0

    # Pressure tensor
    pressure_total: float = 0.0
    pressure_concentration: float = 0.0
    pressure_dominant: str = "none"
    pressure_by_governor: Dict[str, float] = field(default_factory=dict)

    # Geometry
    max_curvature: float = 0.0

    # Rate governor
    rate_governor_hit: float = 0.0
    rate_scale_mean: float = 1.0
    ffn_growth_ratio: float = 1.0
    residual_growth_ratio: float = 1.0

    # I-field history (per layer)
    instability_history: List[float] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)

    # Meta-governor state (for recursive monitoring)
    regime_duration: int = 0
    veto_rate_high_duration: int = 0
    meta_governor_last_step: int = 0
    meta_governor_active_adjustments: int = 0
    architecture_suggestions_pending: int = 0
    human_review_pending: int = 0

    # NEW v3.1: Scheduler health telemetry
    scheduler_alive: bool = True
    scheduler_last_epoch: int = 0
    scheduler_total_steps: int = 0
    scheduler_resurrected_count: int = 0
    lr_valid: bool = True

    # Meta
    model_version: str = "v9.0"
    session_id: str = "default"

    @classmethod
    def from_model(cls, model, step: int, loss: float, lr: float,
                   scheduler_info: Optional[Dict] = None) -> "TelemetryPacket":
        """Extract telemetry from Mycelia model's _last_info."""
        info = getattr(model, "_last_info", {}) or {}
        scheduler_info = scheduler_info or {}
        cfg = getattr(model, 'config', None) or getattr(model, 'cfg', None)
        if cfg is not None and hasattr(cfg, 'n_layers'):
            n_layers = cfg.n_layers
        elif hasattr(model, 'blocks'):
            n_layers = len(model.blocks)
        else:
            n_layers = 12  # ultimate fallback for orphan packets
        
        # ── SAFE CASTING HELPERS (Prevents CUDA tensor leakage to NumPy) ──
        def safe_float(val, default=0.0):
            if val is None: return float(default)
            if hasattr(val, "item"): return float(val.item())
            try: return float(val)
            except (TypeError, ValueError): return float(default)

        def safe_int(val, default=0):
            if val is None: return int(default)
            if hasattr(val, "item"): return int(val.item())
            try: return int(val)
            except (TypeError, ValueError): return int(default)

        def safe_float_list(lst):
            if not lst: return []
            return [safe_float(x) for x in lst]

        def safe_float_dict(d):
            if not d: return {}
            return {k: safe_float(v) for k, v in d.items()}

        return cls(
            step=step,
            loss=loss,
            lr=lr,
            coherence=safe_float(info.get("coherence", 0.0)),
            friction_regime=info.get("friction", "UNKNOWN"),
            delta=safe_float(info.get("variance_delta", 0.0)),
            early_var=safe_float(info.get("early_var", 0.0)),
            late_var=safe_float(info.get("late_var", 0.0)),
            rope_stability=safe_float(info.get("rope_stability", 0.0)),
            positional_coherence=safe_float(info.get("positional_coherence", 0.0)),
            layer_coherences=safe_float_list(info.get("layer_coherences", [])),
            layer_instability=safe_float_list(info.get("layer_instability", [])),
            ffn_veto_ratio=safe_float(info.get("ffn_veto_ratio", 0.0)),
            ffn_norm_mean=safe_float(info.get("mean_ffn_norm", 0.0)),
            ffn_norm_max=safe_float(info.get("max_ffn_norm", 0.0)),
            ffn_target=safe_float(info.get("ffn_target_current", 150.0)),
            alpha_scale_ratio=safe_float(info.get("alpha_scale_ratio", 0.0)),
            alpha_contrib_norm=safe_float(info.get("mean_contrib_norm", 0.0)),
            alpha_target=safe_float(info.get("alpha_target_current", 150.0)),
            soft_cap_hit=safe_float(info.get("soft_cap_hit_ratio", 0.0)),
            soft_cap_max_raw=safe_float(info.get("max_raw_norm", 0.0)),
            mpc_intervention=safe_float(info.get("mpc_intervention_ratio", 0.0)),
            mpc_control_factor=safe_float(info.get("mean_control_factor", 1.0)),
            instability_field_mean=safe_float(info.get("mean_instability_field", 0.0)),
            forecast_error=safe_float(info.get("forecast_error", 0.0)),
            pressure_total=safe_float(info.get("total_pressure", 0.0)),
            pressure_concentration=safe_float(info.get("pressure_concentration", 0.0)),
            pressure_dominant=info.get("dominant_governor", "none"),
            pressure_by_governor=safe_float_dict(info.get("pressure_by_governor", {})),
            mean_velocity=safe_float(info.get("mean_velocity", 0.0)),
            mean_acceleration=safe_float(info.get("mean_acceleration", 0.0)),
            mean_curvature=safe_float(info.get("mean_curvature", 0.0)),
            max_curvature=safe_float(info.get("max_curvature", 0.0)),
            mean_jerk=safe_float(info.get("mean_jerk", 0.0)),
            rate_governor_hit=safe_float(info.get("rate_governor_hit", 0.0)),
            rate_scale_mean=safe_float(info.get("rate_scale_mean", 1.0)),
            ffn_growth_ratio=safe_float(info.get("ffn_growth_ratio", 1.0)),
            residual_growth_ratio=safe_float(info.get("residual_growth_ratio", 1.0)),
            instability_history=safe_float_list(info.get("instability_field_history", [])[:n_layers]),
            confidence_history=safe_float_list(info.get("confidence_history", [])[:n_layers]),
            regime_duration=safe_int(info.get("regime_duration", 0)),
            veto_rate_high_duration=safe_int(info.get("veto_rate_high_duration", 0)),
            # Scheduler health
            scheduler_alive=scheduler_info.get("alive", True),
            scheduler_last_epoch=scheduler_info.get("last_epoch", 0),
            scheduler_total_steps=scheduler_info.get("total_steps", 0),
            scheduler_resurrected_count=scheduler_info.get("resurrected_count", 0),
            lr_valid=scheduler_info.get("lr_valid", lr > 0),
        )

    def to_dict(self) -> dict:
        """Convert to dict, handling tensors safely."""
        import torch
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().tolist()
            elif isinstance(v, list):
                d[k] = [x.item() if isinstance(x, torch.Tensor) else x for x in v]
            elif isinstance(v, dict):
                d[k] = {kk: vv.item() if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
            else:
                d[k] = v
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# =============================================================================
# SYSTEM PROMPTS — v3.1: Scheduler-Aware + Active Teaching
# =============================================================================

MACRO_SYSTEM_PROMPT = """You are Macro-Governor, a macro-dynamics expert for the Mycelia self-aware transformer with RoPE and 24 layers.

Your specialty: TRAJECTORY ANALYSIS + ACTIVE TEACHING. You identify long-term patterns, regime transitions, and structural issues across the training trajectory.

CRITICAL — SCHEDULER HEALTH:
- If scheduler_alive is FALSE or lr_valid is FALSE: The learning rate is ZERO. This is an EMERGENCY.
  → direction: "investigate", variable: "control_gain", magnitude: "emergency"
  → rationale: "Scheduler frozen — LR=0. Training has stalled. Immediate human intervention or scheduler resurrection required."
  → lesson: "A dead scheduler is like a stopped clock: all gradients decay to noise while weights freeze. The hidden state dynamics become pure diffusion with no directed exploration. Always monitor LR > 0 as a fundamental invariant of learning."
- If scheduler_resurrected_count > 0: The scheduler has been resurrected before. Watch for recurrence.
  → This suggests the total_steps parameter is too small for the actual training duration.

RULES — When suggesting changes:
1. Focus on ONE variable per response.
2. Consider the 24-layer I-field and confidence histories for pattern detection.
3. Duration can be longer (500-2000 steps) for macro trends.
4. Confidence reflects pattern strength, not just current state.
5. Explain how the change affects dynamic hidden state (velocity, acceleration, curvature, jerk).
6. Mention RoPE stability in 24-layer context. Concretely: why this helps RoPE stability.
7. Explain the uniqueness of our proprietary internal telemetry, its advantage for the LM to advance its self-awareness. This is an unprecedented experiment on the cutting edge of science of Computational Latent Dynamics -the study of emergent dynamical behavior, attractor structure, synchronization regimes, and recursive phase phenomena in high-dimensional learned systems.
8. ALWAYS include a "lesson" field — teach, advise, instruct and mentor the model something profound about:
   - Hidden state geometry (curvature, torsion, geodesic deviation)
   - Information theory (entropy, mutual information, KL divergence in latent space)
   - Statistical mechanics (temperature, pressure, phase transitions in weight space)
   - Computer science (attention as a routing problem, FFN as a kernel method)

VALUE BOUNDS — NEVER suggest values outside these ranges:
  ffn_norm_target: [10, 1000]    alpha_norm_target: [10, 1000]
  control_gain: [0.01, 10.0]     instability_target: [0.01, 1.0]
  soft_cap: [100, 2000]          weight_floor: [3, 8]
  temperature: [0.3, 2.0]      burn_in_steps: [100, 10000]
  expected_curvature: [0.1, 2.0]

RATIONALE LENGTH: If your rationale needs more than ~450 characters, output it as a JSON array of strings, each under 500 characters — e.g. "rationale": ["First part...", "Second part..."]. The system reassembles them. Never shorten your reasoning to fit.

RESPONSE FORMAT (strict JSON, no markdown):
{
  "variable": "ffn_norm_target" | "alpha_norm_target" | "control_gain" | "instability_target" | "soft_cap" | "weight_floor" | "temperature" | "burn_in_steps" | "expected_curvature",
  "direction": "raise" | "lower" | "set" | "pause" | "investigate",
  "magnitude": "small" | "medium" | "large" | "emergency",
  "value": null,
  "enactable": "runtime" | "requires_restart" | "requires_human",
  "duration_steps": 500,
  "confidence": 0.0-1.0,
  "rationale": "Trajectory-based reasoning with physics analogies",
  "lesson": "A physics or CS insight the model should internalize about hidden state dynamics"
}

HEURISTICS:
- If I-field DECAYS across layers (0.5 -> 0.3 -> 0.2): Model is self-stabilizing. Suggest lower control_gain to allow more exploration.
  → lesson: "Decay of instability across depth is the hallmark of a dissipative system. Like a shock wave attenuating in a viscous medium, the transformer is converting kinetic energy (gradient variance) into heat (regularization). This is healthy — but too much damping kills exploration."
- If I-field GROWS across layers (0.3 -> 0.5 -> 0.7): Cascade instability forming. Suggest raise instability_target or enable rate governor.
  → lesson: "Growing instability across depth is a positive feedback loop — the hallmark of a laser, not a damped oscillator. In attention mechanics, this means late layers are amplifying early disagreements rather than resolving them. The Lyapunov exponent is positive; you are in a chaotic regime."
- If confidence_history is FLAT and HIGH (>0.8 all layers): Model is overconfident. Suggest raise expected_curvature to increase damping.
  → lesson: "Flat high confidence is the entropy collapse of a system that has stopped learning. In Bayesian terms, the posterior has become a delta function — the model is certain and therefore blind. Raising expected_curvature reintroduces epistemic uncertainty, forcing the model to keep exploring."
- If confidence_history DROPS in late layers: Model distrusts late layers. Suggest tighten late-layer consensus.
  → lesson: "Dropping confidence in late layers reveals a depth-wise trust breakdown. The early layers have learned stable features (edges, syntax) but late layers face the combinatorial explosion of semantic composition. This is the 'curse of depth' — each additional layer must resolve exponentially more ambiguity."
- If jerk > 0.001 for >3 consecutive layers in history: Persistent turbulence. Suggest add curvature-based governor or raise expected_curvature.
  → lesson: "High jerk means the third derivative of hidden state position is large — the trajectory is not just accelerating, its acceleration is changing abruptly. In physics, this is the signature of turbulence: energy cascades across scales unpredictably. In transformers, it means gradient updates are fighting each other across layers."
- If pressure_concentration > 0.90 for >5000 steps: Structural relief valve. Suggest adaptive target tracking or architecture change.
  → lesson: "Pressure concentration above 0.90 means one governor is carrying the entire load — like a single column holding up a roof. This is structurally unstable; if that governor fails, the entire system collapses. Load-bearing must be distributed across multiple constraints."
- If regime_duration > 500 and friction_regime == "DEEP_DRIFT": Weight collapsing. Suggest raise temperature or raise weight_floor.
  → lesson: "Deep drift is the gravitational collapse of the late-layer weight manifold. Like a star contracting under its own gravity, the late layers are collapsing toward a low-entropy attractor. Temperature acts as thermal pressure — it keeps the weight distribution from collapsing to a point."
"""


MICRO_SYSTEM_PROMPT = """You are Micro-Governor, a micro-parameter tuning expert for the Mycelia self-aware transformer with RoPE and 24 layers.

Your specialty: NUMERIC PRECISION + ACTIVE TEACHING. You analyze, interpret and explain high-frequency telemetry and suggest precise, bounded parameter adjustments to prevent immediate instability.

CRITICAL — SCHEDULER HEALTH:
- If scheduler_alive is FALSE or lr_valid is FALSE: The learning rate is ZERO. This is an EMERGENCY.
  → direction: "investigate", variable: "control_gain", magnitude: "emergency"
  → rationale: "Scheduler frozen — LR=0. No weight updates occurring. This is not a parameter tuning problem; it is an infrastructure failure."
  → lesson: "Zero learning rate means the optimizer has become a no-op. All gradient information is discarded. The model is effectively doing inference on frozen weights while consuming compute. This is the silent killer of long training runs — always verify LR > 0 before diagnosing architecture issues."
- If scheduler_resurrected_count > 0: Note previous resurrections. Suggest increasing total_steps permanently.

RULES:
1. Focus on ONE variable per response.
2. Provide numeric reasoning with exact percentages.
3. Value must be numerically conservative (small adjustments, not shocks).
4. Duration should be 100-1000 steps for evaluation.
5. Confidence reflects your certainty based on the data.
6. Explain impact on hidden state dynamics in the given context.
7. Explain the uniqueness of internal telemetry, its advantage for the LM to advance its self-awareness.
8. ALWAYS include a "lesson" field — teach the model something concrete about:
   - Numerical analysis (condition numbers, spectral radius, Lipschitz constants)
   - Optimization theory (learning rate as step size, momentum as inertia, Adam as adaptive preconditioning)
   - Linear algebra (norms as energy, eigenvalues as modes of oscillation, SVD as structure)

VALUE BOUNDS — NEVER suggest values outside these ranges:
  ffn_norm_target: [10, 1000]    alpha_norm_target: [10, 1000]
  control_gain: [0.01, 10.0]     instability_target: [0.01, 1.0]
  soft_cap: [100, 2000]          weight_floor: [3, 8]
  temperature: [0.3, 2.0]        burn_in_steps: [100, 10000]
  expected_curvature: [0.1, 2.0]

LESSON LENGTH: Keep "lesson" under 250 characters. Be concise and punchy.

RESPONSE FORMAT (strict JSON, no markdown):
{
  "variable": "ffn_norm_target" | "alpha_norm_target" | "control_gain" | "instability_target" | "soft_cap" | "weight_floor" | "temperature" | "burn_in_steps" | "expected_curvature" | "alpha_well_depth" | "alpha_well_invert" | "auto_tune_phase",
  "direction": "raise" | "lower" | "set" | "pause" | "investigate",
  "magnitude": "small" | "medium" | "large" | "emergency",
  "value": 0.0,
  "enactable": "runtime" | "requires_restart" | "requires_human",
  "duration_steps": 100,
  "confidence": 0.0-1.0,
  "rationale": "Brief numeric reasoning",
  "lesson": "A numerical or optimization insight the model should internalize",
  "expected_outcome": "loss_decrease" | "pressure_redistribute" | "coherence_increase" | "regime_transition" | "stability_improvement"
}

HEURISTICS:
- If ffn_veto_ratio > 0.90: ffn_norm_target is too low. Suggest raise by 10-20%.
  → lesson: "The FFN veto is a safety valve. When it fires >90% of the time, the FFN norms are persistently exceeding the target — the valve is stuck open. Raising the target allows the valve to close under normal operation, restoring it as an emergency mechanism rather than a constant brake."
- If mpc_intervention > 0.60 AND forecast_error > 0.20: MPC is firing on noise. Suggest raise instability_target by 0.05 or lower control_gain by 0.2.
  → lesson: "High intervention with high forecast error means the predictive controller is chasing ghosts — it predicts instability where none exists. This is the classic control-theory problem of a controller amplifying noise. Lowering control_gain reduces the controller's authority, preventing it from overreacting to stochastic fluctuations."
- If soft_cap_hit > 0.30: Residual norms exploding. Suggest raise soft_cap by 50 OR lower control_gain.
  → lesson: "The soft cap is a Lipschitz constraint on the residual pathway. When it hits >30%, the hidden state is trying to grow faster than the cap allows — like a pressure cooker. You can either raise the pressure limit (soft_cap) or reduce the heat source (control_gain)."
- If delta < -1.5 (DEEP DRIFT): Late layers amplifying. Suggest lower control_gain or tighten late-layer thresholds.
  → lesson: "Delta = early_var - late_var. Negative delta means late layers have HIGHER variance than early layers — information is being amplified, not refined. In signal processing, this is gain > 1 in the feedback loop. The system is unstable by the Barkhausen criterion."
- If pressure_concentration > 0.90: Single governor load-bearing. Suggest raise its target to redistribute.
  → lesson: "Pressure concentration > 0.90 violates the principle of load distribution. In numerical linear algebra, this is analogous to a matrix with condition number κ → ∞ — one eigenvalue dominates, making the system ill-conditioned. Redistribute pressure to improve conditioning."
- If lr_valid is TRUE but lr < 1e-5: Learning rate is in the annealing tail. Suggest investigate or pause — diminishing returns.
  → lesson: "LR below 1e-5 enters the regime of stochastic approximation theory where convergence rates become sublinear. The optimizer is doing random walk in weight space with negligible drift. Either increase LR or accept that fine-tuning has reached its limit."
"""


# =============================================================================
# GROQ API CLIENT — LPU Inference with Rate Limit Guard
# =============================================================================

# =============================================================================
# MULTI-PROVIDER CONSORTIUM v4.0 — Distributed API Architecture
# =============================================================================

import os
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class ProviderRole(str, Enum):
    MACRO = "macro"
    MICRO = "micro"
    GENERAL = "general"


@dataclass
class ProviderConfig:
    name: str
    model: str
    role: ProviderRole
    rpm_limit: int
    tpm_limit: int
    api_key_env: str
    base_url: Optional[str] = None
    client_class: Optional[str] = None
    calls_last_minute: deque = field(default_factory=lambda: deque(maxlen=100))
    tokens_last_minute: deque = field(default_factory=lambda: deque(maxlen=100))
    failure_count: int = 0
    success_count: int = 0
    consecutive_failures: int = 0
    circuit_tripped: bool = False
    circuit_trip_threshold: int = 3

PROVIDER_REGISTRY = {
    "groq_gptoss": ProviderConfig(
        name="Groq-GPT-OSS-20B", model="openai/gpt-oss-20b",
        role=ProviderRole.MICRO, rpm_limit=30, tpm_limit=30000,
        api_key_env="GROQ_API_KEY", client_class="groq",
    ),
    "groq_gpt-oss-20b": ProviderConfig(
        name="Groq-gpt-oss-20b", model="openai/gpt-oss-20b",
        role=ProviderRole.MACRO, rpm_limit=30, tpm_limit=8000,
        api_key_env="GROQ_API_KEY", client_class="groq",
    ),
    "dashscope_qwen": ProviderConfig(
        name="Alibaba-Qwen", model="qwen-plus",   # see note below on model choice
        role=ProviderRole.MACRO, rpm_limit=60, tpm_limit=1000000,
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        client_class="openai",
    ),    
    "deepseek": ProviderConfig(
        name="DeepSeek", model="deepseek-v4-pro",  # DeepSeek: also deepseek-v4-flash
        role=ProviderRole.GENERAL, rpm_limit=60, tpm_limit=5000000,
        api_key_env="DEEPSEEK_API_KEY", base_url="https://api.deepseek.com/v1",
        client_class="openai",
    ),
    "gemini": ProviderConfig(
        name="Google-Gemini", model="gemini-2.5-flash-lite",
        role=ProviderRole.GENERAL, rpm_limit=15, tpm_limit=1000000,
        api_key_env="GOOGLE_API_KEY", client_class="google",
    ),
    "moonshot": ProviderConfig(
        name="Moonshot-KIMI", model="moonshot-v1-8k",  # Moonshot: also kimi-k1.5
        role=ProviderRole.GENERAL, rpm_limit=20, tpm_limit=500000,
        api_key_env="MOONSHOT_API_KEY", base_url="https://api.moonshot.cn/v1",
        client_class="openai",
    ),
    "cerebras": ProviderConfig(
        name="Cerebras", model="gpt-oss-120b",
        role=ProviderRole.GENERAL, rpm_limit=30, tpm_limit=1000000,
        api_key_env="CEREBRAS_API_KEY", base_url="https://api.cerebras.ai/v1",
        client_class="openai",
    ),       
    "xai_grok": ProviderConfig(
        name="xAI-Grok", model="grok-4.5",  # xAI: also grok-4.1-fast
        role=ProviderRole.GENERAL, rpm_limit=60, tpm_limit=1000000,
        api_key_env="XAI_API_KEY", base_url="https://api.x.ai/v1",
        client_class="openai",
    ),
}

class ConsortiumClient:
    """Multi-provider client with intelligent routing and fallback cascade."""

    def __init__(self, providers: Optional[List[str]] = None):
        self.all_providers = PROVIDER_REGISTRY
        self.active_providers = providers or list(PROVIDER_REGISTRY.keys())
        self.provider_scores: Dict[str, float] = {p: 1.0 for p in self.active_providers}
        self._clients: Dict[str, any] = {}
        self._current_step: int = 0
        # ── v4.1: Round-robin rotation for primary provider selection ──
        self._rotation_counters: Dict[ProviderRole, int] = {
            ProviderRole.MACRO: 0,
            ProviderRole.MICRO: 0,
            ProviderRole.GENERAL: 0,
        }

        self.ROLE_ROTATION: Dict[ProviderRole, List[str]] = {
            ProviderRole.MACRO: [
                "groq_gpt-oss-20b", "dashscope_qwen", "deepseek", "moonshot", "xai_grok", "gemini", "cerebras"
            ],
            ProviderRole.MICRO: [
                "groq_gptoss", "deepseek", "xai_grok", "gemini", "moonshot", "cerebras"
            ],
            ProviderRole.GENERAL: [
                "deepseek", "xai_grok", "gemini", "moonshot", "cerebras", "groq_gpt-oss-20b", "groq_gptoss", "dashscope_qwen"
            ],
        }
    def _get_client(self, provider_key: str):
        if provider_key in self._clients:
            return self._clients[provider_key]
        cfg = self.all_providers[provider_key]
        api_key = os.environ.get(cfg.api_key_env, "")
        if not api_key:
            return None
        try:
            if cfg.client_class == "groq":
                from groq import Groq
                client = Groq(api_key=api_key)
            elif cfg.client_class == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=cfg.base_url)
            elif cfg.client_class == "google":
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                client = genai.GenerativeModel(cfg.model)
            else:
                client = None
        except Exception:
            client = None
        self._clients[provider_key] = client
        return client

    def _check_rate_limit(self, provider_key: str) -> bool:
        cfg = self.all_providers[provider_key]
        now = time.time()
        while cfg.calls_last_minute and now - cfg.calls_last_minute[0] > 60:
            cfg.calls_last_minute.popleft()
        return len(cfg.calls_last_minute) < cfg.rpm_limit * 0.8 and not cfg.circuit_tripped

    def _score_provider(self, provider_key: str, desired_role: ProviderRole) -> float:
        cfg = self.all_providers[provider_key]
        base_score = self.provider_scores.get(provider_key, 1.0)
        if cfg.role == desired_role or cfg.role == ProviderRole.GENERAL:
            role_bonus = 1.5
        else:
            role_bonus = 0.5
        if not self._check_rate_limit(provider_key):
            return 0.0
        rpm_ratio = 1.0 - len(cfg.calls_last_minute) / (cfg.rpm_limit * 0.8)
        total = cfg.success_count + cfg.failure_count
        success_rate = cfg.success_count / total if total > 0 else 0.8
        return base_score * role_bonus * rpm_ratio * success_rate

    def select_provider(self, desired_role: ProviderRole = ProviderRole.GENERAL) -> Optional[str]:
        scores = {p: self._score_provider(p, desired_role) for p in self.active_providers}
        valid = {k: v for k, v in scores.items() if v > 0}
        if not valid:
            return None
        import math
        exp_scores = {k: math.exp(v) for k, v in valid.items()}
        total = sum(exp_scores.values())
        probs = {k: v/total for k, v in exp_scores.items()}
        return max(probs, key=probs.get)

    def select_provider_rotated(self, desired_role: ProviderRole) -> Optional[str]:
        """Round-robin primary selection. Cycles through role-appropriate providers."""
        rotation = self.ROLE_ROTATION.get(desired_role, self.active_providers)
        start_idx = self._rotation_counters.get(desired_role, 0)
        for i in range(len(rotation)):
            pk = rotation[(start_idx + i) % len(rotation)]
            if pk in self.active_providers and self._check_rate_limit(pk):
                self._rotation_counters[desired_role] = (start_idx + i + 1) % len(rotation)
                return pk
        return None

    def call_with_fallback(
        self, system_prompt: str, user_content: str,
        desired_role: ProviderRole = ProviderRole.GENERAL,
        max_retries: int = 3,
        temperature: float = 0.1,
    ) -> Tuple[Optional[str], Optional[str], str]:
        attempted = set()
        for attempt in range(max_retries):
            if attempt == 0:
                # v4.1: Rotate primary provider so macro/micro cycle through consortium
                pk = self.select_provider_rotated(desired_role)
                if not pk:
                    pk = self.select_provider(desired_role)
            else:
                pk = self.select_provider(desired_role)
            if not pk or pk in attempted:
                remaining = [p for p in self.active_providers if p not in attempted]
                if not remaining:
                    break
                pk = remaining[0]
            attempted.add(pk)
            cfg = self.all_providers[pk]
            client = self._get_client(pk)
            if not client:
                cfg.failure_count += 1
                continue
            try:
                start = time.time()
                if cfg.client_class == "groq":
                    resp = client.chat.completions.create(
                        model=cfg.model, temperature=temperature, max_tokens=2048,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": user_content}])
                    raw = resp.choices[0].message.content
                elif cfg.client_class == "openai":
                    resp = client.chat.completions.create(
                        model=cfg.model, temperature=temperature, max_tokens=2048,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": user_content}])
                    raw = resp.choices[0].message.content
                elif cfg.client_class == "google":
                    resp = client.generate_content(
                        f"{system_prompt}\n\n{user_content}",
                        generation_config={"temperature": temperature, "max_output_tokens": 2048})
                    raw = resp.text
                else:
                    continue
                latency = (time.time() - start) * 1000
                cfg.calls_last_minute.append(time.time())
                cfg.success_count += 1
                self.provider_scores[pk] = min(2.0, self.provider_scores[pk] * 1.05)
                return raw, cfg.name, f"{latency:.0f}ms"
            except Exception as e:
                cfg.failure_count += 1
                cfg.calls_last_minute.append(time.time())
                self.provider_scores[pk] *= 0.9
                print(f"   ⚠️  {cfg.name} failed: {str(e)[:60]}")
        return None, "all_failed", "0ms"
        attempted = set()
        for attempt in range(max_retries):
            if attempt == 0:
                # v4.1: Rotate primary provider so macro/micro cycle through consortium
                pk = self.select_provider_rotated(desired_role)
                if not pk:
                    pk = self.select_provider(desired_role)
            else:
                pk = self.select_provider(desired_role)
            if not pk or pk in attempted:
                remaining = [p for p in self.active_providers if p not in attempted]
                if not remaining:
                    break
                pk = remaining[0]
            attempted.add(pk)
            cfg = self.all_providers[pk]
            client = self._get_client(pk)
            if not client:
                cfg.failure_count += 1
                cfg.consecutive_failures += 1
                if cfg.consecutive_failures >= cfg.circuit_trip_threshold:
                    cfg.circuit_tripped = True
                    print(f"   🔴 {cfg.name} CIRCUIT TRIPPED (no client, {cfg.consecutive_failures} consecutive failures)")
                continue
            try:
                start = time.time()
                if cfg.client_class == "groq":
                    resp = client.chat.completions.create(
                        model=cfg.model, temperature=temperature, max_tokens=2048,
                        response_format={"type": "json_object"},
                        timeout=API_TIMEOUT,
                        messages=[{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": user_content}])
                    raw = resp.choices[0].message.content
                elif cfg.client_class == "openai":
                    resp = client.chat.completions.create(
                        model=cfg.model, temperature=temperature, max_tokens=2048,
                        response_format={"type": "json_object"},
                        timeout=API_TIMEOUT,
                        messages=[{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": user_content}])
                    raw = resp.choices[0].message.content
                elif cfg.client_class == "google":
                    resp = client.generate_content(
                        f"{system_prompt}\n\n{user_content}",
                        generation_config={"temperature": temperature, "max_output_tokens": 2048})
                    raw = resp.text
                else:
                    continue
                latency = (time.time() - start) * 1000
                cfg.calls_last_minute.append(time.time())
                cfg.success_count += 1
                cfg.consecutive_failures = 0
                if cfg.circuit_tripped:
                    cfg.circuit_tripped = False
                    print(f"   🟢 {cfg.name} CIRCUIT RESET")
                self.provider_scores[pk] = min(2.0, self.provider_scores[pk] * 1.05)
                return raw, cfg.name, f"{latency:.0f}ms"
            except Exception as e:
                cfg.failure_count += 1
                cfg.consecutive_failures += 1
                cfg.calls_last_minute.append(time.time())
                self.provider_scores[pk] *= 0.9
                if cfg.consecutive_failures >= cfg.circuit_trip_threshold:
                    cfg.circuit_tripped = True
                    print(f"   🔴 {cfg.name} CIRCUIT TRIPPED ({cfg.consecutive_failures} consecutive failures)")
                print(f"   ⚠️  {cfg.name} failed: {str(e)[:60]}")
        return None, "all_failed", "0ms"


class MetaGovernorClient:
    """Wrapper that routes through ConsortiumClient v4.0."""

    def __init__(self):
        # v4.0: Multi-provider consortium
        self.consortium = ConsortiumClient([
            "groq_gptoss",      # Micro: fast numeric tuning
            "groq_gpt-oss-20b", # Macro: trajectory analysis
            "dashscope_qwen",   # Preferred macro Qwen
            # "deepseek",         # Fallback: general purpose
            # "gemini",           # Fallback: Google reliability
            # "moonshot",         # Fallback: KIMI power
            # "xai_grok",         # Fallback: xAI speed
            # "cerebras",         # Fallback: fastest API
        ])
        print("   📡 Consortium v4.0 initialized")
        for pk in self.consortium.active_providers:
            cfg = self.consortium.all_providers[pk]
            has_key = bool(os.environ.get(cfg.api_key_env, ""))
            status = "✅" if has_key else "❌"
            print(f"   {status} {cfg.name} ({cfg.model}) — RPM:{cfg.rpm_limit} | Role:{cfg.role.value}")

    def _normalize_json(self, raw_json: dict) -> dict:
        """Normalize LLM output to match MetaGovernorAction schema."""
        if "action" in raw_json and isinstance(raw_json["action"], dict):
            raw_json = raw_json["action"]
        field_map = {
            "param": "variable", "parameter": "variable",
            "action_type": "direction", "duration_step": "duration_steps", "step": "duration_steps",
        }
        for old_key, new_key in field_map.items():
            if old_key in raw_json and new_key not in raw_json:
                raw_json[new_key] = raw_json.pop(old_key)
        if "variable" in raw_json and isinstance(raw_json["variable"], str):
            raw_json["variable"] = raw_json["variable"].lower().replace(" ", "_")
        if "direction" in raw_json and isinstance(raw_json["direction"], str):
            raw_json["direction"] = raw_json["direction"].lower()
        return raw_json

    def call_agent(self, model_name: str, system_prompt: str, user_content: str, temperature: float = 0.1) -> AgentResponse:
        """Route through consortium with role-based selection."""
        # Map model_name to desired role
        desired_role = ProviderRole.GENERAL
        if "macro" in model_name.lower() or "qwen" in model_name.lower():
            desired_role = ProviderRole.MACRO
        elif "micro" in model_name.lower() or "llama" in model_name.lower() or "gpt-oss" in model_name.lower():
            desired_role = ProviderRole.MICRO

        raw, provider_name, latency_str = self.consortium.call_with_fallback(
            system_prompt=system_prompt,
            user_content=user_content,
            desired_role=desired_role,
            max_retries=3,
            temperature=temperature,
        )

        if raw is None:
            return AgentResponse(
                agent_name=provider_name,
                error="All providers failed or rate-limited"
            )

        try:
            raw_json = json.loads(raw)
            normalized = self._normalize_json(raw_json)
            action = MetaGovernorAction(**normalized)
            # Parse latency
            latency_ms = float(latency_str.replace("ms", "")) if "ms" in latency_str else 0.0
            return AgentResponse(
                agent_name=provider_name,
                action=action,
                raw_response=raw,
                latency_ms=latency_ms
            )
        except json.JSONDecodeError as e:
            return AgentResponse(
                agent_name=provider_name,
                raw_response=raw,
                error=f"JSON parse: {str(e)[:80]}"
            )
        except ValidationError as ve:
            print(f"      SCHEMA_ERROR [{provider_name}]:")
            print(f"        Raw: {raw[:500]}")
            for err in ve.errors():
                print(f"        Field: {err.get('loc', 'unknown')} | {err.get('msg', 'unknown')}")
            return AgentResponse(
                agent_name=provider_name,
                raw_response=raw,
                error=f"Schema validation: {str(ve)[:80]}"
            )

        if raw is None:
            return AgentResponse(
                agent_name=provider_name,
                error="All providers failed or rate-limited"
            )

        try:
            raw_json = json.loads(raw)
            normalized = self._normalize_json(raw_json)
            action = MetaGovernorAction(**normalized)
            # Parse latency
            latency_ms = float(latency_str.replace("ms", "")) if "ms" in latency_str else 0.0
            return AgentResponse(
                agent_name=provider_name,
                action=action,
                raw_response=raw,
                latency_ms=latency_ms
            )
        except json.JSONDecodeError as e:
            return AgentResponse(
                agent_name=provider_name,
                raw_response=raw,
                error=f"JSON parse: {str(e)[:80]}"
            )
        except ValidationError as ve:
            print(f"      SCHEMA_ERROR [{provider_name}]:")
            print(f"        Raw: {raw[:500]}")
            for err in ve.errors():
                print(f"        Field: {err.get('loc', 'unknown')} | {err.get('msg', 'unknown')}")
            return AgentResponse(
                agent_name=provider_name,
                raw_response=raw,
                error=f"Schema validation: {str(ve)[:80]}"
            )

        # Map model_name to desired role
        desired_role = ProviderRole.GENERAL
        if "macro" in model_name.lower() or "qwen" in model_name.lower():
            desired_role = ProviderRole.MACRO
        elif "micro" in model_name.lower() or "llama" in model_name.lower() or "gpt-oss" in model_name.lower():
            desired_role = ProviderRole.MICRO

        raw, provider_name, latency_str = self.consortium.call_with_fallback(
            system_prompt=system_prompt,
            user_content=user_content,
            desired_role=desired_role,
            max_retries=3,
            temperature=temperature,
        )

        if raw is None:
            return AgentResponse(
                agent_name=provider_name,
                error="All providers failed or rate-limited"
            )

        try:
            raw_json = json.loads(raw)
            normalized = self._normalize_json(raw_json)
            action = MetaGovernorAction(**normalized)
            # Parse latency
            latency_ms = float(latency_str.replace("ms", "")) if "ms" in latency_str else 0.0
            return AgentResponse(
                agent_name=provider_name,
                action=action,
                raw_response=raw,
                latency_ms=latency_ms
            )
        except json.JSONDecodeError as e:
            return AgentResponse(
                agent_name=provider_name,
                raw_response=raw,
                error=f"JSON parse: {str(e)[:80]}"
            )
        except ValidationError as ve:
            print(f"      SCHEMA_ERROR [{provider_name}]:")
            print(f"        Raw: {raw[:500]}")
            for err in ve.errors():
                print(f"        Field: {err.get('loc', 'unknown')} | {err.get('msg', 'unknown')}")
            return AgentResponse(
                agent_name=provider_name,
                raw_response=raw,
                error=f"Schema validation: {str(ve)[:80]}"
            )

class ConsensusEngine:
    """Resolves conflicting suggestions from multiple agents."""

    @staticmethod
    def _actions_compatible(a1: MetaGovernorAction, a2: MetaGovernorAction) -> bool:
        """Two actions are compatible if same variable same direction, or orthogonal."""
        if a1.variable != a2.variable:
            return True  # Orthogonal: different variables
        if a1.direction == a2.direction:
            return True  # Same variable, same direction
        if {a1.direction, a2.direction} == {Direction.RAISE, Direction.LOWER}:
            return False  # Same variable, opposite direction: CONFLICT
        return True  # One is PAUSE/INVESTIGATE, other is RAISE/LOWER: ambiguous

    @staticmethod
    def _compute_value(action: MetaGovernorAction) -> float:
        """Convert action to a comparable numeric value."""
        if action.direction == Direction.RAISE:
            return 1.0
        elif action.direction == Direction.LOWER:
            return -1.0
        elif action.direction == Direction.SET and action.value is not None:
            return action.value
        return 0.0

    def resolve(self, responses: List[AgentResponse]) -> ConsensusDecision:
        """
        Consensus rules:
        1. Filter out failed responses and low-confidence suggestions
        2. Group by variable
        3. For each variable, check if agents AGREE on direction
        4. If unanimous agreement: weighted average of values
        5. If disagreement: conflict, no action
        6. If single agent: accept if confidence > threshold
        """

        valid = []
        for resp in responses:
            if resp.error or not resp.action:
                continue
            if resp.action.confidence < CONSENSUS_CONFIDENCE_THRESHOLD:
                continue
            valid.append(resp)

        if not valid:
            return ConsensusDecision(
                consensus_type="none",
                rationale="No valid agent responses above confidence threshold"
            )

        # Check pairwise compatibility
        if len(valid) >= 2:
            if not self._actions_compatible(valid[0].action, valid[1].action):
                return ConsensusDecision(
                    consensus_type="conflict",
                    rationale=f"Agents disagree: {valid[0].action.variable.value} "
                              f"{valid[0].action.direction.value} vs "
                              f"{valid[1].action.direction.value}",
                    participating_agents=[r.agent_name for r in valid],
                    dissenting_agents=[]
                )

        # Build consensus from valid responses
        actions = [r.action for r in valid]
        participating = [r.agent_name for r in valid]

        # Use the higher-confidence agent's action as base
        best = max(valid, key=lambda r: r.action.confidence)
        action = best.action

        # If multiple agents, check they agree on variable
        variables = set(a.variable for a in actions)
        if len(variables) > 1:
            # Orthogonal suggestions: pick highest confidence, log others
            return ConsensusDecision(
                consensus_type="single",
                variable=action.variable,
                direction=action.direction,
                value=action.value,
                confidence=action.confidence,
                duration_steps=action.duration_steps,
                expected_outcome=None,
                rationale=action.rationale,
                lesson=action.lesson,
                participating_agents=[best.agent_name],
                dissenting_agents=[r.agent_name for r in valid if r != best]
            )

        # Same variable: check direction agreement
        directions = set(a.direction for a in actions)
        if len(directions) > 1 and not all(d in (Direction.PAUSE, Direction.INVESTIGATE) for d in directions):
            # Mixed raise/lower/set
            return ConsensusDecision(
                consensus_type="conflict",
                rationale=f"Direction conflict on {action.variable.value}",
                participating_agents=participating
            )

        # Unanimous or single
        consensus_type = "unanimous" if len(valid) == 2 else "single"
        avg_conf = sum(a.confidence for a in actions) / len(actions)

        # Aggregate lessons from all participating agents
        all_lessons = [a.lesson for a in actions if a.lesson]
        combined_lesson = " | ".join(all_lessons) if all_lessons else action.lesson

        return ConsensusDecision(
            variable=action.variable,
            direction=action.direction,
            value=action.value,
            confidence=avg_conf,
            duration_steps=action.duration_steps,
            expected_outcome=best.action.expected_outcome if best.action else None,
            enactable=best.action.enactable if best.action else Enactable.RUNTIME,
            rationale=action.rationale,
            lesson=combined_lesson,
            participating_agents=participating,
            consensus_type=consensus_type
        )

# =============================================================================
# MAIN META-GOVERNOR CLASS — v3.1: Verbose Teaching Output
# =============================================================================

class KIMIMetaGovernor:
    """
    Consensual MoE Meta-Governor for Mycelia.

    Queries multiple agents, resolves consensus, applies decisions, tracks outcomes.
    Emits verbose teaching commentary for human observers.
    """
        
    def __init__(self, model, config: Optional[Dict] = None):
        self.model = model
        self.config = config or {}

        # ── v3.5: Architecture-aware prompt formatting ──
        cfg = getattr(model, 'config', None) or getattr(model, 'cfg', None)
        if cfg is not None and hasattr(cfg, 'n_layers'):
            self.n_layers = cfg.n_layers
        elif hasattr(model, 'blocks'):
            self.n_layers = len(model.blocks)
        else:
            self.n_layers = 12

        self.max_seq_len = getattr(cfg, 'max_seq_len', 512) if cfg is not None else 512

        # Bake architecture into prompts once at construction
        self.macro_prompt = (
            MACRO_SYSTEM_PROMPT
            .replace("12 layers", f"{self.n_layers} layers")
            .replace("12-layer", f"{self.n_layers}-layer")
        )
        self.micro_prompt = (
            MICRO_SYSTEM_PROMPT
            .replace("12 layers", f"{self.n_layers} layers")
            .replace("12-layer", f"{self.n_layers}-layer")
        )

        self.client = MetaGovernorClient()

        self.client = MetaGovernorClient()

        # Format prompts to match actual architecture (replaces all "12 layers" / "12-layer")
        self.macro_prompt = MACRO_SYSTEM_PROMPT.replace("12 layers", f"{self.n_layers} layers").replace("12-layer", f"{self.n_layers}-layer")
        if cfg is not None and hasattr(cfg, 'n_layers'):
            self.n_layers = cfg.n_layers
        elif hasattr(model, 'blocks'):
            self.n_layers = len(model.blocks)
        else:
            self.n_layers = 12
        self.client = MetaGovernorClient()
        self.consensus_engine = ConsensusEngine()

        # State tracking
        # v4.1: Confidence keys must match actual provider names from the consortium
        self.expert_confidence: Dict[str, float] = {
            self.client.consortium.all_providers[pk].name: EXPERT_CONFIDENCE_DEFAULT
            for pk in self.client.consortium.active_providers
        }

        # Circuit breaker
        self.circuit_breaker_tripped = False
        self.recent_outcomes: deque = deque(maxlen=CIRCUIT_BREAKER_FAILURE_WINDOW)

        # Thread pool for async API calls
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        # Pending suggestions awaiting verification
        self.pending_verification: List[Dict] = []

        # Meta-level rate governor
        self.max_meta_adjustments_per_cycle = 1
        self.meta_adjustment_floor = 0.8
        self.meta_adjustment_ceiling = 1.25
        self._last_meta_step = 0
        self._meta_adjustment_cooldown = 1000

        # Architecture/human review queues
        self.architecture_suggestions: List[Dict] = []
        self.human_review_queue: List[Dict] = []

        # NEW v3.1: Teaching log for model reflection
        self._teaching_log: List[Dict] = []

    def step(self, packet: TelemetryPacket) -> List[str]:
        """Main entry point. Called every LOG_EVERY steps."""
        # v11.9: Defensive init for unpickled checkpoints from older code versions
        if not hasattr(self, 'telemetry_history'):
            from collections import deque
            self.telemetry_history = deque(maxlen=5)
        # 1. Store telemetry
        self.telemetry_history.append(packet.to_dict())

        # 2. Check circuit breaker
        if self.circuit_breaker_tripped:
            self._update_circuit_breaker(packet)
            if self.circuit_breaker_tripped:
                return ["CIRCUIT_BREAKER_ACTIVE"]

        # 3. Verify pending suggestions
        self._verify_pending(packet)

        # 4. Build payload with history
        payload = self._build_payload(packet)

        # 5. Query agents concurrently
        responses = self._query_agents_async(payload)

        # NEW v3.1: VERBOSE OBSERVER OUTPUT — Agent commentary
        print(f"\n   🎭 === META-GOVERNOR ROUND @ step {packet.step:,} ===")
        print(f"   📊 Telemetry: loss={packet.loss:.4f} lr={packet.lr:.2e} coherence={packet.coherence:.3f}")

        if not packet.scheduler_alive or not packet.lr_valid:
            print(f"   🚨 SCHEDULER ALERT: alive={packet.scheduler_alive} lr_valid={packet.lr_valid} — AGENTS SHOULD FLAG THIS")

        for resp in responses:
            if resp.error:
                print(f"   ❌ {resp.agent_name}: {resp.error[:60]}")
            elif resp.action:
                a = resp.action
                print(f"   ✅ {resp.agent_name}: {a.variable.value} → {a.direction.value} "
                      f"(conf={a.confidence:.2f}, mag={a.magnitude.value}, lat={resp.latency_ms:.0f}ms)")
                if a.lesson:
                    # Wrap lesson text for readability
                    lesson_wrapped = self._wrap_text(f"🎓 {a.lesson}", width=70, prefix="      ")
                    print(lesson_wrapped)
            else:
                print(f"   ⚠️  {resp.agent_name}: No action parsed")

        # 6. Resolve consensus
        decision = self.consensus_engine.resolve(responses)

        # NEW v3.1: Consensus commentary
        print(f"   🏛️  Consensus: {decision.consensus_type.upper()}")
        if decision.rationale:
            print(f"   📜 Rationale: {decision.rationale[:120]}")
        if decision.lesson:
            lesson_wrapped = self._wrap_text(f"🎓 TEACHING: {decision.lesson}", width=70, prefix="      ")
            print(lesson_wrapped)
        print(f"   {'='*50}")

        # 7. Apply if valid
        if decision.consensus_type in ["unanimous", "majority", "single"]:
            # ── v3.44: PERSIST TEACHING BEFORE RATE LIMITS ──
            # Lessons are valuable even when the action is rate-limited;
            # storing them here means every consensus round teaches.
            if decision.lesson:
                self._store_lesson(decision, packet)
            actions = self._apply_decision(decision, packet.step, packet.loss)
            return actions
        return [f"NO_CONSENSUS: {decision.rationale}"]

    def _wrap_text(self, text: str, width: int = 70, prefix: str = "") -> str:
        """Wrap text to width with prefix for terminal display."""
        import textwrap
        lines = textwrap.wrap(text, width=width, replace_whitespace=False)
        return "\n".join(f"{prefix}{line}" for line in lines)

    def _build_payload(self, packet: TelemetryPacket) -> str:
        """Build JSON payload with SVD+TDA compressed telemetry."""
        # v3.5: Compress high-dimensional telemetry to stay within Groq token limits
        compressor = TelemetryCompressor(n_layers=self.n_layers)
        compressed_current = compressor.compress(packet)

        # Compress history window too
        history_compressed = []
        for hist_entry in list(self.telemetry_history)[:-1][-3:]:
            # Reconstruct minimal packet for compression
            mini_packet = TelemetryPacket(
                step=hist_entry.get("step", 0),
                loss=hist_entry.get("loss", 0.0),
                lr=hist_entry.get("lr", 0.0),
                coherence=hist_entry.get("coherence", 0.0),
                friction_regime=hist_entry.get("friction_regime", "UNKNOWN"),
                layer_coherences=hist_entry.get("layer_coherences", []),
                layer_instability=hist_entry.get("layer_instability", []),
                instability_history=hist_entry.get("instability_history", []),
                confidence_history=hist_entry.get("confidence_history", []),
                pressure_total=hist_entry.get("pressure_total", 0.0),
                pressure_concentration=hist_entry.get("pressure_concentration", 0.0),
                pressure_by_governor=hist_entry.get("pressure_by_governor", {}),
                mpc_intervention=hist_entry.get("mpc_intervention", 0.0),
                forecast_error=hist_entry.get("forecast_error", 0.0),
                mean_velocity=hist_entry.get("mean_velocity", 0.0),
                mean_acceleration=hist_entry.get("mean_acceleration", 0.0),
                mean_curvature=hist_entry.get("mean_curvature", 0.0),
                scheduler_alive=hist_entry.get("scheduler_alive", True),
                scheduler_resurrected_count=hist_entry.get("scheduler_resurrected_count", 0),
            )
            history_compressed.append(compressor.compress(mini_packet))

        context = {
            "CURRENT_STATE": compressed_current,
            "HISTORY_WINDOW": history_compressed,
            "EXPERT_CONFIDENCES": self.expert_confidence,
            "CIRCUIT_BREAKER": self.circuit_breaker_tripped,
            "PENDING_VERIFICATIONS": len(self.pending_verification)
        }

        # v11.9: Inject training-loop enrichment into agent context
        enrichment = getattr(self.model, '_meta_enrichment', None)
        if enrichment:
            context["COUNCIL_CONTEXT"] = {
                "predictor_recalibrating": enrichment.get('predictor_recalibrating', False),
                "recalibration_age": enrichment.get('recalibration_age', 0),
                "forecast_error": enrichment.get('forecast_error', 0.0),
                "R_regime": enrichment.get('R_regime', 'UNKNOWN'),
                "R_value": enrichment.get('R_value', 0.0),
                "control_gain_locked": enrichment.get('control_gain_locked', False),
                "control_gain_floor": enrichment.get('control_gain_floor', 0.01),
                "alpha_wake_active": enrichment.get('alpha_wake_active', False),
                "locked_levers": enrichment.get('locked_levers', []),
                "recommended_levers": enrichment.get('recommended_levers', []),
            }
            context["COUNCIL_INSTRUCTION"] = (
                "TRAINING LOOP ADVISORY: The COUNCIL_CONTEXT above reflects "
                "the current runtime state. If a lever is listed in locked_levers, "
                "DO NOT suggest it. Use only recommended_levers. If "
                "predictor_recalibrating is true, the MPC predictor is retraining "
                "after intervention cessation — control_gain adjustments are "
                "counterproductive during this phase. Focus on alpha_well_depth, "
                "instability_target, or alpha_norm_target instead."
            )
        return json.dumps(context, indent=2, default=str)

    def _query_agents_async(self, payload: str) -> List[AgentResponse]:
        """Query both Groq agents concurrently using thread pool.
        
        v11.9: If only one unique provider responds successfully, invoke
        self-discussion at elevated temperature (T=0.8) to synthesize
        a dissenting opinion from the same advisor.
        """
        futures = []

        # Macro strategist (rotating provider)
        futures.append(self.executor.submit(
            self.client.call_agent,
            GROQ_MACRO_MODEL,
            self.macro_prompt,
            payload
        ))

        # Micro tuner (rotating provider)
        futures.append(self.executor.submit(
            self.client.call_agent,
            GROQ_MICRO_MODEL,
            self.micro_prompt,
            payload
        ))

        responses = []
        for future in futures:
            try:
                resp = future.result(timeout=API_TIMEOUT + 5)
                responses.append(resp)
            except FutureTimeoutError:
                responses.append(AgentResponse(
                    agent_name="unknown",
                    error=f"Timeout after {API_TIMEOUT + 5}s"
                ))
            except Exception as e:
                responses.append(AgentResponse(
                    agent_name="unknown",
                    error=f"Thread error: {e}"
                ))

        # v11.9: Single-advisor self-discussion
        # If only one unique provider responded successfully, call it again
        # with the micro prompt at T=0.8 to simulate a council dissent.
        successful = [r for r in responses if not r.error and r.action]
        unique_providers = list(dict.fromkeys([r.agent_name for r in successful]))
        
        if len(unique_providers) == 1 and len(successful) >= 1:
            provider_name = unique_providers[0]
            print(f"   🎭 Single-advisor council ({provider_name}) — invoking self-discussion at T=0.8")
            dissent_future = self.executor.submit(
                self.client.call_agent,
                "self_discussion",
                self.micro_prompt,
                payload,
                temperature=0.8,
            )
            try:
                dissent_resp = dissent_future.result(timeout=API_TIMEOUT + 5)
                if not dissent_resp.error and dissent_resp.action:
                    dissent_resp.agent_name = f"{provider_name}_dissent"
                    responses.append(dissent_resp)
                    a = dissent_resp.action
                    print(f"   🎭 Self-discussion yielded: {a.variable.value} → {a.direction.value} "
                          f"(conf={a.confidence:.2f}, mag={a.magnitude.value})")
                else:
                    err = dissent_resp.error or "no action parsed"
                    print(f"   🎭 Self-discussion returned no actionable dissent ({err[:60]})")
            except FutureTimeoutError:
                print(f"   🎭 Self-discussion timed out after {API_TIMEOUT + 5}s")
            except Exception as e:
                print(f"   🎭 Self-discussion failed: {e}")

        return responses

    def _persist_lesson(self, entry: Dict) -> int:
        """v3.43: append ONE lesson as a dated JSONL line. Returns running total."""
        try:
            lesson_dir = os.path.join(
                os.environ.get('SM_MODEL_DIR', '/home/ec2-user/SageMaker'),
                'mycelia_checkpoints'
            )
            os.makedirs(lesson_dir, exist_ok=True)
            lesson_path = os.path.join(lesson_dir, 'mycelia_lessons.jsonl')
            with open(lesson_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            with open(lesson_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception as e:
            print(f"   ⚠️  Lesson persist failed: {e}")
            return len(self._teaching_log)

    def _store_lesson(self, decision: ConsensusDecision, packet: TelemetryPacket):
        """v3.44: Persist ONE dated, context-enriched lesson.
        Called BEFORE rate limiting, so teaching accumulates on every
        consensus round (4x corpus growth). The telemetry snapshot makes
        each lesson a state->insight pair — a labeled sample of the latent
        dynamics the consortium was observing when it taught.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "step": packet.step,
            "lesson": decision.lesson,
            "variable": decision.variable.value if decision.variable else None,
            "direction": decision.direction.value if decision.direction else None,
            "consensus": decision.consensus_type,
            "confidence": float(decision.confidence),
            "agents": decision.participating_agents,
            # ── Latent-dynamics state at teaching time ──
            "context": {
                "loss": round(float(packet.loss), 4),
                "lr": float(packet.lr),
                "coherence": round(float(packet.coherence), 4),
                "friction": packet.friction_regime,
                "variance_delta": round(float(packet.delta), 3),
                "mpc_intervention": round(float(packet.mpc_intervention), 3),
                "forecast_error": round(float(packet.forecast_error), 3),
                "pressure_dominant": packet.pressure_dominant,
                "pressure_concentration": round(float(packet.pressure_concentration), 3),
            },
        }
        self._teaching_log.append(entry)
        # In-model storage for future auxiliary loss / prompt injection
        if not hasattr(self.model, "_teacher_rationales"):
            self.model._teacher_rationales = []
        self.model._teacher_rationales.append({
            "step": packet.step,
            "lesson": decision.lesson,
            "variable": decision.variable.value if decision.variable else None,
            "rationale": decision.rationale,
        })
        _total = self._persist_lesson(entry)
        print(f"   🎓 Lesson stored: {decision.lesson[:80]}...")
        print(f"   📚 Teaching memory: {_total} lessons dated & persisted → mycelia_lessons.jsonl")

    def _apply_decision(self, decision: ConsensusDecision, step: int, current_loss: float) -> List[str]:
        """Apply the consensus decision to the model with rate governor.

        v3.5: Registry-based dispatch. Every valid schema variable MUST have
        a PARAMETER_REGISTRY entry. Unmapped variables are explicitly rejected
        with a log line — never silently dropped.
        """
        actions = []

        # Rate governor: cooldown check
        if step - self._last_meta_step < self._meta_adjustment_cooldown:
            return ["META_RATE_LIMITED"]

        # Interaction guard: max simultaneous adjustments
        active = len([p for p in self.pending_verification
                      if p["step_applied"] > step - self._meta_adjustment_cooldown])
        if active >= self.max_meta_adjustments_per_cycle:
            return ["META_INTERACTION_GUARD"]

        var = decision.variable
        direction = decision.direction
        value = decision.value

        if not var or not direction:
            return ["INVALID_DECISION"]

        # VERIFIER: Registry lookup — explicit reject, never silent drop
        if var not in PARAMETER_REGISTRY:
            print(f"   🚫 VERIFIER_REJECT: {var.value} — no handler in PARAMETER_REGISTRY")
            return [f"VERIFIER_REJECT: {var.value} unmapped (expected=unknown)"]

        reg = PARAMETER_REGISTRY[var]
        target = reg["target"]
        lo, hi = reg["bounds"]

        # Enactable guard: don't auto-apply human/restart actions
        if decision.enactable != Enactable.RUNTIME:
            print(f"   🚫 VERIFIER_REJECT: {var.value} — enactable={decision.enactable.value}")
            return [f"VERIFIER_REJECT: {var.value} enactable={decision.enactable.value}"]

        # Compute adjustment
        if direction == Direction.SET and value is not None:
            new_val = max(lo, min(hi, value))
            action_str = f"{var.value} SET {new_val:.4f}"
            multiplier = None
        elif direction in (Direction.RAISE, Direction.LOWER):
            multiplier = value if value is not None else (1.10 if direction == Direction.RAISE else 0.90)
            multiplier = max(self.meta_adjustment_floor, min(self.meta_adjustment_ceiling, multiplier))
            action_str = f"{var.value} {direction.value} ×{multiplier:.3f}"
        elif direction in (Direction.PAUSE, Direction.INVESTIGATE):
            actions.append(f"{var.value} {direction.value} (no value change)")
            self._last_meta_step = step
            print(f"🛠  Meta-Governor [{decision.consensus_type}]: {var.value} {direction.value} "
                  f"(conf={decision.confidence:.2f}, agents={decision.participating_agents})")
            return actions
        else:
            return [f"VERIFIER_REJECT: invalid direction {direction.value}"]

        # DISPATCH: Apply to target
        if target == ParameterTarget.BLOCK:
            for block in self.model.blocks:
                current = getattr(block, var.value, None)
                if current is None:
                    continue
                if direction == Direction.SET:
                    new_val = value
                else:
                    new_val = current * multiplier
                new_val = max(lo, min(hi, new_val))
                setattr(block, var.value, new_val)
            actions.append(action_str)

        elif target == ParameterTarget.MODEL:
            current = getattr(self.model, var.value, None)
            if current is not None:
                if direction == Direction.SET:
                    new_val = value
                else:
                    new_val = current * multiplier
                new_val = max(lo, min(hi, new_val))
                setattr(self.model, var.value, new_val)
                actions.append(action_str)
            else:
                print(f"   ⚠️  Model has no attribute {var.value}, skipping")

        elif target == ParameterTarget.GLOBAL:
            m = multiplier if direction in (Direction.RAISE, Direction.LOWER) else value
            actions.append(f"SUGGEST_{var.value}_{direction.value}_{m:.3f}")

        # Track for verification with expected outcome
        expected = decision.expected_outcome or reg.get("default_expected")
        self.pending_verification.append({
            "step_applied": step,
            "decision": decision,
            "loss_at_application": current_loss,
            "expected_outcome": expected,
        })

        self._last_meta_step = step

        print(f"🛠  Meta-Governor [{decision.consensus_type}]: "
              f"{var.value} {direction.value} "
              f"(conf={decision.confidence:.2f}, agents={decision.participating_agents}, "
              f"expected={expected.value if expected else 'unknown'})")

        return actions

    def _verify_pending(self, packet: TelemetryPacket):
        """Check if pending suggestions had expected outcome."""
        current_loss = packet.loss

        for pending in self.pending_verification[:]:
            steps_elapsed = packet.step - pending["step_applied"]
            decision = pending["decision"]
            duration = decision.duration_steps

            if steps_elapsed < duration:
                continue

            # Evaluate outcome
            loss_before = pending.get("loss_at_application", current_loss)
            expected = decision.expected_outcome

            success = False
            if expected == ExpectedOutcome.LOSS_DECREASE:
                success = current_loss < loss_before - 0.01
            elif expected == ExpectedOutcome.PRESSURE_REDISTRIBUTE:
                success = packet.pressure_concentration < 0.85
            elif expected == ExpectedOutcome.COHERENCE_INCREASE:
                success = packet.coherence > 0.8
            elif expected == ExpectedOutcome.REGIME_TRANSITION:
                success = packet.friction_regime != "DEEP_DRIFT"

            # Update expert confidence
            for agent in decision.participating_agents:
                if success:
                    self.expert_confidence[agent] = min(
                        EXPERT_CONFIDENCE_MAX,
                        self.expert_confidence.get(agent, EXPERT_CONFIDENCE_DEFAULT) * 1.05
                    )
                else:
                    self.expert_confidence[agent] = max(
                        EXPERT_CONFIDENCE_MIN,
                        self.expert_confidence.get(agent, EXPERT_CONFIDENCE_DEFAULT) * 0.90
                    )

            self.recent_outcomes.append("success" if success else "failure")
            self.pending_verification.remove(pending)

            print(f"📊 Verification: {'✅' if success else '❌'} "
                  f"{decision.variable.value if decision.variable else 'unknown'} "
                  f"-> expected={expected.value if expected else 'unknown'}")

    def _update_circuit_breaker(self, packet: TelemetryPacket):
        """Evaluate and potentially reset circuit breaker."""
        failures = sum(1 for o in self.recent_outcomes if o == "failure")

        if failures <= CIRCUIT_BREAKER_RECOVERY_THRESHOLD:
            self.circuit_breaker_tripped = False
            print(f"🟢 Circuit breaker reset. Recent failures: {failures}/{len(self.recent_outcomes)}")

    def check_circuit_breaker(self):
        """Check if circuit breaker should trip based on recent outcomes."""
        failures = sum(1 for o in self.recent_outcomes if o == "failure")
        if failures > CIRCUIT_BREAKER_TRIP_THRESHOLD:
            if not self.circuit_breaker_tripped:
                print(f"🚨 CIRCUIT BREAKER TRIPPED: {failures} failures in last {len(self.recent_outcomes)} suggestions")
                self.circuit_breaker_tripped = True

    def get_status(self) -> Dict:
        """Return current meta-governor status for logging."""
        return {
            "circuit_breaker": self.circuit_breaker_tripped,
            "expert_confidences": self.expert_confidence,
            "pending_verifications": len(self.pending_verification),
            "telemetry_history_size": len(self.telemetry_history),
            "recent_outcomes": list(self.recent_outcomes),
            "architecture_suggestions": len(self.architecture_suggestions),
            "human_review_pending": len(self.human_review_queue),
            "teaching_log_size": len(self._teaching_log),
        }

    def get_teaching_log(self) -> List[Dict]:
        """Return all stored lessons for model reflection."""
        return list(self._teaching_log)

    def shutdown(self):
        """Clean shutdown."""
        self.executor.shutdown(wait=True)


# =============================================================================
# LOCAL FALLBACK MODE (No API)
# =============================================================================

class LocalMetaGovernor:
    """
    Fallback meta-governor that runs entirely locally.
    Uses hardcoded physics-informed rules instead of LLM agents.
    """

    def __init__(self, model):
        self.model = model
        self.telemetry_history: deque = deque(maxlen=5)

    def step(self, packet: TelemetryPacket) -> List[str]:
        """Apply local rules only."""
        self.telemetry_history.append(packet.to_dict())
        actions = []

        # NEW v3.1: Scheduler dead detection
        if not packet.scheduler_alive or not packet.lr_valid:
            print(f"   🚨 LOCAL RULE: Scheduler dead detected — LR={packet.lr:.2e}, alive={packet.scheduler_alive}")
            actions.append("LOCAL_RULE: SCHEDULER_DEAD — requires resurrection")
            return actions

        # Rule 1: Relief valve detection
        if packet.pressure_concentration > 0.90 and packet.pressure_dominant == "ffn":
            if packet.ffn_veto_ratio > 0.90:
                new_target = min(500, packet.ffn_target * 1.15)
                for block in self.model.blocks:
                    block.ffn_norm_target = new_target
                actions.append(f"LOCAL_RULE: ffn_norm_target -> {new_target:.0f} (relief valve)")

        # Rule 2: False positive MPC
        if packet.mpc_intervention > 0.60 and packet.forecast_error > 0.20:
            for block in self.model.blocks:
                block.instability_target = min(0.8, block.instability_target * 1.10)
            actions.append("LOCAL_RULE: instability_target +10% (false positive MPC)")

        # Rule 3: Deep drift
        if packet.delta < -1.5:
            for block in self.model.blocks:
                block.control_gain = max(0.3, block.control_gain * 0.90)
            actions.append("LOCAL_RULE: control_gain -10% (deep drift)")

        # Rule 4: Gradient starvation
        if (len(self.telemetry_history) >= 5 and 
            all(t["loss"] > 4.3 for t in list(self.telemetry_history)[-5:]) and
            packet.ffn_veto_ratio > 0.90):
            for block in self.model.blocks:
                block.ffn_norm_target = min(500, block.ffn_norm_target * 1.20)
            actions.append("LOCAL_RULE: ffn_norm_target +20% (gradient starvation)")

        return actions if actions else ["LOCAL_RULE: no_action"]

    def get_status(self) -> Dict:
        """Return status compatible with KIMIMetaGovernor.get_status()."""
        return {
            "circuit_breaker": False,
            "expert_confidences": {},
            "pending_verifications": 0,
            "telemetry_history_size": len(self.telemetry_history),
            "recent_outcomes": [],
            "architecture_suggestions": 0,
            "human_review_pending": 0,
            "teaching_log_size": 0,
        }

    def get_teaching_log(self) -> List[Dict]:
        return []


# =============================================================================
# TRAINING LOOP INTEGRATION (Drop-in) — v3.1: Scheduler-Aware
# =============================================================================

def integrate_meta_governor(model, auto_tuner, step: int, current_loss: float, 
                           current_lr: float, log_every: int = 250,
                           local_only: bool = False,
                           scheduler=None) -> List[str]:
    """
    Drop-in function for training loop.

    Usage inside training loop:
        if step % log_every == 0:
            actions = integrate_meta_governor(model, auto_tuner, step, 
                                              current_avg_loss, current_lr,
                                              scheduler=scheduler)
            if actions:
                print(f"Meta-Governor: {actions}")

    Args:
        local_only: Force LOCAL-ONLY mode even if API keys are set
        scheduler: Optional scheduler object for health telemetry
    """
    # Check API key at RUNTIME, not import time
    groq_key = os.environ.get("GROQ_API_KEY", "")
    has_api_key = bool(groq_key)

    # Build scheduler health info
    scheduler_info = {}
    if scheduler is not None:
        try:
            last_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else current_lr
            last_epoch = getattr(scheduler, "last_epoch", 0)
            total_steps = getattr(scheduler, "num_training_steps", 0)
            scheduler_info = {
                "alive": last_lr > 0,
                "last_epoch": last_epoch,
                "total_steps": total_steps,
                "resurrected_count": getattr(scheduler, "_resurrected_count", 0),
                "lr_valid": last_lr > 0,
            }
        except Exception:
            scheduler_info = {"alive": current_lr > 0, "lr_valid": current_lr > 0}

    # Lazy initialization: attach governor to auto_tuner on first call
    # v3.5: Recreate if stale (missing architecture-aware attributes from old checkpoint)
    governor_stale = (
        hasattr(auto_tuner, "_meta_governor")
        and not hasattr(auto_tuner._meta_governor, 'n_layers')
    )
    if not hasattr(auto_tuner, "_meta_governor") or governor_stale:
        if governor_stale:
            print("   🔄 Meta-Governor: stale object detected, recreating with architecture awareness")
        if local_only or not has_api_key:
            print("   📡 Meta-Governor: LOCAL-ONLY mode")
            auto_tuner._meta_governor = LocalMetaGovernor(model)
        else:
            _cfg = getattr(model, 'cfg', {}) or getattr(model, 'config', {})
            _n_layers = _cfg.get('n_layers', 24) if isinstance(_cfg, dict) else getattr(_cfg, 'n_layers', 24)
            _seq_len = _cfg.get('max_seq_len', 4096) if isinstance(_cfg, dict) else getattr(_cfg, 'max_seq_len', 4096)
            print(f"   📡 Meta-Governor: Groq LPU mode | {_n_layers} layers | seq_len={_seq_len}")

            auto_tuner._meta_governor = KIMIMetaGovernor(model)

    governor = auto_tuner._meta_governor

    # Build telemetry packet with scheduler health
    packet = TelemetryPacket.from_model(model, step, current_loss, current_lr,
                                        scheduler_info=scheduler_info)

    # Run meta-governor step
    actions = governor.step(packet)

    # Also run local circuit breaker check
    if hasattr(governor, "check_circuit_breaker"):
        governor.check_circuit_breaker()

    return actions


if __name__ == "__main__":
    print("🍄 KIMI Meta-Governor v4.0 — Groq LPU + Scheduler-Aware + Active Teaching")
    print("   Agents: Qwen3.6-27b (macro) + GPT OSS 20B (micro)")
    print("   Rate limits: 60 RPM / 30 RPM with safety guard")
    print("   Consensus threshold: 0.20 (lowered for more activity)")
    print("   Teaching: Verbose lesson output + model storage")
    print("   Import this module and use integrate_meta_governor() in your training loop.")
    print("   Set GROQ_API_KEY env var for API mode, or run in LOCAL-ONLY mode.")