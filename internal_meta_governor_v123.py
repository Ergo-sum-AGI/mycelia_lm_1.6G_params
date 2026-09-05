"""
Internal Meta-Governor v12.0 — Lesson-Based Retrieval (LBR)
==========================================================
Zero external API calls. Learns from mycelia_lessons.jsonl and live telemetry.
Drop-in replacement for Meta-Governor v4.py.

Architecture:
1. LessonMemory: Cosine-similarity k-NN over normalized telemetry vectors.
2. InternalMetaGovernor: retrieval + local physics rules + online learning.
3. Self-Geometry enrichment: optional fusion with EN_Self_Geometry embeddings.
"""

import json
import os
import random
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

LESSON_PATH_DEFAULT = os.path.join(
    os.environ.get("SM_MODEL_DIR", "/home/ec2-user/SageMaker"),
    "mycelia_checkpoints",
    "mycelia_lessons.jsonl"
)

GEOMETRY_PATH_DEFAULT = os.path.join(
    os.environ.get("SM_MODEL_DIR", "/home/ec2-user/SageMaker"),
    "mycelia_s3_chunks",
    "EN_Self_Geometry_p0.npy"
)

# Feature normalization ranges (empirical from Mycelia v8-v11 telemetry)
FEATURE_RANGES = {
    "loss": 5.0,
    "lr": 1e-3,
    "coherence": 1.0,
    "variance_delta": 3.0,
    "mpc_intervention": 1.0,
    "forecast_error": 1.0,
    "pressure_concentration": 1.0,
    "mean_curvature": 1.0,
    "mean_velocity": 1.0,
    "mean_acceleration": 1.0,
    "phase": 1.0
}

# Actionable variable registry (must match training loop block attributes)
PARAMETER_REGISTRY = {
    "ffn_norm_target":     {"attr": "ffn_norm_target",     "lo": 10,   "hi": 1000, "target": "block"},
    "alpha_norm_target":   {"attr": "alpha_norm_target",   "lo": 10,   "hi": 1000, "target": "block"},
    "control_gain":        {"attr": "control_gain",        "lo": 0.01, "hi": 10.0,  "target": "block"},
    "instability_target":  {"attr": "instability_target",  "lo": 0.01, "hi": 0.85,  "target": "block"},
    "soft_cap":            {"attr": "soft_cap_target",       "lo": 100,  "hi": 2000,  "target": "block"},
    "alpha_well_depth":    {"attr": None,                  "lo": -0.2, "hi": 0.6,   "target": "optimizer"},
    "temperature":         {"attr": None,                  "lo": 0.3,  "hi": 2.0,   "target": "model"},
}


# =============================================================================
# LESSON MEMORY — k-NN over telemetry manifolds
# =============================================================================

class LessonMemory:
    """
    Loads persisted lessons from meta-governor v4 format (JSONL) and builds
    a searchable vector index. Supports online append for lifelong learning.
    """

    def prune_sterile_lessons(self):
        """Remove instability_target lessons generated while MPC was dormant."""
        if self.vectors is None or len(self.lessons) == 0:
            return
        keep_indices = []
        for i, lesson in enumerate(self.lessons):
            if lesson.get("variable") == "instability_target":
                ctx = lesson.get("context", {})
                mpc = ctx.get("mpc_intervention", 1.0)
                try:
                    mpc_val = float(mpc)
                except (ValueError, TypeError):
                    mpc_val = 1.0
                if mpc_val < 0.05:
                    continue  # Skip poisoned lesson
            keep_indices.append(i)
        pruned_count = len(self.lessons) - len(keep_indices)
        self.lessons = [self.lessons[i] for i in keep_indices]
        self.vectors = self.vectors[keep_indices]
        print(f"🧹 LBR-Memory: pruned {pruned_count} sterile lessons, {len(self.lessons)} remain")

    def __init__(self, lessons_path: str, geometry_path: Optional[str] = None,
                 feature_keys: Optional[List[str]] = None):
        self.lessons_path = lessons_path
        self.geometry_path = geometry_path
        self.feature_keys = feature_keys or [
            "loss", "coherence", "variance_delta", "mpc_intervention",
            "forecast_error", "pressure_concentration", "mean_curvature"
        ]
        self.lessons: List[Dict] = []
        self.vectors: Optional[np.ndarray] = None
        self.geometry: Optional[np.ndarray] = None
        self._load_lessons()
        self._load_geometry()

    def _load_lessons(self):
        if not os.path.exists(self.lessons_path):
            print(f"🧠 LBR-Memory: no lesson file at {self.lessons_path}")
            return

        count = 0
        with open(self.lessons_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ctx = entry.get("context") or {}
                vec = self._context_to_vector(ctx)
                if vec is None:
                    continue

                var = entry.get("variable")
                direction = entry.get("direction")
                if not var or not direction:
                    continue

                self.lessons.append(entry)
                if self.vectors is None:
                    self.vectors = vec.reshape(1, -1)
                else:
                    self.vectors = np.vstack([self.vectors, vec])
                count += 1

        dim = self.vectors.shape[1] if self.vectors is not None else 0
        print(f"🧠 LBR-Memory: indexed {count} actionable lessons | vector_dim={dim}")

    def _load_geometry(self):
        if not self.geometry_path or not os.path.exists(self.geometry_path):
            return
        try:
            geo = np.load(self.geometry_path, mmap_mode="r")
            print(f"🧬 LBR-Memory: self-geometry manifold ready | shape={geo.shape}")
            self.geometry = geo
        except Exception as e:
            print(f"⚠️  LBR-Memory: self-geometry load failed: {e}")
            self.geometry = None

    def _context_to_vector(self, ctx: Dict[str, Any]) -> Optional[np.ndarray]:
        vec = []
        for key in self.feature_keys:
            raw = ctx.get(key, 0.0)
            if hasattr(raw, "item"):
                raw = raw.item()
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = 0.0
            rng = FEATURE_RANGES.get(key, 1.0)
            if rng == 0.0:
                rng = 1.0
            vec.append(val / rng)

        arr = np.array(vec, dtype=np.float32)
        arr = np.tanh(arr)
        return arr

    def find_similar(self, ctx: Dict[str, Any], k: int = 5,
                     min_similarity: float = 0.3) -> List[Tuple[int, float]]:
        if self.vectors is None or len(self.vectors) == 0:
            return []

        q = self._context_to_vector(ctx).reshape(1, -1)
        q_norm = np.linalg.norm(q) + 1e-8
        v_norm = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8
        sims = (self.vectors @ q.T).flatten() / (v_norm.flatten() * q_norm)
        sims = (sims + 1.0) / 2.0

        valid_idx = np.where(sims >= min_similarity)[0]
        if len(valid_idx) == 0:
            return []

        top_k_idx = valid_idx[np.argsort(sims[valid_idx])[-k:][::-1]]
        return [(int(idx), float(sims[idx])) for idx in top_k_idx]

    def add_experience(self, ctx: Dict[str, Any], action: Dict[str, Any],
                       outcome: Optional[str] = None):
        vec = self._context_to_vector(ctx)
        if vec is None:
            return

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "context": ctx,
            "variable": action.get("variable"),
            "direction": action.get("direction"),
            "outcome": outcome,
        }

        self.lessons.append(entry)
        if self.vectors is None:
            self.vectors = vec.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vec])

        try:
            with open(self.lessons_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️  LBR-Memory: disk append failed: {e}")


# =============================================================================
# INTERNAL META-GOVERNOR
# =============================================================================

class InternalMetaGovernor:
    """
    Autonomous governor that retrieves past interventions and applies them
    with physics-informed guardrails. No network required.
    """

    def __init__(self, model, lessons_path: Optional[str] = None,
                 geometry_path: Optional[str] = None):
        self.model = model
        self.memory = LessonMemory(
            lessons_path or LESSON_PATH_DEFAULT,
            geometry_path or GEOMETRY_PATH_DEFAULT
        )
        self.telemetry_history: deque = deque(maxlen=20)
        self.pending_verification: List[Dict] = []
        self._last_meta_step: int = 0
        self._meta_cooldown: int = 500
        self.circuit_breaker: bool = False
        self.recent_outcomes: deque = deque(maxlen=10)
        self.exploration_rate: float = 0.10
        self.min_confidence: float = 0.35

    def step(self, packet, auto_tuner=None) -> List[str]:
        """Called every LOG_EVERY steps. Returns list of action strings."""
        self.telemetry_history.append(packet.to_dict())

        if self.circuit_breaker:
            return ["CIRCUIT_BREAKER_ACTIVE"]

        ctx = {
            "loss": packet.loss,
            "coherence": packet.coherence,
            "variance_delta": packet.delta,
            "mpc_intervention": packet.mpc_intervention,
            "forecast_error": packet.forecast_error,
            "pressure_concentration": packet.pressure_concentration,
            "mean_curvature": packet.mean_curvature,
            "pressure_dominant": getattr(packet, "pressure_dominant", "ffn"),
            "phase": auto_tuner.phase if (auto_tuner is not None and hasattr(auto_tuner, 'phase')) else 0.0,
            # --- v12.1b: Ghost-parameter telemetry for alpha channel ---
            "fiber_curvature_head_var": getattr(packet, "fiber_curvature_head_var", 0.0),
            "optimization_response_chi_R": getattr(packet, "optimization_response_chi_R", 0.0),
            "alpha_scale": getattr(packet, "alpha_scale", 1.0),
            "contrib_norm": getattr(packet, "contrib_norm", 0.0),
            # -----------------------------------------------------------
        }

        similar = self.memory.find_similar(ctx, k=5, min_similarity=0.25)

        if not similar:
            return self._local_fallback(packet)

        decision = self._aggregate_retrieval(similar, ctx)
        if decision is None:
            return self._local_fallback(packet)

        if random.random() < self.exploration_rate:
            decision = self._explore(decision)

        if decision["confidence"] < self.min_confidence:
            return ["NO_CONFIDENCE"]

        actions = self._apply(decision, packet.step, packet.loss, packet.mpc_intervention, ctx)
        return actions

    def _aggregate_retrieval(self, similar: List[Tuple[int, float]],
                             ctx: Dict) -> Optional[Dict]:
        votes = defaultdict(lambda: {"weight": 0.0, "directions": defaultdict(float)})

        for idx, sim in similar:
            lesson = self.memory.lessons[idx]
            var = lesson.get("variable")
            direction = lesson.get("direction")
            if not var:
                continue
            # v12.1: Deprioritize instability_target lessons when MPC is dormant
            if var == "instability_target" and ctx.get("mpc_intervention", 1.0) < 0.05:
                sim *= 0.1  # heavily discount sterile lessons
            direction = self._normalize_direction(direction, var, ctx)
            if direction is None:
                continue
            votes[var]["weight"] += sim
            votes[var]["directions"][direction] += sim

        if not votes:
            return None

        best_var = max(votes, key=lambda v: votes[v]["weight"])
        entry = votes[best_var]

        best_dir = max(entry["directions"], key=entry["directions"].get)
        total_weight = entry["weight"]
        dir_weight = entry["directions"][best_dir]
        confidence = dir_weight / (total_weight + 1e-8)

        return {
            "variable": best_var,
            "direction": best_dir,
            "confidence": confidence,
            "context": ctx,
        }

    def _normalize_direction(self, direction: Optional[str], variable: str,
                              ctx: Dict) -> Optional[str]:
        if direction is None:
            return None

        d = str(direction).lower().strip()

        if d in ("raise", "lower", "set", "investigate", "pause"):
            return d

        if d == "healthy_regime":
            return "investigate"
        if d == "safety_mechanism":
            if variable in ("ffn_norm_target", "alpha_norm_target", "soft_cap"):
                return "raise"
            if variable in ("control_gain", "instability_target"):
                return "lower"
            return "investigate"

        synonyms = {
            "up": "raise", "increase": "raise", "boost": "raise", "deepen": "raise",
            "down": "lower", "decrease": "lower", "reduce": "lower", "shrink": "lower",
            "fix": "set", "assign": "set",
            "stop": "pause", "halt": "pause", "freeze": "pause",
            "check": "investigate", "inspect": "investigate", "analyze": "investigate",
        }
        return synonyms.get(d, None)

    def _explore(self, decision: Dict) -> Dict:
        mutated = dict(decision)
        mutated["direction"] = random.choice(["raise", "lower"])
        mutated["confidence"] *= 0.6
        mutated["exploration"] = True
        return mutated

    def _local_fallback(self, packet) -> List[str]:
        actions = []

        if not packet.scheduler_alive or not packet.lr_valid:
            actions.append("LOCAL_RULE: SCHEDULER_DEAD — LR=0, resurrection required")

        if packet.pressure_concentration > 0.90:
            actions.append("LOCAL_RULE: pressure_concentration>0.90 — redistribute targets")

        if packet.mpc_intervention > 0.60 and packet.forecast_error > 0.20:
            actions.append("LOCAL_RULE: MPC false-positive — raise instability_target +10%")

        if packet.delta < -1.5:
            actions.append("LOCAL_RULE: DEEP_DRIFT — dampen control_gain 10%")

        if packet.mean_curvature > 0.8 and packet.coherence < 0.5:
            actions.append("LOCAL_RULE: high curvature + low coherence — raise expected_curvature")

        # v12.1: Compensatory deadlock (R≈0, FFN-dominant, MPC dormant)
        # Force alpha channel engagement by lowering alpha_norm_target
        if (packet.pressure_concentration > 0.0
                and packet.pressure_dominant == "ffn"
                and packet.mpc_intervention < 0.05):
            for block in self.model.blocks:
                current = getattr(block, 'alpha_norm_target', 150.0)
                if hasattr(current, 'item'):
                    current = current.item()
                block.alpha_norm_target = max(20.0, current * 0.90)
            actions.append("LOCAL_RULE: alpha_norm_target -10% (compensatory deadlock wake-up)")

        return actions if actions else ["LOCAL_RULE: no_action"]

    def _apply(self, decision: Dict, step: int, current_loss: float, mpc_intervention: float = 1.0, ctx: Optional[Dict] = None) -> List[str]:
        if step - self._last_meta_step < self._meta_cooldown:
            return ["META_RATE_LIMITED"]
        var = decision["variable"]
        direction = decision["direction"]
        ctx = ctx or {}
        
        # =====================================================================
        # =====================================================================
        # 🚑 GOVERNOR RESCUE (v12.1d) - The Chi-Based Fix
        # Uses guaranteed telemetry (pressure_concentration) instead of phantom alpha_scale.
        # =====================================================================
        if var == "alpha_norm_target" and direction == "raise":
            chi = ctx.get("pressure_concentration", 0.0)
            if chi > 0.80:
                print(f"   🚑 GOVERNOR_RESCUE: FFN dominant (χ={chi:.2f}), forcing target to 30.0 to trigger compression")
                direction = "set"
                decision["value"] = 30.0
                decision["direction"] = direction
        # =====================================================================
        # v12.1: Sterile-action guard — MPC dormant means instability_target...
        if var == "instability_target" and mpc_intervention < 0.05:
            print(f"   🚫 VERIFIER_REJECT: {var} — MPC dormant ({mpc_intervention:.3f})")
            return [f"VERIFIER_REJECT: {var} sterile"]
            
        if var not in PARAMETER_REGISTRY:
            return [f"UNKNOWN_VARIABLE: {var}"]
        reg = PARAMETER_REGISTRY[var]
        attr = reg["attr"]
        lo, hi = reg["lo"], reg["hi"]
        target = reg["target"]
        actions = []
        
        if var == "alpha_well_depth":
            actions.extend(self._apply_alpha_well(direction, decision.get("value")))
            self._last_meta_step = step
            self.pending_verification.append({"step_applied": step, "loss_at_application": current_loss, "decision": decision})
            return actions
            
        if target == "model":
            current = getattr(self.model, attr, None)
            if current is not None:
                new_val = self._compute_new_value(current, direction, decision.get("value"))
                if new_val is not None:
                    setattr(self.model, attr, max(lo, min(hi, new_val)))
                    actions.append(f"{var} {direction} -> {new_val:.4f} (model)")
                    self._last_meta_step = step
                    self.pending_verification.append({"step_applied": step, "loss_at_application": current_loss, "decision": decision})
            return actions if actions else [f"NO_APPLY: model has no attr {attr}"]
            
        if target == "block":
            applied = False
            for block in self.model.blocks:
                current = getattr(block, attr, None)
                if current is None: continue
                new_val = self._compute_new_value(current, direction, decision.get("value"))
                if new_val is None: continue
                setattr(block, attr, max(lo, min(hi, new_val)))
                applied = True
            if applied:
                actions.append(f"{var} {direction} -> {new_val:.4f} (all blocks)")
                self._last_meta_step = step
                self.pending_verification.append({"step_applied": step, "loss_at_application": current_loss, "decision": decision})
        return actions if actions else [f"NO_APPLY: {var} not found"]

    def _compute_new_value(self, current: float, direction: str,
                               value: Optional[float]) -> Optional[float]:
            if direction == "set" and value is not None:
                return value
            if direction == "raise":
                return current * 1.10
            if direction == "lower":
                return current * 0.90
            if direction in ("investigate", "pause"):
                return None
            return None

    def _apply_alpha_well(self, direction: str, value: Optional[float]) -> List[str]:
            if direction == "set" and value is not None:
                return [f"ALPHA_WELL_DEPTH_set_{value:.3f}"]
            if direction == "raise":
                return ["ALPHA_WELL_DEPTH_raise_0.050"]
            if direction == "lower":
                return ["ALPHA_WELL_DEPTH_lower_0.050"]
            return []

    def verify(self, packet):
            for pending in self.pending_verification[:]:
                if packet.step - pending["step_applied"] < 1000:
                    continue

                loss_before = pending["loss_at_application"]
                decision = pending["decision"]

                success = packet.loss < loss_before * 0.995
                outcome = "success" if success else "failure"
                self.recent_outcomes.append(outcome)

                self.memory.add_experience(
                    decision.get("context", {}),
                    {"variable": decision["variable"], "direction": decision["direction"]},
                    outcome=outcome
                )

                self.pending_verification.remove(pending)

            failures = sum(1 for o in self.recent_outcomes if o == "failure")
            if failures > 7 and not self.circuit_breaker:
                self.circuit_breaker = True
                print("🔴 LBR-Governor: CIRCUIT BREAKER TRIPPED")

    def get_status(self) -> Dict:
            return {
                "circuit_breaker": self.circuit_breaker,
                "memory_size": len(self.memory.lessons),
                "vector_dim": self.memory.vectors.shape[1] if self.memory.vectors is not None else 0,
                "pending_verifications": len(self.pending_verification),
                "exploration_rate": self.exploration_rate,
                "recent_outcomes": list(self.recent_outcomes),
            }



# =============================================================================
# DROP-IN REPLACEMENT for integrate_meta_governor()
# =============================================================================

class _MinimalTelemetryPacket:
    """Lightweight telemetry packet compatible with InternalMetaGovernor."""
    __slots__ = ("step", "loss", "lr", "coherence", "delta", "mpc_intervention",
                 "forecast_error", "pressure_concentration", "mean_curvature",
                 "scheduler_alive", "lr_valid", "pressure_dominant",
                 "fiber_curvature_head_var", "optimization_response_chi_R",
                 "alpha_scale", "contrib_norm")

    def __init__(self, step, loss, lr, coherence, delta, mpc_intervention,
                 forecast_error, pressure_concentration, mean_curvature,
                 scheduler_alive, lr_valid, pressure_dominant="ffn",
                 fiber_curvature_head_var=0.0, optimization_response_chi_R=0.0,
                 alpha_scale=1.0, contrib_norm=0.0):

        self.step = step
        self.loss = loss
        self.lr = lr
        self.coherence = coherence
        self.delta = delta
        self.mpc_intervention = mpc_intervention
        self.forecast_error = forecast_error
        self.pressure_concentration = pressure_concentration
        self.mean_curvature = mean_curvature
        self.scheduler_alive = scheduler_alive
        self.lr_valid = lr_valid
        self.pressure_dominant = pressure_dominant
        self.fiber_curvature_head_var = fiber_curvature_head_var
        self.optimization_response_chi_R = optimization_response_chi_R
        self.alpha_scale = alpha_scale
        self.contrib_norm = contrib_norm

    def to_dict(self):
        return {s: getattr(self, s) for s in self.__slots__}

def integrate_meta_governor(model, auto_tuner, step: int, current_loss: float,
                            current_lr: float, log_every: int = 250,
                            local_only: bool = True,
                            scheduler=None,
                            lessons_path: Optional[str] = None,
                            geometry_path: Optional[str] = None) -> List[str]:
    """
    Drop-in replacement for Meta-Governor v4 integrate_meta_governor().
    Uses zero external APIs — pure retrieval + physics.
    """
    # Lazy init
    if not hasattr(auto_tuner, "_meta_governor") or not isinstance(auto_tuner._meta_governor, InternalMetaGovernor):
        print("   🧠 Internal Meta-Governor v1.0 (LBR) initializing...")
        auto_tuner._meta_governor = InternalMetaGovernor(
            model,
            lessons_path=lessons_path,
            geometry_path=geometry_path
        )

    governor = auto_tuner._meta_governor
    governor.memory.prune_sterile_lessons()  # ← one-time cleanup

    # Build minimal packet from model telemetry (compatible with v4 TelemetryPacket)
    info = getattr(model, "_last_info", {}) or {}
    packet = _MinimalTelemetryPacket(
        step=step, loss=current_loss, lr=current_lr,
        coherence=float(info.get("coherence", info.get("kuramoto_R", 0.0))), delta=float(info.get("variance_delta", 0.0)),
        mpc_intervention=float(info.get("mpc_intervention_ratio", 0.0)), forecast_error=float(info.get("forecast_error", 0.0)),
        pressure_concentration=float(info.get("pressure_concentration", 0.0)),
        pressure_dominant=str(info.get("pressure_dominant", info.get("dominant", "ffn"))),
        mean_curvature=float(info.get("mean_curvature", 0.0)),
        scheduler_alive=(current_lr > 0), lr_valid=(current_lr > 0),
    )

    # =====================================================================
    # v12.1c: Wire real telemetry from forward pass 'info' dict to packet.
    # This prevents ctx from receiving phantom defaults (1.0 / 0.0).
    # =====================================================================
    packet.alpha_scale = float(info.get('alpha_scale', info.get('alphascale', 1.0)))
    packet.contrib_norm = float(info.get('contrib_norm', info.get('alpha_contrib_norm', 0.0)))
    packet.fiber_curvature_head_var = float(info.get('fiber_curvature_head_var', info.get('head_variance', 0.0)))
    packet.optimization_response_chi_R = float(info.get('optimization_response_chi_R', info.get('chi_R', 0.0)))
    # =====================================================================

    actions = governor.step(packet, auto_tuner=auto_tuner)
    governor.verify(packet)

    # Status print
    status = governor.get_status()
    mem_icon = "🧠" if status["memory_size"] > 0 else "🫙"
    cb_icon = "🔴" if status["circuit_breaker"] else "🟢"
    print(f"   {mem_icon} LBR-Gov: CB={cb_icon} | memory={status['memory_size']} | "
          f"pending={status['pending_verifications']} | ε={status['exploration_rate']:.2f}")

    return actions


if __name__ == "__main__":
    print("🍄 Internal Meta-Governor v1.0 — Lesson-Based Retrieval (LBR)")
    print("   Zero external APIs. Learns from mycelia_lessons.jsonl.")
    print("   Import and call integrate_meta_governor() as drop-in replacement.")