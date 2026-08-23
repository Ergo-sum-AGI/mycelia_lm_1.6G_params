import torch

# ================================================================
# MANUAL MUON + 8-BIT ADAMW HYBRID OPTIMIZER  (v10.6-final)
# Inherits from torch.optim.Optimizer so HF schedulers recognize it.
#
# Fixes applied:
#   1. Newton-Schulz runs entirely in fp32 (no bf16 cast)
#   2. Momentum buffer allocated in fp32
#   3. Gradient cast to fp32 before momentum update
#   4. Update cast back to param dtype only at application
#   5. Aspect-ratio scaling: ** 0.5  (was truncated)
#   6. Learning-rate scaling: update.mul_(self.muon_lr)  (was missing)
#   7. Checkpoint keys use sequential index, not id(p)
#   8. halve_all_lr() covers both Muon and AdamW sub-optimizer
#   9. Shims reach into self.adamw for meta-governor well depth
# ================================================================


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration for the polar factor (orthogonal projection).

    G : (m, n) momentum matrix — MUST arrive as fp32 or will be cast.
    Returns UV^T in fp32.  Caller casts to param dtype.

    Coefficients from the Muon paper (Jordan et al., 2024).
    """
    assert G.dim() == 2, f"Expected 2-D tensor, got {G.dim()}-D"

    a, b, c = (3.4445, -4.7750, 2.0315)

    # ── CRITICAL: force fp32 for every matrix multiply ──
    X = G.float()
    X = X / (X.norm() + eps)

    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * (A @ B)

    if transposed:
        X = X.T

    return X  # fp32


class ManualMuonOptimizer(torch.optim.Optimizer):
    """
    Muon + 8-bit AdamW hybrid.

    Routing
    -------
    Muon  : every 2-D weight matrix EXCEPT embedding / lm_head
    AdamW8: embeddings, LayerNorms, biases, raw_alpha scalars,
            positional buffers, compressor parameters

    The AdamW sub-optimizer is managed internally via
    ``bitsandbytes.optim.AdamW8bit``.
    """

    def __init__(
        self,
        model,
        muon_lr: float = 3e-5,
        adamw_lr: float = 3e-5,
        muon_wd: float = 0.01,
        adamw_wd: float = 0.3,
        muon_momentum: float = 0.95,
        ns_steps: int = 5,
    ):
        # ── 1. Partition parameters ──────────────────────────────────
        self.muon_params = []
        self.adamw_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                param.dim() == 2
                and "embedding" not in name
                and "lm_head" not in name
            ):
                self.muon_params.append(param)
            else:
                self.adamw_params.append(param)

        # ── 2. Register muon params with base Optimizer ──────────────
        # This is what HF schedulers inspect via opt.param_groups
        defaults = dict(lr=muon_lr, weight_decay=muon_wd)
        super().__init__(self.muon_params, defaults)

        # ── 3. Muon momentum buffers (fp32, keyed by INDEX) ─────────
        # id(p) changes across process restarts → checkpoint corruption.
        # Sequential indices are stable and survive serialization.
        self.muon_state = {}
        self._muon_index = {}          # id(p) → sequential index
        for i, p in enumerate(self.muon_params):
            self.muon_state[i] = torch.zeros(
                p.shape, dtype=torch.float32, device=p.device
            )
            self._muon_index[id(p)] = i

        # ── 4. 8-bit AdamW for everything else ──────────────────────
        import bitsandbytes as bnb

        self.adamw = bnb.optim.AdamW8bit(
            self.adamw_params,
            lr=adamw_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=adamw_wd,
        )

        # ── 5. Cache hyper-parameters ───────────────────────────────
        self.muon_lr = muon_lr
        self.adamw_lr = adamw_lr
        self.muon_wd = muon_wd
        self.muon_momentum = muon_momentum
        self.ns_steps = ns_steps

        n_muon = sum(p.numel() for p in self.muon_params)
        n_adam = sum(p.numel() for p in self.adamw_params)
        print(f"[ManualMuon] Muon:   {n_muon:,} params ({len(self.muon_params)} tensors)")
        print(f"[ManualMuon] AdamW8: {n_adam:,} params ({len(self.adamw_params)} tensors)")

    # ── zero_grad ────────────────────────────────────────────────────
    def zero_grad(self, set_to_none: bool = False):
        super().zero_grad(set_to_none=set_to_none)      # muon params
        self.adamw.zero_grad(set_to_none=set_to_none)    # adamw params

    # ── step ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # --- Muon step (fp32 math, bf16 application) ---
        for p in self.muon_params:
            if p.grad is None:
                continue

            g = p.grad
            idx = self._muon_index[id(p)]
            state = self.muon_state[idx]

            # Momentum: fp32 buffer, cast grad to fp32
            state.mul_(self.muon_momentum).add_(g.float())

            # Polar decomposition (entirely in fp32)
            update = zeropower_via_newtonschulz5(state, steps=self.ns_steps)

            # Aspect-ratio scaling  (FIX: was truncated to ** 0.)
            update.mul_(max(1.0, g.size(0) / g.size(1)) ** 0.5)

            # Learning-rate scaling  (FIX: was missing entirely)
            update.mul_(self.muon_lr)

            # Decoupled weight decay on the parameter itself
            if self.muon_wd > 0:
                p.data.mul_(1.0 - self.muon_lr * self.muon_wd)

            # Apply: cast fp32 update → param dtype (bf16) at the last moment
            p.data.add_(update.to(p.dtype), alpha=-1.0)

        # --- AdamW8 step ---
        self.adamw.step(closure)

        return loss

    # ── LR helpers (training-loop NaN recovery) ──────────────────────
    def halve_all_lr(self):
        """Halve BOTH Muon and AdamW learning rates (called on NaN)."""
        self.muon_lr *= 0.5
        for pg in self.param_groups:
            pg["lr"] = self.muon_lr
        for pg in self.adamw.param_groups:
            pg["lr"] *= 0.5
        self.adamw_lr *= 0.5

    def sync_adamw_lr(self, new_lr: float):
        """Push a new LR into the internal AdamW sub-optimizer."""
        self.adamw_lr = new_lr
        for pg in self.adamw.param_groups:
            pg["lr"] = new_lr

    # ── checkpoint save / load ───────────────────────────────────────
    def state_dict(self):
        return {
            # Integer keys survive torch.save / torch.load
            # ── v11.1: NO eager .cpu() — that was a 5.25 GB CPU RAM spike ──
            # torch.save streams CUDA tensors to disk one buffer at a time;
            # load_state_dict() already moves them back to device on resume.
            "muon_state": dict(self.muon_state),
            "adamw_state": self.adamw.state_dict(),
            "muon_lr": self.muon_lr,
            "adamw_lr": self.adamw_lr,
        }

    def load_state_dict(self, state_dict):
        if "muon_state" in state_dict:
            for k, v in state_dict["muon_state"].items():
                k_int = int(k)  # keys may deserialise as strings
                if k_int in self.muon_state:
                    self.muon_state[k_int].copy_(
                        v.to(self.muon_state[k_int].device)
                    )
        if "adamw_state" in state_dict:
            self.adamw.load_state_dict(state_dict["adamw_state"])
        if "muon_lr" in state_dict:
            self.muon_lr = state_dict["muon_lr"]
            for pg in self.param_groups:
                pg["lr"] = self.muon_lr
        if "adamw_lr" in state_dict:
            self.sync_adamw_lr(state_dict["adamw_lr"])


# ================================================================
# FACTORY  — drop-in replacement for the old make_mycelia_optimizer
# ================================================================
def make_mycelia_optimizer(
    model,
    muon_lr: float = 3e-5,
    adamw_lr: float = 3e-5,
    muon_wd: float = 0.01,
    adamw_wd: float = 0.3,
):
    """Returns a ManualMuonOptimizer.  No external muon package needed."""
    return ManualMuonOptimizer(
        model,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        muon_wd=muon_wd,
        adamw_wd=adamw_wd,
    )


# ================================================================
# META-GOVERNOR SHIMS
# The Meta-Governor reads / writes the alpha potential-well depth.
# Under ManualMuonOptimizer the well depth lives inside the internal
# AdamW sub-optimizer (self.adamw.param_groups[0]['weight_decay']).
# ================================================================
def get_alpha_well_depth(opt) -> float:
    try:
        if hasattr(opt, "adamw") and opt.adamw.param_groups:
            return float(opt.adamw.param_groups[0].get("weight_decay", 0.3))
    except Exception:
        pass
    return 0.3


def set_alpha_well_depth(opt, new_wd: float):
    try:
        if hasattr(opt, "adamw") and opt.adamw.param_groups:
            opt.adamw.param_groups[0]["weight_decay"] = float(new_wd)
    except Exception:
        pass