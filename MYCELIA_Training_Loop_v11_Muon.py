# ============================================
# MYCELIA Training Loop — v11 (1.5B Muon Integration)
# Refactored: canonical PressureState, TuningDecision, lineage receipts
# v10.5: Alpha Potential Well integration with meta-governor telemetry
# v1.5B: Muon + 8-bit AdamW Hybrid Optimizer + R-Macroscopic Telemetry
# v9.1 → v10.0 CHANGES:
#   - Imports: optimization_state, governor_auto_tuner, lineage_receipt
#   - GovernorAutoTuner: refactored into external module, returns TuningDecision
#   - R-guarded + phase-aware logic: moved into auto_tuner.tune()
#   - Checkpoint loading: safe loader with progress, size check, weights_only fallback
#   - Checkpoints: SHA-256 checksum + lineage_receipt on ALL saves
#   - Meta-governor: auto_tune_info built internally
#   - Removed: _last_phase_action, _last_r_action dead code
#
# v10.0 → v10.5 CHANGES:
#   - Alpha potential well telemetry enrichment for meta-governor
#   - ALPHA_WELL_DEPTH / ALPHA_WELL_INVERT action handling from meta-governor
#   - Alpha statistics logging (attn/ffn min/max/mean)
#
# v10.5 → v10.5-Muon (1.5B) CHANGES:
#   - Replaced dual-group AdamW with Muon + 8-bit AdamW hybrid (make_mycelia_optimizer)
#   - Hook A: alpha_regularization_loss() injected into forward loss (Muon has no 1D WD)
#   - Hook B: alpha_grad_norms telemetry after backward for meta-governor
#   - Hook C: Macroscopic R = α_work / (ffn_work + mpc_work) regime order parameter
#   - Meta-governor shims: get/set_alpha_well_depth() for Muon wrapper compatibility
#
# v11.0: New Hook C, alphas off the hook, TEACHING DISTILLATION STREAM
# ============================================
# Get ready for the ride!
# ============================================
# IMPORTS
# ============================================

import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import gc
import json
import time
import math
import signal
import hashlib
import boto3
import io
import requests
import warnings
import numpy as np
import glob
import random
import bisect
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datetime import datetime, timedelta
from tqdm import tqdm
# ─── NEW REFACTOR IMPORTS ────────────────────────────────────────────────
from optimization_state import PressureState, OptimizationRegime
from governor_auto_tuner import GovernorAutoTuner, TuningDecision
from lineage_receipt import compute_lineage_receipt
from meta_governor import integrate_meta_governor
# ── 1.5B OPTIMIZER IMPORTS
from manual_muon_optimizer import ManualMuonOptimizer, make_mycelia_optimizer

warnings.filterwarnings('ignore')

class UniversalNpyDataset(Dataset):
    def __init__(self, npy_dir, max_seq_len=512, teaching_path=None, teaching_weight=0.05, tokenizer=None):
        self.shards = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
        print(f"📚 Universal Dataset loaded: {len(self.shards)} shards found.")
        self.max_seq_len = max_seq_len
        self.target_len = max_seq_len + 1
        self.mmaps = [np.load(s, mmap_mode='r') for s in self.shards]
        self.shard_sizes = [m.shape[0] for m in self.mmaps]
        self.total_chunks = sum(self.shard_sizes)
        print(f"📊 Total base chunks available: {self.total_chunks:,}")

        # Precompute cumulative sizes for fast global-index → (shard, row) lookup
        self.cumulative_sizes = []
        cumsum = 0
        for size in self.shard_sizes:
            cumsum += size
            self.cumulative_sizes.append(cumsum)

        # Build a domain map: shard index → domain name (parsed from filename)
        self.shard_domains = []
        for s in self.shards:
            name = os.path.basename(s)
            # Strip _p0.npy, _p1.npy etc. to get domain name
            domain = name.rsplit('_p', 1)[0] if '_p' in name else name.replace('.npy', '')
            self.shard_domains.append(domain)

        # Epoch-based shuffled index
        self.indices = list(range(self.total_chunks))
        random.shuffle(self.indices)
        self.current_pos = 0
        self.epoch_count = 1
        self.seen_in_epoch = 0

        # Teaching integration
        self.teaching_weight = teaching_weight
        self.teaching_lessons = []
        self.teaching_path = teaching_path
        self.tokenizer = tokenizer
        self.teaching_mtime = 0
        self.teaching_triggers = 0
        self.teaching_log_interval = 20  # log every 20 teaching triggers

        if self.teaching_path and self.tokenizer:
            self.refresh_teaching()

    def refresh_teaching(self):
        if not self.teaching_path or not os.path.exists(self.teaching_path):
            return
        current_mtime = os.path.getmtime(self.teaching_path)
        if current_mtime == self.teaching_mtime:
            return
        self.teaching_mtime = current_mtime

        new_lessons = []
        seen = set()
        with open(self.teaching_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = (row.get('lesson') or '').strip()
                    if len(text) < 40 or text in seen:
                        continue
                    seen.add(text)
                    ctx = row.get('context') or {}
                    header = (f"[MYCELIA TEACHING | step {row.get('step', '?')} | "
                              f"variable: {row.get('variable', '?')} | "
                              f"action: {row.get('direction', '?')}")
                    if ctx:
                        header += (f" | loss {ctx.get('loss', '?')} | coh {ctx.get('coherence', '?')}"
                                   f" | friction {ctx.get('friction', '?')} | mpc {ctx.get('mpc_intervention', '?')}")
                    header += "]\n"
                    full_text = header + text

                    toks = self.tokenizer.encode(full_text, add_special_tokens=False)
                    if self.tokenizer.eos_token_id is not None:
                        toks.append(self.tokenizer.eos_token_id)

                    for i in range(0, len(toks) - self.target_len + 1, self.target_len):
                        chunk = toks[i:i + self.target_len]
                        if len(chunk) == self.target_len:
                            new_lessons.append(torch.tensor(chunk, dtype=torch.long))
                except Exception:
                    continue

        self.teaching_lessons = new_lessons
        if len(self.teaching_lessons) > 0:
            print(f"🧠 Refreshed Teaching Stream: {len(self.teaching_lessons)} lesson chunks loaded.")

    def _global_to_shard(self, global_idx):
        """Convert a global chunk index to (shard_idx, row_idx) using binary search."""
        shard_idx = bisect.bisect_right(self.cumulative_sizes, global_idx)
        if shard_idx == 0:
            row_idx = global_idx
        else:
            row_idx = global_idx - self.cumulative_sizes[shard_idx - 1]
        return shard_idx, row_idx

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        # --- Teaching stream (5% chance) ---
        if self.teaching_lessons and random.random() < self.teaching_weight:
            if random.random() < 0.01:
                self.refresh_teaching()
            self.teaching_triggers += 1
            if self.teaching_triggers % self.teaching_log_interval == 0:
                print(f"🧠 Teaching trigger #{self.teaching_triggers} | "
                      f"epoch={self.epoch_count} | pos={self.seen_in_epoch}/{self.total_chunks:,} | "
                      f"lessons_loaded={len(self.teaching_lessons)}")
            return random.choice(self.teaching_lessons)

        # --- Epoch-based corpus sampling (no replacement within epoch) ---
        if self.current_pos >= len(self.indices):
            random.shuffle(self.indices)
            self.current_pos = 0
            self.epoch_count += 1
            self.seen_in_epoch = 0
            print(f"🔄 Epoch {self.epoch_count} started: reshuffled {self.total_chunks:,} chunks")

        global_idx = self.indices[self.current_pos]
        self.current_pos += 1
        self.seen_in_epoch += 1

        shard_idx, row_idx = self._global_to_shard(global_idx)
        chunk = self.mmaps[shard_idx][row_idx]

        # Log domain coverage every 50,000 chunks
        if self.seen_in_epoch % 50_000 == 0:
            domain = self.shard_domains[shard_idx]
            pct = (self.seen_in_epoch / self.total_chunks) * 100
            print(f"📊 Epoch {self.epoch_count} | {pct:.1f}% seen | "
                  f"current domain: {domain} (shard {shard_idx}/{len(self.shards)})")

        return torch.tensor(chunk, dtype=torch.long)

# ============================================
# CONFIGURATION
# ============================================
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
ACCUM_STEPS = 32
WEIGHT_DECAY = 0.01
GRAD_CLIP = 2.0
SAVE_EVERY = 5000
LOG_EVERY = 250
CACHE_CLEAN_EVERY = 1000

PEAK_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 500
TOTAL_TOKENS_TARGET = 5_000_000_000

ENABLE_LR_BURST = False
CONSENSUS_ROUNDS = 2
AUTO_TUNE_EVERY = 10000

CONTROL_GAIN_MIN = 0.5
CONTROL_GAIN_MAX = 1.5
CONTROL_GAIN_DEFAULT = 1.0

ADAPTIVE_TARGETS_ENABLED = False
USE_GRADUAL_TRANSITION = True
TRANSITION_DURATION = 10000
FFN_TARGET_START = 50.0
FFN_TARGET_END = 150.0
ALPHA_TARGET_START = 50.0 # lowered from 50.0 to shrink the gap
ALPHA_TARGET_END = 150.0

MAX_SIMULTANEOUS_GOVERNORS = 2
PRESSURE_CONCENTRATION_ALERT = 0.85

S3_BUCKET = "sagemaker-eu-central-1-119287771635"
HQ_PREFIX = "massif-llm-highquality"
FINEWEB_PREFIX = "fineweb_cache"
STANFORD_ONLY = ["stanford_philosophy_processed.jsonl"]

CKPT_DIR = os.path.join(os.environ.get('SM_MODEL_DIR', '/home/ec2-user/SageMaker'), 'mycelia_checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
LATEST_CKPT = os.path.join(CKPT_DIR, "mycelia_latest.pt")
BEST_CKPT = os.path.join(CKPT_DIR, "mycelia_best.pt")

_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n🛑 Shutdown signal received, finishing current step...")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ============================================
# MUON COMPATIBILITY SHIMS FOR META-GOVERNOR
# ============================================ 
def get_alpha_well_depth(opt):
    """Safely retrieves the alpha well depth regardless of optimizer wrapper."""
    try:
        # ManualMuonOptimizer: reach into internal AdamW
        if hasattr(opt, 'adamw') and hasattr(opt.adamw, 'param_groups'):
            return opt.adamw.param_groups[0].get('weight_decay', 0.3)
        # Muon wrapper with param_groups
        if hasattr(opt, 'param_groups') and len(opt.param_groups) > 1:
            return opt.param_groups[1].get('weight_decay', 0.3)
        # Direct attribute fallback
        if hasattr(opt, 'adamw_wd'):
            return opt.adamw_wd
    except Exception:
        pass
    return 0.3

def set_alpha_well_depth(opt, new_wd):
    """Safely sets the alpha well depth regardless of optimizer wrapper."""
    try:
        # ManualMuonOptimizer: reach into internal AdamW
        if hasattr(opt, 'adamw') and hasattr(opt.adamw, 'param_groups'):
            opt.adamw.param_groups[0]['weight_decay'] = new_wd
            return
        # Muon wrapper with param_groups
        if hasattr(opt, 'param_groups') and len(opt.param_groups) > 1:
            opt.param_groups[1]['weight_decay'] = new_wd
            return
        # Direct attribute fallback
        if hasattr(opt, 'adamw_wd'):
            opt.adamw_wd = new_wd
    except Exception:
        pass

# 1. OPTIMIZER FACTORY (RUNS ONCE AT STARTUP)

# ============================================
# IMPORT ARCHITECTURE
# ============================================

try:
    from MYCELIA_architecture import MyceliaLM, MyceliaConfig
    print("🍄 Mycelia architecture loaded successfully - get ready for the ride!")
except ImportError:
    raise ImportError("MYCELIA_architecture not found!")

LESSONS_PATH = os.path.join(CKPT_DIR, 'mycelia_lessons.jsonl')
# ============================================
# PRESSURE TENSOR LOGGER
# ============================================

class PressureTensorLogger:
    def __init__(self):
        self.chi_history = []
        self.alert_count = 0
        self.last_alert_step = 0

    def update(self, info, step):
        chi = info.get('pressure_concentration', 0.0)
        self.chi_history.append(chi)
        if len(self.chi_history) > 1000:
            self.chi_history.pop(0)
        if chi > PRESSURE_CONCENTRATION_ALERT and step - self.last_alert_step > AUTO_TUNE_EVERY:
            dominant = info.get('dominant_governor', 'unknown')
            self.alert_count += 1
            self.last_alert_step = step
            return f"⚠️  Pressure concentration χ={chi:.2f} (dominant={dominant})"
        return None

# ============================================
# THROUGHPUT TRACKER
# ============================================

class ThroughputTracker:
    def __init__(self, tokens_per_step, total_tokens):
        self.tokens_per_step = tokens_per_step
        self.total_tokens = total_tokens
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_step = -1
        self._cache = None
        self.window_tokens = []
        self.window_times = []
        self.window_size = 50
        self._first_call = True

    def update(self, step):
        if step == self.last_step:
            return self._cache
        now = time.time()
        elapsed = now - self.start_time
        total_proc = step * self.tokens_per_step
        if self._first_call and self.last_step >= 0:
            tokens_since = (step - self.last_step) * self.tokens_per_step
            time_since = now - self.last_time
            self._first_call = False
        elif self.last_step >= 0:
            tokens_since = (step - self.last_step) * self.tokens_per_step
            time_since = now - self.last_time
        else:
            tokens_since = total_proc
            time_since = elapsed
        if time_since > 0 and tokens_since > 0:
            self.window_tokens.append(tokens_since)
            self.window_times.append(time_since)
            if len(self.window_tokens) > self.window_size:
                self.window_tokens.pop(0)
                self.window_times.pop(0)
        smoothed = sum(self.window_tokens) / sum(self.window_times) if self.window_times else 0
        remaining = max(0, self.total_tokens - total_proc)
        eta = remaining / smoothed if smoothed > 0 else 0
        self.last_time = now
        self.last_step = step
        raw_progress = (total_proc / self.total_tokens) * 100 if self.total_tokens > 0 else 0
        self._cache = {
            'step': step,
            'smoothed_tps': smoothed,
            'total_gb': total_proc / 1e9,
            'target_gb': self.total_tokens / 1e9,
            'progress': min(100.0, raw_progress),
            'raw_progress': raw_progress,
            'eta_h': eta / 3600,
            'elapsed_h': elapsed / 3600,
        }
        return self._cache

    def log(self, step):
        s = self.update(step)
        eta_str = str(timedelta(seconds=int(s['eta_h'] * 3600))) if s['eta_h'] > 0 else "N/A"
        elapsed_str = str(timedelta(seconds=int(s['elapsed_h'] * 3600)))
        progress_str = f"{s['progress']:.1f}%"
        if s['raw_progress'] > 100:
            progress_str = f"{s['raw_progress']:.1f}% (>{s['target_gb']:.1f}B target)"
        print(f"\n⏱️  Step {s['step']:,} | {s['smoothed_tps']:.0f} tok/s | "
              f"{s['total_gb']:.2f}/{s['target_gb']:.1f} GB | "
              f"{progress_str} | ETA {eta_str} | Elapsed {elapsed_str}")
        sys.stdout.flush()
        return s

# ============================================ 
# DATASETS
# ============================================
class StanfordDataset(IterableDataset):
    def __init__(self, bucket, prefix, tokenizer, max_seq_len=4096):
        self.bucket = bucket
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.target = max_seq_len + 1
        self.s3 = boto3.client('s3', region_name='eu-central-1')
        self.seen = set()
        self.epoch_count = 0

    def _stream(self):
        for key in STANFORD_ONLY:
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}/{key}")
                for line in obj['Body'].iter_lines():
                    if not line:
                        continue
                    try:
                        row = json.loads(line.decode('utf-8'))
                        text = row.get("text") or row.get("content") or ""
                        if len(text) < 50:
                            continue
                        h = hashlib.md5(text[:200].encode()).hexdigest()
                        if h in self.seen:
                            continue
                        self.seen.add(h)
                        yield text
                    except:
                        continue
            except BaseException as e:
                # Catch BaseException to suppress boto3 internal TypeErrors
                # ("catching classes that do not inherit from BaseException")
                # that can fire during connection teardown after a crash.
                print(f"⚠️  S3 error: {type(e).__name__}: {str(e)[:120]}")

    def __iter__(self):
        self.epoch_count += 1
        if self.epoch_count > 1:
            self.seen.clear()
            print(f"   🔄 Stanford epoch {self.epoch_count}: dedup cache cleared")
        while True:
            buffer = []
            for text in self._stream():
                try:
                    toks = self.tokenizer.encode(text, allowed_special="all")
                except:
                    toks = self.tokenizer.encode(text)
                for t in toks:
                    buffer.append(t)
                buffer.append(self.tokenizer.eos_token_id or 0)
                while len(buffer) >= self.target:
                    yield torch.tensor(buffer[:self.target], dtype=torch.long)
                    buffer = buffer[self.target:]

class S3FineWebDatasetChunked(IterableDataset):
    def __init__(self, bucket, prefix, max_seq_len=4096, max_chunks=500):
        self.bucket = bucket
        self.prefix = prefix
        self.target = max_seq_len + 1
        self.s3 = boto3.client('s3', region_name='eu-central-1')
        print("   📥 Loading FineWeb chunks...")
        sys.stdout.flush()
        chunks = []
        cont = None
        while True:
            kwargs = {'Bucket': bucket, 'Prefix': prefix}
            if cont:
                kwargs['ContinuationToken'] = cont
            resp = self.s3.list_objects_v2(**kwargs)
            chunks.extend([o['Key'] for o in resp.get('Contents', []) if o['Key'].endswith('.npy')])
            if not resp.get('IsTruncated'):
                break
            cont = resp.get('NextContinuationToken')
        chunks = sorted(chunks)[:max_chunks]
        print(f"   📚 Loading {len(chunks)} chunks...")
        self.all_tokens = []
        total = 0
        for i, ck in enumerate(chunks):
            try:
                data = self.s3.get_object(Bucket=bucket, Key=ck)['Body'].read()
                arr = np.load(io.BytesIO(data))
                self.all_tokens.append(arr)
                total += len(arr)
                if (i + 1) % 100 == 0:
                    print(f"      {i+1} chunks ({total:,} tokens, {total*4/1e9:.2f} GB)")
                    sys.stdout.flush()
            except Exception as e:
                print(f"⚠️  Chunk {ck} failed: {e}")
        print(f"   ✅ {len(self.all_tokens)} arrays | {total:,} tokens | {total*4/1e9:.2f} GB")
        sys.stdout.flush()

    def __iter__(self):
        buffer = []
        for arr in self.all_tokens:
            for t in arr:
                buffer.append(int(t))
                if len(buffer) >= self.target:
                    yield torch.tensor(buffer[:self.target], dtype=torch.long)
                    buffer = buffer[self.target:]
        if len(buffer) >= min(256, self.target):
            while len(buffer) < self.target:
                buffer.append(0)
            yield torch.tensor(buffer[:self.target], dtype=torch.long)

# class MixedDataset(IterableDataset):
    # def __init__(self, stanford, fineweb, stanford_weight=0.3, teaching=None, teaching_weight=0.0):
        # self.stanford = stanford
        # self.fineweb = fineweb
        # self.weight = stanford_weight
        # self.teaching_weight = teaching_weight if teaching is not None else 0.0
LESSONS_PATH = os.path.join(CKPT_DIR, 'mycelia_lessons.jsonl')
mixed = UniversalNpyDataset(
    "mycelia_s3_chunks", 
    max_seq_len=MAX_SEQ_LEN, 
    teaching_path=LESSONS_PATH, 
    teaching_weight=0.05, 

)
loader = DataLoader(mixed, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

def __iter__(self):
        import random
        s_iter = iter(self.stanford)
        f_iter = iter(self.fineweb)
        t_iter = iter(self.teaching) if self.teaching is not None else None
        while True:
            r = random.random()
            # ── v11.5: Teaching distillation stream (meta-governor lessons) ──
            if self.teaching is not None and r < self.teaching_weight:
                try:
                    yield next(t_iter)
                except StopIteration:
                    t_iter = iter(self.teaching)   # fresh pass re-reads the JSONL
                    try:
                        yield next(t_iter)
                    except StopIteration:
                        yield next(f_iter)         # no lessons yet → FineWeb
            elif r < self.teaching_weight + self.weight:
                try:
                    yield next(s_iter)
                except StopIteration:
                    yield next(f_iter)
            else:
                yield next(f_iter)

# ============================================
# v11.5: TEACHING DISTILLATION STREAM
# ============================================
# Lessons persisted by the Meta-Governor are fed back into the corpus.
# Mycelia has never seen physics/control-theory/LM-mechanics text; this
# distills the consortium's know-how about hidden-state dynamics into it.
# LESSONS_PATH = os.path.join(CKPT_DIR, 'mycelia_lessons.jsonl')

class TeachingDataset(IterableDataset):
    """Packs dated meta-governor lessons into 4096-token training sequences."""
    def __init__(self, path, tokenizer, max_seq_len=4096):
        self.path = path
        self.tokenizer = tokenizer
        self.target = max_seq_len + 1
        self.lessons = []

    def refresh(self) -> int:
        """Re-read the JSONL, dedup by lesson text."""
        self.lessons, seen = [], set()
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        text = (row.get('lesson') or '').strip()
                        if len(text) < 40 or text in seen:
                            continue
                        seen.add(text)
                        ctx = row.get('context') or {}
                        header = (f"[MYCELIA TEACHING | step {row.get('step', '?')} | "
                                  f"variable: {row.get('variable', '?')} | "
                                  f"action: {row.get('direction', '?')}")
                        if ctx:
                            # v3.44: surface the state→insight pairing
                            header += (f" | loss {ctx.get('loss', '?')} | coh {ctx.get('coherence', '?')}"
                                       f" | friction {ctx.get('friction', '?')} | mpc {ctx.get('mpc_intervention', '?')}")
                        header += "]\n"
                        self.lessons.append(header + text)
                    except Exception:
                        continue
        return len(self.lessons)

    def __iter__(self):
        if self.refresh() == 0:
            return                      # StopIteration → mixer falls back to FineWeb
        buffer, yielded = [], 0
        for _pass in range(24):         # repeat small corpora until samples form
            for lesson in self.lessons:
                try:
                    toks = self.tokenizer.encode(lesson)
                except Exception:
                    toks = self.tokenizer.encode(lesson, allowed_special="all")
                buffer.extend(toks)
                buffer.append(self.tokenizer.eos_token_id or 0)
                while len(buffer) >= self.target:
                    yield torch.tensor(buffer[:self.target], dtype=torch.long)
                    buffer = buffer[self.target:]
                    yielded += 1
            if yielded >= 4:
                break
        # iterator exhausts → mixer rebuilds it → refresh() picks up new lessons

def collate(batch):
    return torch.stack(batch)

# ============================================
# CHECKPOINT HELPERS
# ============================================

def _add_checksum(data):
    """Add SHA-256 checksum to checkpoint dict before saving."""
    try:
        checksum = hashlib.sha256(str(sorted(data.keys())).encode()).hexdigest()[:16]
        data['_checkpoint_checksum'] = checksum
    except Exception:
        pass
    return data

def _safe_load_checkpoint(path):
    """Load checkpoint with progress output and integrity checks."""
    if not os.path.exists(path):
        return None
    file_size = os.path.getsize(path)
    print(f"   📂 File: {path}")
    print(f"   💾 Size: {file_size/1e9:.2f} GB")

    if file_size < 1e6:
        print(f"   ⚠️  File too small ({file_size} bytes), skipping — probably corrupted")
        return None

    try:
        t0 = time.time()
        print(f"   ⏳ Loading... (may take 30-60s for {file_size/1e9:.1f}GB)")
        sys.stdout.flush()

        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
            print(f"   ✅ Loaded with weights_only=True")
        except Exception:
            print(f"   ⚠️  weights_only=True failed, retrying legacy...")
            sys.stdout.flush()
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            print(f"   ✅ Loaded with weights_only=False")

        elapsed = time.time() - t0
        print(f"   ⏱️  Load time: {elapsed:.1f}s")
        return ckpt
    except Exception as e:
        print(f"   🚨 Failed to load: {type(e).__name__}: {str(e)[:200]}")
        return None

def cleanup_checkpoints(ckpt_dir, keep=2):
    import glob
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "mycelia_step_*.pt")), key=os.path.getmtime)
    for old in ckpts[:-keep]:
        try:
            os.remove(old)
        except:
            pass
# ============================================
# MAIN
# ============================================

print("\n" + "="*70)
print("🍄 MYCELIA TRAINING v10.5 (1.5B Muon Integration)")
print("   Refactored: PressureState + TuningDecision + Lineage Receipts")
print("   v10.5: Alpha Potential Well + Meta-Governor Integration")
print("   1.5B: Muon + 8-bit AdamW Hybrid Optimizer")
print("   Stanford (30%) + FineWeb (70%) | No compression")
print("="*70)

print("\n📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id or 0
print(f"   Vocab: {tokenizer.vocab_size:,}")

print("\n🏗️ Building model...")
cfg = MyceliaConfig()
cfg.use_gradient_checkpointing = True  # MANDATORY for 1.5B on 16GB
cfg.max_seq_len = MAX_SEQ_LEN
cfg.vocab_size = 151643
cfg.compress_window = 128
cfg.compress_ratio = 8
cfg.use_compression = False
cfg.consensus_rounds = CONSENSUS_ROUNDS

cfg.ffn_norm_target = FFN_TARGET_START
cfg.alpha_norm_target = ALPHA_TARGET_START
cfg.soft_cap = 400.0
cfg.instability_target = 0.80  # v11.8: Recalibrated. Was 0.45; I-field baseline in compensatory regime is ~0.72.
cfg.control_gain = CONTROL_GAIN_DEFAULT
cfg.control_factor_floor = 0.7
cfg.predictive_scale = True

cfg.use_rate_governor = False
cfg.ffn_growth_ratio_max = 2.0
cfg.residual_growth_ratio_max = 1.5

cfg.use_gradual_transition = True
cfg.transition_duration = TRANSITION_DURATION
cfg.ffn_target_end = FFN_TARGET_END
cfg.alpha_target_end = ALPHA_TARGET_END

cfg.max_simultaneous_governors = MAX_SIMULTANEOUS_GOVERNORS

model = MyceliaLM(cfg)
if torch.cuda.is_available():
    # MANDATORY: Cast to bfloat16 to halve weights and Muon states
    model = model.to(device='cuda', dtype=torch.bfloat16)
else:
    model = model.to('cpu')
device = next(model.parameters()).device

print(f"   {sum(p.numel() for p in model.parameters()):,} params on {device}")

auto_tuner = GovernorAutoTuner(model, interval=AUTO_TUNE_EVERY)
pressure_logger = PressureTensorLogger()

# ============================================
# INSTANTIATE MUON HYBRID OPTIMIZER
# ============================================
# Muon: 2D weight matrices (polar decomposition keeps them orthogonal)
# 8-bit AdamW: embeddings, norms, alphas, biases, positional buffers
# Alpha potential well is applied via explicit alpha_regularization_loss (Hook A)
# because Muon does not apply standard L2 WD to 1D tensors.
opt = make_mycelia_optimizer(
    model,
    muon_lr=PEAK_LR,
    adamw_lr=PEAK_LR,
    muon_wd=WEIGHT_DECAY,
    # now let the explicit alpha_regularization_loss() do the work:
    adamw_wd=0.01,  # initial well depth for alphas decresed from 0.3
)

print(f"\n🔍 Optimizer: Muon + 8-bit AdamW hybrid | alpha_well_depth={get_alpha_well_depth(opt):.3f}")
print("="*70 + "\n")

total_steps = TOTAL_TOKENS_TARGET // (BATCH_SIZE * ACCUM_STEPS * MAX_SEQ_LEN)
scheduler = get_cosine_schedule_with_warmup(
    opt,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps,
)
if hasattr(opt, 'sync_adamw_lr'):
    opt.sync_adamw_lr(scheduler.get_last_lr()[0])
print(f"\n🔥 Scheduler: HF cosine | peak={PEAK_LR:.2e} | min={MIN_LR:.2e} | "
      f"warmup={WARMUP_STEPS} | total={total_steps:,}")
print(f"⚙️  Auto-tune: every {AUTO_TUNE_EVERY} steps | control_gain ∈ "
      f"[{CONTROL_GAIN_MIN}, {CONTROL_GAIN_MAX}]")
print(f"📈 Gradual transition: FFN {FFN_TARGET_START:.0f}→{FFN_TARGET_END:.0f} | "
      f"α {ALPHA_TARGET_START:.0f}→{ALPHA_TARGET_END:.0f} over {TRANSITION_DURATION:,} steps")
print(f"🛡️  Rate governor: DISABLED for first {TRANSITION_DURATION//2:,} steps")
print(f"🛡️  LR Burst: {'ENABLED' if ENABLE_LR_BURST else 'DISABLED'}")

# ============================================
# RESUME
# ============================================
# ─── RESUME (1.5B de novo — no legacy baggage)

start_epoch = 0
best_loss = float('inf')
best_step = 0
step = 0
ckpt = None

for path, label in [(BEST_CKPT, "🏆 BEST"), (LATEST_CKPT, "📂 LATEST")]:
    if os.path.exists(path):
        print(f"\n{'='*70}\n{label} CHECKPOINT\n{'='*70}")
        ckpt = _safe_load_checkpoint(path)
        if ckpt is not None:
            break

if ckpt is None:
    print(f"\n{'='*70}\n🚀 FRESH START — 1.5B de novo\n{'='*70}")

if ckpt is not None:
    # ── v10.7: Safe state_dict loading (filters shape mismatches) ──
    print("   🔄 Filtering checkpoint state_dict for shape mismatches...")
    ckpt_state = ckpt['model_state_dict']
    model_state = model.state_dict()
    
    safe_state = {}
    skipped = 0
    for k, v in ckpt_state.items():
        if k in model_state and model_state[k].shape != v.shape:
            # Skip geometric observable buffers that changed shape between versions
            print(f"      ⚠️  Skipping {k} (ckpt: {tuple(v.shape)} → model: {tuple(model_state[k].shape)})")
            skipped += 1
        else:
            safe_state[k] = v

    model.load_state_dict(safe_state, strict=False)  # ← ONLY here, inside the if block
    if skipped > 0:
        print(f"   ✅ Skipped {skipped} mismatched geometric buffers")

# NO load_state_dict call here. Fresh start means random init.

    if 'optimizer_state_dict' in ckpt:
        try:
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            print("   ✅ Optimizer state restored")
        except Exception as e:
            print(f"   ⚠️  Optimizer state load failed: {e}")
            print("   🔄 Muon momentum reset — training continues")

    model = model.to(device)
    print("   ✅ Model loaded")

    step = ckpt.get('global_step', 0)
    start_epoch = ckpt.get('epoch', 0) + 1
    prev_loss = ckpt.get('loss', 'N/A')
    best_loss_ckpt = ckpt.get('best_loss', float('inf'))

    if isinstance(prev_loss, (int, float)) and prev_loss > 0:
        print(f"   📊 Resumed: step={step:,} | loss={prev_loss:.4f}")
    else:
        print(f"   📊 Resumed: step={step:,} | loss=N/A")

    if isinstance(best_loss_ckpt, (int, float)) and best_loss_ckpt > 0:
        best_loss = best_loss_ckpt
        print(f"   🏆 Best loss: {best_loss:.4f}")

    # Scheduler: try restore, else rebuild
    scheduler_alive = False
    if 'scheduler_state_dict' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            current_lr = scheduler.get_last_lr()[0]
            if current_lr > 0:
                print(f"   ✅ Scheduler restored | LR={current_lr:.2e}")
                scheduler_alive = True
            else:
                print(f"   🚨 Scheduler dead: LR={current_lr:.2e}")
        except Exception as e:
            print(f"   ⚠️  Scheduler restore failed: {e}")

    if not scheduler_alive:
        new_total = max(step + 2_000_000, total_steps)
        scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=new_total)
        for _ in range(step):
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"   🔥 Scheduler rebuilt: total={new_total:,} | LR={current_lr:.2e}")

    for g in opt.param_groups:
        g['lr'] = scheduler.get_last_lr()[0]
        if hasattr(opt, 'sync_adamw_lr'):
                opt.sync_adamw_lr(scheduler.get_last_lr()[0])
        print(f"   🔄 LR synced: {opt.param_groups[0]['lr']:.2e}")

    if 'auto_tuner_state' in ckpt:
        try:
            auto_tuner.load_state(ckpt['auto_tuner_state'])
            print(f"   ✅ Auto-tuner restored")
        except Exception as e:
            print(f"   ⚠️  Auto-tuner restore failed: {e}")

    if 'alpha_well_history' in ckpt:
        auto_tuner._well_history = ckpt['alpha_well_history']
        auto_tuner._best_loss_since_well = ckpt.get('alpha_best_loss_since_well', float('inf'))
        print(f"   ✅ Alpha well history restored ({len(auto_tuner._well_history)} entries)")

    # ── v11.4: Free the checkpoint dict from CPU RAM ──
    # MUST sit OUTSIDE the alpha_well_history if (as a sibling) so it runs
    # for EVERY checkpoint type. BEST/emergency checkpoints carry no
    # alpha_well_history, and nesting the free inside that if skipped it.
    del ckpt_state   # model-weights view (~3 GB) — still referenced after del ckpt
    del safe_state   # filtered copy (same tensors)
    del ckpt         # top-level dict: optimizer (~5 GB) + scheduler + metadata
    gc.collect()
else:
    scheduler._resurrected_count = 0
    auto_tuner.scheduler_resurrected_count = 0
    for block in model.blocks:
        block.config.transition_start_step = 0
    print(f"\n{'='*70}\n🚀 FRESH START — 1.5B de novo\n{'='*70}")
    
# ============================================    
# DATA LOADING
# ============================================

print("\n📖 Loading datasets...")
mixed = UniversalNpyDataset("mycelia_s3_chunks", max_seq_len=MAX_SEQ_LEN)
loader = DataLoader(mixed, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
data_iter = iter(loader)
print("   ✅ Data ready")

tokens_per_step = BATCH_SIZE * ACCUM_STEPS * MAX_SEQ_LEN
actual_total_tokens = max(total_steps * tokens_per_step, step * tokens_per_step)
tracker = ThroughputTracker(tokens_per_step, actual_total_tokens)
print(f"\n⏱️  Tracker: {tracker.tokens_per_step:,} tok/step | {tracker.total_tokens/1e9:.1f}B total")

torch.cuda.empty_cache()
gc.collect()

# ============================================
# TRAINING LOOP 
# ============================================
print("\n" + "="*70)
print(f"🚀 EPOCH {start_epoch} — STEP {step:,}")
print("="*70 + "\n")

model.train()
losses_window = []
nan_count = 0
accum_counter = 0
mean_alpha_grad = 0.0
max_alpha_grad = 0.0

for b in model.blocks:
    b.mycelia.reset_stats()

# ============================================
# v11.7: PROFILER TRIGGER
# ============================================
PROFILE_STEP = 17_873  # Pick an upcoming step (e.g., 17500 or 18000)
prof = None           # ← initialized so stop() never raises NameError

for step in tqdm(range(step, step + 610351), desc="Training", initial=step):
        # ── v11.7 PROFILER START (inside loop so it fires at the correct step) ──
    if step == PROFILE_STEP and prof is None:
        print(f"\n🔬 Starting profiler for step {step}... (will add ~10% overhead this step)")
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            record_shapes=True,
            profile_memory=True
        )
        prof.start()
    if _shutdown_requested:
        print("\n🛑 Graceful shutdown, saving checkpoint...")
        try:
            _lineage_receipt = compute_lineage_receipt(
                model, step=step, epoch=start_epoch, cfg=cfg,
                batch_size=BATCH_SIZE, accum_steps=ACCUM_STEPS,
                max_seq_len=MAX_SEQ_LEN, peak_lr=PEAK_LR, min_lr=MIN_LR,
                warmup_steps=WARMUP_STEPS, grad_clip=GRAD_CLIP,
                weight_decay=WEIGHT_DECAY,
                ffn_target_start=FFN_TARGET_START, ffn_target_end=FFN_TARGET_END,
                alpha_target_start=ALPHA_TARGET_START, alpha_target_end=ALPHA_TARGET_END,
                transition_duration=TRANSITION_DURATION,
            )
            print(f"   📜 Lineage receipt: {_lineage_receipt.get('checkpoint_hash', 'N/A')[:16]}... "
                  f"(arch={_lineage_receipt.get('architecture_hash', 'N/A')[:8]}, "
                  f"loop={_lineage_receipt.get('training_loop_hash', 'N/A')[:8]})")

            emergency_ckpt = {
                'epoch': start_epoch,
                'global_step': step,
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'auto_tuner_state': auto_tuner.get_state(),
                'loss': float(losses_window[-1]) if losses_window else None,
                'best_loss': float(best_loss),
                'best_step': best_step,
                'timestamp': datetime.now().isoformat(),
                'lineage_receipt': _lineage_receipt,
            }
            emergency_ckpt = _add_checksum(emergency_ckpt)
            torch.save(emergency_ckpt, LATEST_CKPT)
            print(f"\n💾 Emergency save: step {step:,} → {LATEST_CKPT}")
        except Exception as e:
            print(f"\n🚨 Emergency save failed: {e}")
        break

    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch = next(data_iter)

    batch = batch.to(device)
    input_ids = batch[:, :-1].contiguous()
    targets = batch[:, 1:].contiguous()

    if USE_GRADUAL_TRANSITION:
        progress = min(1.0, (step - cfg.transition_start_step) / TRANSITION_DURATION)
        ease = 0.5 - 0.5 * math.cos(math.pi * progress)
        current_ffn_target = FFN_TARGET_START + ease * (FFN_TARGET_END - FFN_TARGET_START)
        current_alpha_target = ALPHA_TARGET_START + ease * (ALPHA_TARGET_END - ALPHA_TARGET_START)
        use_rate = progress > 0.5
        for block in model.blocks:
            # v11.9: After transition completes (progress=1.0), do NOT overwrite
            # targets that the auto-tuner or meta-governor may have adjusted.
            # The ramp is a bootstrap; once done, dynamic governance takes over.
            if progress < 1.0:
                block.ffn_norm_target = current_ffn_target
                block.alpha_norm_target = current_alpha_target
            block.use_rate_governor = use_rate
    else:
        current_ffn_target = cfg.ffn_norm_target
        current_alpha_target = cfg.alpha_norm_target
        use_rate = cfg.use_rate_governor
    else:
        current_ffn_target = cfg.ffn_norm_target
        current_alpha_target = cfg.alpha_norm_target
        use_rate = cfg.use_rate_governor

    with autocast(dtype=torch.bfloat16):
        logits_out = model(input_ids, padding_mask=(input_ids == PAD_ID),
                           use_compression=False, log_during_train=False)
        
        # ── v10.7: Chunked Cross-Entropy Loss ──
        if isinstance(logits_out, list):
            ce_loss = 0.0
            chunk_size = 128
            for i, chunk_logits in enumerate(logits_out):
                chunk_targets = targets[:, i*chunk_size:(i+1)*chunk_size]
                # Cast to fp32 HERE, chunk-by-chunk
                ce_loss += F.cross_entropy(
                    chunk_logits.float().reshape(-1, chunk_logits.size(-1)),
                    chunk_targets.reshape(-1),
                    ignore_index=PAD_ID
                )
            ce_loss = (ce_loss / len(logits_out)) / ACCUM_STEPS
        else:
            # Fallback for eval/inference
            ce_loss = F.cross_entropy(logits_out.float().reshape(-1, logits_out.size(-1)),
                                      targets.reshape(-1),
                                      ignore_index=PAD_ID) / ACCUM_STEPS
# ============================================
# HOOKS
# ============================================
        # ================================================================
        # HOOK A: Add alpha potential well loss (MANDATORY under Muon)
        # Muon does not apply standard L2 weight decay to 1D tensors
        # (alphas, norms, biases). This explicit quadratic well is the
        # only force preventing raw_alpha scalars from drifting to infinity.
        # ================================================================
        alpha_loss = model.alpha_regularization_loss() / ACCUM_STEPS    
        loss = ce_loss + alpha_loss

    is_bad_loss = torch.isnan(loss) or torch.isinf(loss)
    if is_bad_loss:
        nan_count += 1
        print(f"\n⚠️  NaN/Inf at step {step} (count: {nan_count})")
        if nan_count >= 2:
            if hasattr(opt, 'halve_all_lr'):
                opt.halve_all_lr()
            else:
                for g in opt.param_groups:
                    g['lr'] *= 0.5
            print(f"   🚨 LR halved to {opt.param_groups[0]['lr']:.2e}")
    if nan_count >= 3:
        print(f"   🚨🚨 Persistent NaN, rebuilding Muon hybrid optimizer")
        current_lr = opt.param_groups[0]['lr'] if hasattr(opt, 'param_groups') else PEAK_LR
        opt = make_mycelia_optimizer(
            model,
            muon_lr=current_lr * 2,
            adamw_lr=current_lr * 2,
            muon_wd=WEIGHT_DECAY,
            adamw_wd=get_alpha_well_depth(opt),
        )
        opt = make_mycelia_optimizer(
            model,
            muon_lr=PEAK_LR,
            adamw_lr=PEAK_LR,
            muon_wd=WEIGHT_DECAY,
            adamw_wd=0.01,  # ← was 0.3. Let the explicit alpha_regularization_loss() do the work.
        )
        
        nan_count = 0
        opt.zero_grad()
        continue

    nan_count = 0
    loss.backward()
    accum_counter += 1
    if accum_counter >= ACCUM_STEPS:
        # ================================================================
        # HOOK B: Log alpha gradient norms for meta-governor dashboard
        # ================================================================
        alpha_grad_norms = model.alpha_gradient_norms()
        mean_alpha_grad = sum(alpha_grad_norms) / len(alpha_grad_norms) if alpha_grad_norms else 0.0
        max_alpha_grad = max(alpha_grad_norms) if alpha_grad_norms else 0.0
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"\n⚠️  Bad gradients at step {step}, skipping step")
            opt.zero_grad()
        else:
            opt.step()
            opt.zero_grad()
            scheduler.step()
            if step % 100 == 0:
                lr_after_step = scheduler.get_last_lr()[0]
                if lr_after_step <= 0:
                    print(f"\n🚨 DEBUG: LR hit zero at step {step}!")
        accum_counter = 0

    losses_window.append(loss.item() * ACCUM_STEPS)
    if len(losses_window) > 1000:
        losses_window.pop(0)

    if hasattr(model, '_last_info') and model._last_info:
        auto_tuner.update_telemetry_emas(model._last_info)

    # Emergency checkpoint every LOG_EVERY
    if step % LOG_EVERY == 0 and step > 0:
        try:
            _lineage_receipt = compute_lineage_receipt(
                model, step=step, epoch=start_epoch, cfg=cfg,
                batch_size=BATCH_SIZE, accum_steps=ACCUM_STEPS,
                max_seq_len=MAX_SEQ_LEN, peak_lr=PEAK_LR, min_lr=MIN_LR,
                warmup_steps=WARMUP_STEPS, grad_clip=GRAD_CLIP,
                weight_decay=WEIGHT_DECAY,
                ffn_target_start=FFN_TARGET_START, ffn_target_end=FFN_TARGET_END,
                alpha_target_start=ALPHA_TARGET_START, alpha_target_end=ALPHA_TARGET_END,
                transition_duration=TRANSITION_DURATION,
            )
            print(f"   📜 Lineage receipt: {_lineage_receipt.get('checkpoint_hash', 'N/A')[:16]}... "
                  f"(arch={_lineage_receipt.get('architecture_hash', 'N/A')[:8]}, "
                  f"loop={_lineage_receipt.get('training_loop_hash', 'N/A')[:8]})")

            emergency_ckpt = {
                'epoch': start_epoch,
                'global_step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'auto_tuner_state': auto_tuner.get_state(),
                'scheduler_resurrected_count': getattr(scheduler, '_resurrected_count', 0),
                'loss': float(losses_window[-1]) if losses_window else None,
                'best_loss': float(best_loss),
                'best_step': best_step,
                'timestamp': datetime.now().isoformat(),
                'lineage_receipt': _lineage_receipt,
            }
            emergency_ckpt = _add_checksum(emergency_ckpt)
            torch.save(emergency_ckpt, LATEST_CKPT)
        except Exception as e:
            print(f"\n🚨 Emergency save failed at step {step}: {e}")

    # Logging every LOG_EVERY
    if step % LOG_EVERY == 0 and step > 0:
        current_avg_loss = float(np.mean(losses_window[-100:])) if losses_window else float('inf')
        current_lr = opt.param_groups[0]['lr'] if hasattr(opt, 'param_groups') and opt.param_groups else PEAK_LR

        # Build PressureState and get TuningDecision from auto-tuner
        _info_src = getattr(model, '_last_info', None) or {}

        # v10.5: Enrich telemetry with alpha potential well state
        if _info_src:
            # Collect alpha statistics across all layers
            alpha_attn_vals = []
            alpha_ffn_vals = []
            for block in model.blocks:
                if hasattr(block, 'raw_alpha_attn'):
                    alpha_attn_vals.append(float((1.0 + block.raw_alpha_attn).item()))
                elif hasattr(block, 'alpha_attn'):
                    alpha_attn_vals.append(float(block.alpha_attn.item()))
                if hasattr(block, 'raw_alpha_ffn'):
                    alpha_ffn_vals.append(float((1.0 + block.raw_alpha_ffn).item()))
                elif hasattr(block, 'alpha_ffn'):
                    alpha_ffn_vals.append(float(block.alpha_ffn.item()))

            if alpha_attn_vals:
                _info_src['alpha_attn_min']  = min(alpha_attn_vals)
                _info_src['alpha_attn_max']  = max(alpha_attn_vals)
                _info_src['alpha_attn_mean'] = sum(alpha_attn_vals) / len(alpha_attn_vals)
            if alpha_ffn_vals:
                _info_src['alpha_ffn_min']  = min(alpha_ffn_vals)
                _info_src['alpha_ffn_max']  = max(alpha_ffn_vals)
                _info_src['alpha_ffn_mean'] = sum(alpha_ffn_vals) / len(alpha_ffn_vals)

            # Current well state (Muon-safe accessor)
            _info_src['alpha_well_depth'] = get_alpha_well_depth(opt)
            _info_src['alpha_well_target'] = 1.0
            _info_src['alpha_well_history'] = getattr(auto_tuner, '_well_history', [])

            # Hook B telemetry (last values from most recent optim step)
            _info_src['alpha_grad_norm_mean'] = mean_alpha_grad
            _info_src['alpha_grad_norm_max'] = max_alpha_grad

        pressure = PressureState.from_telemetry(_info_src) if _info_src else None
        tune_decision = auto_tuner.tune(step, _info_src)

        # Apply LR multiplier if present
        if tune_decision.lr_multiplier is not None:
            for pg in opt.param_groups:
                pg['lr'] *= tune_decision.lr_multiplier

        # Log R and phase
        if pressure and tune_decision.r_action:
            regime_icon = "🔴" if pressure.ccr < 1.0 else "🟡" if pressure.ccr < 2.5 else "🟢"
            print(f"   🔥 R={pressure.ccr:.3f} {regime_icon} | "
                  f"phase={auto_tuner.phase:.3f} | action={tune_decision.r_action}")
        if tune_decision.phase_action:
            print(f"   🌊 Phase action: {tune_decision.phase_action}")
        if tune_decision.actions:
            for action in tune_decision.actions:
                print(f"   ⚙️  {action}")

        # ── v11.5: ADAPTIVE ALPHA POTENTIAL WELL (alpha-drift gated) ──────
        if _info_src:
            coherence = _info_src.get('coherence', 0.5)
            pressure_conc = _info_src.get('pressure_concentration', 0.5)
            mpc_ratio = _info_src.get('mpc_intervention_ratio', 0.0)
            dominant = _info_src.get('dominant_governor', 'none')
            # ── v11.5: deep-well requires ACTUAL alpha drift, not MPC dominance ──
            # MPC dominance (χ≈1.0) is structural at init, not instability.
            # Deepen the well only if alphas genuinely ran away (|α−1| > 1).
            _alpha_ffn_max = _info_src.get('alpha_ffn_max', 1.0)
            _alpha_attn_max = _info_src.get('alpha_attn_max', 1.0)
            _alpha_drift = (abs(_alpha_ffn_max - 1.0) > 1.0) or (abs(_alpha_attn_max - 1.0) > 1.0)
            if _alpha_drift and (pressure_conc > 0.7 or mpc_ratio > 0.25 or dominant == 'mpc'):
                target_wd = 0.5
                wd_reason = "deep_well (alpha_drift)"
            elif coherence > 0.6 and pressure_conc < 0.5:
                target_wd = 0.05
                wd_reason = "shallow_well (healthy)"
            elif hasattr(auto_tuner, '_last_loss') and current_avg_loss < auto_tuner._last_loss * 0.99:
                target_wd = 0.0
                wd_reason = "flat_well (loss_dropping)"
            else:
                # Default during learning: keep the well shallow so alphas stay free
                target_wd = 0.01
                wd_reason = "shallow_well (learning)"
            # Slingshot detection: deep well + sudden loss improvement → invert briefly
            if hasattr(auto_tuner, '_well_history') and len(auto_tuner._well_history) > 0:
                recent_wells = auto_tuner._well_history[-10:]
                avg_recent_wd = sum(recent_wells) / len(recent_wells)
                if avg_recent_wd > 0.4 and current_avg_loss < getattr(auto_tuner, '_best_loss_since_well', float('inf')) * 0.98:
                    target_wd = -0.1
                    wd_reason = "INVERTED (slingshot)"
                    auto_tuner._best_loss_since_well = current_avg_loss
            else:
                if not hasattr(auto_tuner, '_well_history'):
                    auto_tuner._well_history = []
                auto_tuner._best_loss_since_well = current_avg_loss
            # Smooth transition: don't jerk the well depth (Muon-safe)
            current_wd = get_alpha_well_depth(opt)
            new_wd = 0.9 * current_wd + 0.1 * target_wd
            set_alpha_well_depth(opt, max(-0.2, min(0.6, new_wd)))
            auto_tuner._well_history.append(float(get_alpha_well_depth(opt)))
            if len(auto_tuner._well_history) > 50:
                auto_tuner._well_history.pop(0)
            if step % (LOG_EVERY * 4) == 0:
                print(f"   🌍 Alpha well: WD={get_alpha_well_depth(opt):.3f} ({wd_reason}) | "
                      f"target={target_wd:.2f} | coh={coherence:.2f} | χ={pressure_conc:.2f}")
                alpha_attn_mean = _info_src.get('alpha_attn_mean', 1.0)
                alpha_ffn_mean  = _info_src.get('alpha_ffn_mean', 1.0)
                alpha_ffn_max   = _info_src.get('alpha_ffn_max', 1.0)
                print(f"   🍄 Alphas: attn_mean={alpha_attn_mean:.3f} "
                      f"ffn_mean={alpha_ffn_mean:.3f} ffn_max={alpha_ffn_max:.3f}")

        # Runtime scheduler health check
        scheduler_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else current_lr
        if scheduler_lr <= 0:
            print(f"\n   🚨🚨🚨 SCHEDULER FROZEN at step {step}: LR={scheduler_lr:.2e}")
            new_total = max(step + 1_000_000, getattr(scheduler, 'num_training_steps', step) + 500_000)
            scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=new_total)

            for _ in range(step):
                scheduler.step()
            recovered_lr = scheduler.get_last_lr()[0]
            if hasattr(opt, 'sync_adamw_lr'):
                opt.sync_adamw_lr(recovered_lr)
                scheduler._resurrected_count = getattr(scheduler, '_resurrected_count', 0) + 1
                auto_tuner.scheduler_resurrected_count = scheduler._resurrected_count
            print(f"   🔥 Scheduler RESURRECTED #{scheduler._resurrected_count}: total={new_total:,} | LR={recovered_lr:.2e}")
            if recovered_lr <= 0:
                raise RuntimeError(f"CRITICAL: Scheduler resurrection failed. LR={recovered_lr:.2e}")

        stats = tracker.log(step)

        coherence = 0.0
        early_var, late_var, delta = 0.0, 0.0, 0.0
        friction = ""
        cap_hit_ratio = 0.0
        max_raw_norm = 0.0
        mean_raw_norm = 0.0
        info = {}

        if hasattr(model, '_last_info') and model._last_info:
            info = model._last_info
            # v11.9: MPC dormancy tracker for predictor recalibration lock
            # Once MPC intervention drops to 0% for > 2,000 steps, the predictor
            # is recalibrating. control_gain suggestions are irrelevant noise.
            if not hasattr(auto_tuner, '_mpc_dormant_since'):
                auto_tuner._mpc_dormant_since = None
            if not hasattr(auto_tuner, '_control_gain_locked'):
                auto_tuner._control_gain_locked = False
            
            mpc_recent = info.get('mpc_intervention_ratio', 0.0)
            if mpc_recent < 0.05:
                if auto_tuner._mpc_dormant_since is None:
                    auto_tuner._mpc_dormant_since = step
                    print(f"   🎯 MPC dormancy detected at step {step}")
                elif step - auto_tuner._mpc_dormant_since > 2000 and not auto_tuner._control_gain_locked:
                    auto_tuner._control_gain_locked = True
                    print(f"   🔒 control_gain LOCKED: predictor recalibrating "
                          f"(MPC dormant since {auto_tuner._mpc_dormant_since})")
            else:
                if auto_tuner._control_gain_locked:
                    print(f"   🔓 control_gain UNLOCKED: MPC reactivated")
                auto_tuner._mpc_dormant_since = None
                auto_tuner._control_gain_locked = False
                
            # v11.8: MPC I-field baseline tracker — prevents threshold from sitting below natural operating point
            if not hasattr(auto_tuner, '_i_field_ema'):
                auto_tuner._i_field_ema = 0.70  # seed near observed compensatory baseline
            i_field_now = info.get('mean_instability_field', 0.0)
            if i_field_now > 0:
                auto_tuner._i_field_ema = 0.98 * auto_tuner._i_field_ema + 0.02 * i_field_now
            
            # Adaptive floor: target must stay ≥12% above I-field EMA, hard floor 0.45, ceiling 0.95
            adaptive_floor = min(0.95, max(0.45, auto_tuner._i_field_ema * 1.12))
            current_target = getattr(model.blocks[0], 'instability_target', cfg.instability_target)
            if current_target < adaptive_floor * 0.98:  # target lagging behind baseline by >2%
                new_target = adaptive_floor
                for block in model.blocks:
                    block.instability_target = new_target
                cfg.instability_target = new_target
                print(f"   🎯 MPC auto-recalibrated: instability_target → {new_target:.3f} (I-EMA={auto_tuner._i_field_ema:.3f})")            
            coherence = info.get('coherence', 0.0)
            early_var = info.get('early_var', 0.0)
            late_var = info.get('late_var', 0.0)
            delta = info.get('variance_delta', 0.0)
            cap_hit_ratio = info.get('soft_cap_hit_ratio', 0.0)
            max_raw_norm = info.get('max_raw_norm', 0.0)
            mean_raw_norm = info.get('mean_raw_norm', 0.0)
            if delta > 1.0:
                friction = "✅ DISSIPATED"
            elif delta < -1.0:
                friction = "🌋 DEEP DRIFT"
            elif early_var < 2.0 and late_var < 2.0:
                friction = "🟢 HARMONIZED"
            else:
                friction = "🟡 PROCESSING"

        coh_icon = "📈" if coherence > 0.8 else "📉" if coherence < 0.5 else "➡️"

        if USE_GRADUAL_TRANSITION:
            progress = min(1.0, (step - cfg.transition_start_step) / TRANSITION_DURATION)
            transition_status = f" | 📈 Transition: {progress*100:.0f}% (FFN={current_ffn_target:.0f} α={current_alpha_target:.0f})"
        else:
            transition_status = ""

        print(f"\n📊 Step {step:,} | Loss: {current_avg_loss:.4f} | LR: {current_lr:.2e} | 📉 Annealing{transition_status}")
        print(f"   Coherence: {coherence:.4f} {coh_icon}")
        if friction:
            print(f"   Friction: {friction} | early={early_var:.2f} late={late_var:.2f} Δ={delta:+.2f}")

        ffn_veto_ratio = info.get('ffn_veto_ratio', 0.0)
        mean_ffn_norm = info.get('mean_ffn_norm', 0.0)
        max_ffn_norm = info.get('max_ffn_norm', 0.0)
        if ffn_veto_ratio > 0 or mean_ffn_norm > 0:
            print(f"   FFNVeto: {ffn_veto_ratio*100:.1f}% mean_norm={mean_ffn_norm:.1f} max_norm={max_ffn_norm:.1f} | target={current_ffn_target:.0f}")

        alpha_scale_ratio = info.get('alpha_scale_ratio', 0.0)
        mean_alpha_scale = info.get('mean_alpha_scale', 1.0)
        mean_contrib_norm = info.get('mean_contrib_norm', 0.0)
        if alpha_scale_ratio > 0 or mean_contrib_norm > 0:
            print(f"   AlphaScale: {alpha_scale_ratio*100:.1f}% scale={mean_alpha_scale:.3f} contrib_norm={mean_contrib_norm:.1f} | target={current_alpha_target:.0f}")

        if cap_hit_ratio > 0 or max_raw_norm > 0:
            print(f"   SoftCap: hit={cap_hit_ratio*100:.1f}% max_raw={max_raw_norm:.1f} mean_raw={mean_raw_norm:.1f}")

        mpc_intervention_ratio = info.get('mpc_intervention_ratio', 0.0)
        mean_control_factor = info.get('mean_control_factor', 1.0)
        mean_instability_field = info.get('mean_instability_field', 0.0)
        if mpc_intervention_ratio > 0 or mean_instability_field > 0:
            print(f"\n   🔮 MPC: intervene={mpc_intervention_ratio*100:.1f}% control={mean_control_factor:.3f} I={mean_instability_field:.3f}")
            print(f"   📊 Pred: {info.get('mean_prediction', 0):.3f} | Conf: {info.get('mean_confidence', 1):.3f}")
            print(f"   📈 Dynamics: v={info.get('instability_velocity', 0):+.4f} a={info.get('instability_acceleration', 0):+.4f}")
            print(f"   🎯 Forecast Error: {info.get('forecast_error', 0):.3f}")

        instability_history = info.get('instability_field_history', [])
        confidence_history = info.get('confidence_history', [])
        if len(instability_history) >= 6:
            print(f"   I-field:  {' '.join([f'{v:.2f}' for v in instability_history])}")
        if len(confidence_history) >= 6:
            print(f"   Conf-field:{' '.join([f'{v:.2f}' for v in confidence_history])}")

        total_pressure = info.get('total_pressure', 0.0)
        pressure_conc = info.get('pressure_concentration', 0.0)
        dominant = info.get('dominant_governor', 'none')
        if total_pressure > 0:
            print(f"\n   🔥 Π={total_pressure:.1f} | χ={pressure_conc:.2f} | dominant={dominant}")
            pi_breakdown = info.get('pressure_by_governor', {})
            pi_str = ' '.join([f"{k}={v:.1f}" for k, v in pi_breakdown.items()])
            print(f"   🔥 Π breakdown: {pi_str}")
            pressure_alert = pressure_logger.update(info, step)
            if pressure_alert:
                print(f"   {pressure_alert}")

        rate_governor_hit = info.get('rate_governor_hit', 0.0)
        rate_scale_mean = info.get('rate_scale_mean', 1.0)
        ffn_growth = info.get('ffn_growth_ratio', 1.0)
        res_growth = info.get('residual_growth_ratio', 1.0)
        if use_rate and (rate_governor_hit > 0.01 or rate_scale_mean < 0.99):
            print(f"   📐 Rate Governor: hit={rate_governor_hit*100:.1f}% scale={rate_scale_mean:.3f} | ffn_growth={ffn_growth:.2f}x res_growth={res_growth:.2f}x")
        elif not use_rate:
            print(f"   📐 Rate Governor: DISABLED (transition progress < 50%)")

        sys.stdout.flush()

        active_govs = sum([
            1 if ffn_veto_ratio > 0.5 else 0,
            1 if alpha_scale_ratio > 0.5 else 0,
            1 if cap_hit_ratio > 0.5 else 0,
            1 if mpc_intervention_ratio > 0.5 else 0,
            1 if (use_rate and rate_governor_hit > 0.5) else 0,
        ])
        if active_govs > MAX_SIMULTANEOUS_GOVERNORS:
            print(f"   ⚠️  GOVERNOR INTERACTION GUARD: {active_govs} governors active, limit is {MAX_SIMULTANEOUS_GOVERNORS}")

        if USE_GRADUAL_TRANSITION and progress >= 1.0:
            if not hasattr(model, '_transition_complete_logged'):
                model._transition_complete_logged = True
                print(f"\n{'='*70}")
                print(f"✅ TRANSITION COMPLETE at step {step:,}")
                print(f"   Loss at completion: {current_avg_loss:.4f}")
                print(f"   v8.6 baseline best: 4.1132")
                print(f"   Δ from baseline: {current_avg_loss - 4.1132:+.4f}")
                print(f"{'='*70}\n")
                sys.stdout.flush()
                              
        # ================================================================
        # HOOK C: Compute Macroscopic R strictly per Zenodo 21402446
        # R = Π_α / (Π_FFN + Π_MPC)
        # Three regimes:  R > 1.5  → CONSTRUCTIVE
        #                 R > 0.8  → MARGINAL (critical region)
        #                 else     → COMPENSATORY
        # ================================================================

        # ── Primary: canonical pressure_by_governor from MODEL level ──
        # Aggregated in MyceliaLM.forward(); SAME source the "🔥 Π breakdown"
        # telemetry uses, so Hook C and the breakdown can never disagree.
        # (It is NOT in block._last_info — that lookup would always be {}.)
        model_pbg = (getattr(model, '_last_info', None) or {}).get('pressure_by_governor', {})

        # Single pass: raw work (diagnostic) + fallback pressure reconstruction
        total_alpha_work = total_ffn_work = total_mpc_work = 0.0
        fb_alpha = fb_ffn = fb_mpc = 0.0
        for block in model.blocks:
            if hasattr(block, '_last_info') and block._last_info:
                ib = block._last_info
                aw = ib.get('alpha_work', 0.0)
                fw = ib.get('ffn_work', 0.0)
                mw = ib.get('mpc_work', 0.0)
                total_alpha_work += aw
                total_ffn_work   += fw
                total_mpc_work   += mw
                # work × norm, matching the architecture's own defaults (0.0)
                fb_alpha += aw * ib.get('mean_contrib_norm', 0.0)
                fb_ffn   += fw * ib.get('mean_ffn_norm', 0.0)
                fb_mpc   += mw * ib.get('mean_instability_field', 0.0)

        if model_pbg:
            pi_alpha = float(model_pbg.get('alpha', 0.0))
            pi_ffn   = float(model_pbg.get('ffn', 0.0))
            pi_mpc   = float(model_pbg.get('mpc', 0.0))
        else:
            pi_alpha, pi_ffn, pi_mpc = fb_alpha, fb_ffn, fb_mpc

        denominator = pi_ffn + pi_mpc
        if denominator > 1e-8:
            R = pi_alpha / denominator
        else:
            R = float('inf') if pi_alpha > 1e-8 else 0.0

        if R > 1.5:
            regime_state = "CONSTRUCTIVE"
        elif R > 0.8:
            regime_state = "MARGINAL"
        else:
            regime_state = "COMPENSATORY"

        R_print = f"{R:.3f}" if R != float('inf') else "∞"
        print(f"   🔭 Macroscopic R={R_print} | Regime: {regime_state} "
              f"(Π_α={pi_alpha:.2f}, Π_FFN={pi_ffn:.2f}, Π_MPC={pi_mpc:.2f} | "
              f"raw: α={total_alpha_work:.2f} ffn={total_ffn_work:.2f} mpc={total_mpc_work:.2f})")

        # v11.8: Compensatory regime MPC relief — continuous floor, no one-shot flag
        if regime_state == "COMPENSATORY" and cfg.instability_target < 0.80:
            for block in model.blocks:
                block.instability_target = 0.80
            cfg.instability_target = 0.80
            print(f"   🎯 MPC compensatory relief: instability_target → 0.80 (R={R_print})")
        # v11.9: Alpha channel wake-up call
        # If R has been pinned near zero for >5k steps, lower alpha_norm_target
        # temporarily to force AlphaScale activation and build alpha pressure.
        if R < 0.1 and regime_state == "COMPENSATORY":
            if not hasattr(auto_tuner, '_alpha_wake_step'):
                auto_tuner._alpha_wake_step = step
            elif step - auto_tuner._alpha_wake_step > 5000:
                current_alpha_target = getattr(model.blocks[0], 'alpha_norm_target', 150.0)
                if current_alpha_target > 50.0:
                    new_target = max(50.0, current_alpha_target * 0.9)
                    for block in model.blocks:
                        block.alpha_norm_target = new_target
                    print(f"   🍄 Alpha wake-up: alpha_norm_target → {new_target:.0f} "
                          f"(R={R:.3f}, step={step})")
                    auto_tuner._alpha_wake_step = step  # reset timer

        # ================================================================
        # v11.9: PRE-META telemetry enrichment
        # Compute predictor recalibration status BEFORE the council sees
        # the telemetry, so advisors know which levers are locked and what
        # regime the model is in. Attach payload to model because
        # KIMIMetaGovernor has self.model and reads _last_info from it.
        # ================================================================
        _mpc_recent = _info_src.get('mpc_intervention_ratio', 0.0)
        _forecast_err = _info_src.get('forecast_error', 0.0)

        # Grace-period detector (lifted from downstream action handler)
        if not hasattr(auto_tuner, '_mpc_release_step'):
            auto_tuner._mpc_release_step = 0
        _in_grace = False
        if _mpc_recent < 0.05:
            if auto_tuner._mpc_release_step == 0:
                auto_tuner._mpc_release_step = step
                print(f"   🎯 MPC predictor recalibration started at step {step}")
            elif step - auto_tuner._mpc_release_step < 15000:
                _in_grace = True
        else:
            auto_tuner._mpc_release_step = 0

        # Enrichment payload — meta_governor.py will read this when building
        # the advisor prompt. This is how we "force the council to explore
        # other levers": we explicitly tell them control_gain is frozen.
        model._meta_enrichment = {
            'predictor_recalibrating': _in_grace,
            'recalibration_age': step - auto_tuner._mpc_release_step if _in_grace else 0,
            'forecast_error': _forecast_err,
            'mpc_intervention_ratio': _mpc_recent,
            'R_regime': regime_state,
            'R_value': 999.0 if R == float('inf') else float(R),
            'control_gain_locked': _in_grace and _forecast_err > 0.5,
            'control_gain_floor': 0.85 if (R < 0.1) else 0.01,
            'alpha_wake_active': (
                hasattr(auto_tuner, '_alpha_wake_step') and
                (step - auto_tuner._alpha_wake_step) > 5000
            ),
            'alpha_norm_target': getattr(model.blocks[0], 'alpha_norm_target', 150.0),
            'instability_target': cfg.instability_target,
            'well_depth': get_alpha_well_depth(opt),
            # Explicitly list what the council MAY touch vs. what is frozen
            'locked_levers': (
                ['control_gain'] if (_in_grace and _forecast_err > 0.5) else []
            ),
            'recommended_levers': [
                'alpha_well_depth',
                'instability_target',
                'alpha_norm_target',
            ] + ([] if (_in_grace and _forecast_err > 0.5) else ['control_gain']),
        }

        # ================================================================
        # META-GOVERNOR
        # ================================================================
        meta_actions = integrate_meta_governor(
            model=model, auto_tuner=auto_tuner,
            step=step, current_loss=current_avg_loss,
            current_lr=current_lr, scheduler=scheduler,
            log_every=LOG_EVERY, local_only=False,
        )
        if meta_actions and meta_actions[0] not in [
            "CIRCUIT_BREAKER_ACTIVE", "NO_CONSENSUS",
            "LOCAL_RULE: no_action", "META_RATE_LIMITED", "META_INTERACTION_GUARD"
        ]:
            print(f"   🛠  Meta-Governor: {meta_actions}")

            # v10.5: Apply alpha well depth decisions from meta-governor
            # Uses Muon-safe get/set shims so commands reach the 8-bit AdamW
            # state inside the Muon wrapper without IndexError.
            for action in meta_actions:
                if action.startswith("ALPHA_WELL_DEPTH_"):
                    parts = action.split("_")
                    if len(parts) >= 4:
                        direction = parts[3]
                        try:
                            value = float(parts[4]) if len(parts) > 4 else None
                        except (ValueError, IndexError):
                            value = None

                        current_wd = get_alpha_well_depth(opt)

                        if direction == "raise" and value is not None:
                            new_wd = min(0.6, current_wd + value)
                        elif direction == "lower" and value is not None:
                            new_wd = max(-0.2, current_wd - value)
                        elif direction == "set" and value is not None:
                            new_wd = max(-0.2, min(0.6, value))
                        else:
                            continue

                        set_alpha_well_depth(opt, new_wd)
                        print(f"   🌍 Meta-Gov alpha well: "
                              f"{current_wd:.3f} → {new_wd:.3f} ({direction})")

                elif action.startswith("ALPHA_WELL_INVERT_"):
                    parts = action.split("_")
                    if len(parts) >= 4:
                        direction = parts[3]
                        try:
                            value = float(parts[4]) if len(parts) > 4 else -0.1
                        except (ValueError, IndexError):
                            value = -0.1

                        if direction in ("raise", "set"):
                            set_alpha_well_depth(opt, max(-0.2, value))
                            print(f"   🌍 Meta-Gov alpha well INVERTED: "
                                  f"{value:.3f} (slingshot)")
                        elif direction in ("lower", "release"):
                            set_alpha_well_depth(opt, 0.3)
                            print(f"   🌍 Meta-Gov alpha well restored: 0.3")
                            
# ============================================
# OPTION "B" — v11.8 complete actuator block
# ============================================
                # ── v11.6 Option B: Actuate global-var suggestions ──
                # control_gain / instability_target arrive as SUGGEST_ strings.
                # Previously missing entirely on SageMaker → suggestions were printed but dropped.
                elif action.startswith("SUGGEST_control_gain_"):
                    parts = action.split("_")
                    if len(parts) >= 5:
                        direction = parts[-2]
                        try:
                            multiplier = float(parts[-1])
                        except (ValueError, IndexError):
                            multiplier = None
                        if multiplier is not None:
                            # v11.9: Predictor recalibration grace period.
                            # After MPC intervention drops to 0%, forecast error spikes
                            # because the predictor was trained on 100% intervention regime.
                            # Do NOT lower control_gain in response to recalibration error.
                            mpc_recent = _info_src.get('mpc_intervention_ratio', 0.0)
                            forecast_err = _info_src.get('forecast_error', 0.0)
                            in_grace = False
                            if not hasattr(auto_tuner, '_mpc_release_step'):
                                auto_tuner._mpc_release_step = 0
                            if mpc_recent < 0.05:
                                if auto_tuner._mpc_release_step == 0:
                                    auto_tuner._mpc_release_step = step
                                    print(f"   🎯 MPC predictor recalibration started at step {step}")
                                elif step - auto_tuner._mpc_release_step < 15000:
                                    in_grace = True
                            else:
                                auto_tuner._mpc_release_step = 0

                            if direction == "lower" and in_grace and forecast_err > 0.5:
                                print(f"   🎛️ Meta-Gov REJECTED: control_gain lower blocked "
                                      f"(predictor recalibrating: forecast_err={forecast_err:.3f}, "
                                      f"grace={(step - auto_tuner._mpc_release_step)} steps)")
                                continue

                            for block in model.blocks:
                                current = getattr(block, 'control_gain', CONTROL_GAIN_DEFAULT)
                                if direction == "set":
                                    new_val = multiplier
                                elif direction == "raise":
                                    new_val = current / multiplier
                                else:  # lower
                                    new_val = current * multiplier
                                # v11.9: Hard floor prevents death spiral during recalibration
                                # v11.9: Hard floor during compensatory regime.
                                # R=0 means alpha channel is dormant; lowering gain
                                # further does not help predictor recalibration.
                                floor = 0.85 if R < 0.1 else 0.01
                                new_val = max(floor, min(10.0, new_val))
                                block.control_gain = new_val
                            cfg.control_gain = model.blocks[-1].control_gain
                            op = "÷" if direction == "raise" else "×"
                            print(f"   🎛️ Meta-Gov APPLIED: control_gain → {cfg.control_gain:.3f} ({direction} {op}{multiplier:.3f})"
                                  + (" [GRACE]" if in_grace else ""))

                elif action.startswith("SUGGEST_instability_target_"):
                    parts = action.split("_")
                    if len(parts) >= 5:
                        direction = parts[-2]
                        try:
                            multiplier = float(parts[-1])
                        except (ValueError, IndexError):
                            multiplier = None
                        if multiplier is not None:
                            # v11.9: Reject control_gain suggestions during predictor recalibration
                            if getattr(auto_tuner, '_control_gain_locked', False):
                                print(f"   🎛️ Meta-Gov REJECTED: control_gain {direction} blocked "
                                      f"(predictor recalibrating since step {auto_tuner._mpc_dormant_since})")
                                continue
                            for block in model.blocks:
                                current = getattr(block, 'instability_target', 0.45)
                                # v11.8 FIX: "raise" widens deadband (÷mult), "lower" tightens (×mult)
                                if direction == "set":
                                    new_val = multiplier
                                elif direction == "raise":
                                    new_val = current / multiplier
                                else:  # lower
                                    new_val = current * multiplier
                                block.instability_target = max(0.01, min(1.0, new_val))
                            cfg.instability_target = model.blocks[-1].instability_target
                            op = "÷" if direction == "raise" else "×"
                            print(f"   🎯 Meta-Gov APPLIED: instability_target → {cfg.instability_target:.3f} ({direction} {op}{multiplier:.3f})")

# ============================================
# Well History update
# ============================================
            # v10.5: Update well history after meta-governor actions
            if not hasattr(auto_tuner, '_well_history'):
                auto_tuner._well_history = []
            auto_tuner._well_history.append(get_alpha_well_depth(opt))
            if len(auto_tuner._well_history) > 256:
                auto_tuner._well_history = auto_tuner._well_history[-256:]

        if hasattr(auto_tuner, '_meta_governor'):
            status = auto_tuner._meta_governor.get_status()
            cb_icon = "🔴" if status['circuit_breaker'] else "🟢"
            print(f"   📡 Meta-Gov: CB={cb_icon} | pending={status['pending_verifications']} | "
                  f"history={status['telemetry_history_size']} | conf={status['expert_confidences']}")
# ============================================
# Checkpoints
# ============================================

        # Best checkpoint
        if current_avg_loss < best_loss:
            best_loss = current_avg_loss
            best_step = step
            _lineage_receipt = compute_lineage_receipt(
                model, step=step, epoch=start_epoch, cfg=cfg,
                batch_size=BATCH_SIZE, accum_steps=ACCUM_STEPS,
                max_seq_len=MAX_SEQ_LEN, peak_lr=PEAK_LR, min_lr=MIN_LR,
                warmup_steps=WARMUP_STEPS, grad_clip=GRAD_CLIP,
                weight_decay=WEIGHT_DECAY,
                ffn_target_start=FFN_TARGET_START, ffn_target_end=FFN_TARGET_END,
                alpha_target_start=ALPHA_TARGET_START, alpha_target_end=ALPHA_TARGET_END,
                transition_duration=TRANSITION_DURATION,
            )
            ckpt_data = {
                'epoch': start_epoch,
                'global_step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'auto_tuner_state': auto_tuner.get_state(),
                'loss': float(current_avg_loss),
                'best_loss': float(best_loss),
                'coherence': float(coherence),
                'friction': friction,
                'early_var': float(early_var),
                'late_var': float(late_var),
                'delta': float(delta),
                'timestamp': datetime.now().isoformat(),
                'lineage_receipt': _lineage_receipt,
            }
            ckpt_data = _add_checksum(ckpt_data)
            try:
                torch.save(ckpt_data, BEST_CKPT)
                # ── v11.3: NO torch.load verification (OOM risk) ──
                _sz = os.path.getsize(BEST_CKPT) / 1e9
                if _sz > 1.0:
                    print(f"\n🏆 BEST SAVED: {best_loss:.4f} at step {step:,} ({_sz:.2f} GB)")
            except Exception as e:
                print(f"\n🚨 BEST save FAILED at step {step}: {e}")
            sys.stdout.flush()

    # Regular checkpoint
    if step % SAVE_EVERY == 0 and step > 0:
        _lineage_receipt = compute_lineage_receipt(
            model, step=step, epoch=start_epoch, cfg=cfg,
            batch_size=BATCH_SIZE, accum_steps=ACCUM_STEPS,
            max_seq_len=MAX_SEQ_LEN, peak_lr=PEAK_LR, min_lr=MIN_LR,
            warmup_steps=WARMUP_STEPS, grad_clip=GRAD_CLIP,
            weight_decay=WEIGHT_DECAY,
            ffn_target_start=FFN_TARGET_START, ffn_target_end=FFN_TARGET_END,
            alpha_target_start=ALPHA_TARGET_START, alpha_target_end=ALPHA_TARGET_END,
            transition_duration=TRANSITION_DURATION,
        )
        print(f"   📜 Lineage receipt: {_lineage_receipt.get('checkpoint_hash', 'N/A')[:16]}... "
              f"(arch={_lineage_receipt.get('architecture_hash', 'N/A')[:8]}, "
              f"loop={_lineage_receipt.get('training_loop_hash', 'N/A')[:8]})")

        ckpt_data = {
            'epoch': start_epoch,
            'global_step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'auto_tuner_state': auto_tuner.get_state(),
            'scheduler_resurrected_count': getattr(scheduler, '_resurrected_count', 0),
            'loss': float(losses_window[-1]) if losses_window else None,
            'avg_loss_100': float(np.mean(losses_window[-100:])) if len(losses_window) >= 100 else None,
            'best_loss': float(best_loss),
            'best_step': best_step,
            'timestamp': datetime.now().isoformat(),
            'lineage_receipt': _lineage_receipt,
            'alpha_well_history': getattr(auto_tuner, '_well_history', []),
            'alpha_best_loss_since_well': getattr(auto_tuner, '_best_loss_since_well', float('inf')),
        }
        path = os.path.join(CKPT_DIR, f"mycelia_step_{step:05x}.pt")
        ckpt_data = _add_checksum(ckpt_data)
        try:
            torch.save(ckpt_data, path)
            torch.save(ckpt_data, LATEST_CKPT)
            print(f"\n💾 Checkpoint: step {step:,} → {path}")
            cleanup_checkpoints(CKPT_DIR)
        except Exception as e:
            print(f"\n🚨 Checkpoint save failed: {e}")
        sys.stdout.flush()

    if step % CACHE_CLEAN_EVERY == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    # ── v11.7 PROFILER STOP ──
    if prof is not None:
        prof.stop()
        print("\n" + "="*80)
        print("🔬 TOP 25 OPS BY CUDA TIME")
        print("="*80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        print("\n" + "="*80)
        print("🔬 TOP 15 OPS BY CPU TIME")
        print("="*80)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        prof.export_chrome_trace(f"mycelia_trace_step_{step}.json")
        print(f"\n💾 Trace saved → mycelia_trace_step_{step}.json")
        print("="*80 + "\n")
        prof = None  # ← prevent re-running on subsequent steps
# ============================================
# Clean shutdown
# ============================================

if hasattr(auto_tuner, '_meta_governor'):
    if hasattr(auto_tuner._meta_governor, 'shutdown'):
        auto_tuner._meta_governor.shutdown()
        print("   📡 Meta-Governor shutdown complete")

# ============================================
# FINAL SAVE
# ============================================
print("\n" + "="*70)
print("💾 Final save...")

_lineage_receipt = compute_lineage_receipt(
    model, step=step, epoch=start_epoch, cfg=cfg,
    batch_size=BATCH_SIZE, accum_steps=ACCUM_STEPS,
    max_seq_len=MAX_SEQ_LEN, peak_lr=PEAK_LR, min_lr=MIN_LR,
    warmup_steps=WARMUP_STEPS, grad_clip=GRAD_CLIP,
    weight_decay=WEIGHT_DECAY,
    ffn_target_start=FFN_TARGET_START, ffn_target_end=FFN_TARGET_END,
    alpha_target_start=ALPHA_TARGET_START, alpha_target_end=ALPHA_TARGET_END,
    transition_duration=TRANSITION_DURATION,
)
print(f"   📜 Lineage receipt: {_lineage_receipt.get('checkpoint_hash', 'N/A')[:16]}... "
      f"(arch={_lineage_receipt.get('architecture_hash', 'N/A')[:8]}, "
      f"loop={_lineage_receipt.get('training_loop_hash', 'N/A')[:8]})")

final = {
    'epoch': start_epoch,
    'global_step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'auto_tuner_state': auto_tuner.get_state(),
    'scheduler_resurrected_count': getattr(scheduler, '_resurrected_count', 0),
    'loss': float(losses_window[-1]) if losses_window else None,
    'avg_loss_100': float(np.mean(losses_window[-100:])) if len(losses_window) >= 100 else None,
    'best_loss': float(best_loss),
    'best_step': best_step,
    'timestamp': datetime.now().isoformat(),
    'lineage_receipt': _lineage_receipt,
    'alpha_well_history': getattr(auto_tuner, '_well_history', []),
    'alpha_best_loss_since_well': getattr(auto_tuner, '_best_loss_since_well', float('inf')),
}
final = _add_checksum(final)

for ckpt_path, label in [(LATEST_CKPT, "LATEST"), (BEST_CKPT, "BEST")]:
    try:
        torch.save(final, ckpt_path)
        print(f"   ✅ {label} saved")
    except Exception as e:
        print(f"   🚨 {label} save failed: {e}")

print("\n" + "="*70)
print("🍄 MYCELIA TRAINING v10.5 (1.5B Muon) COMPLETE")
print(f"   Final step: {step:,}")
print(f"   Best loss: {best_loss:.4f} at step {best_step:,}")
print("="*70)