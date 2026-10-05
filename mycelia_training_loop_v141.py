# ============================================
# MYCELIA Training Loop — v12.9 (1.5B Muon Integration)
# v12.8: Cold-Start Artifact Suppressor Integration
# v12.9: Independent SNLI Evaluation (Sequential/Inline) + MASSIF Latent Stability Observatory
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
import threading
import queue
import csv
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
from collections import Counter
from datasets import load_dataset

# ─── NEW REFACTOR IMPORTS ────────────────────────────────────────────────
from optimization_state import PressureState, OptimizationRegime
from governor_auto_tuner import GovernorAutoTuner, TuningDecision
from lineage_receipt import compute_lineage_receipt
from internal_meta_governor import integrate_meta_governor

# ── 1.5B OPTIMIZER IMPORTS
from manual_muon_optimizer import ManualMuonOptimizer, make_mycelia_optimizer

warnings.filterwarnings('ignore')

# --- CONFIG HASHING FOR REPRODUCIBILITY ---
def generate_run_fingerprint(config_dict):
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:12]

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
        self.yield_counts = Counter()
        self.cumulative_sizes = []
        cumsum = 0
        for size in self.shard_sizes:
            cumsum += size
            self.cumulative_sizes.append(cumsum)

        self.shard_domains = []
        for s in self.shards:
            name = os.path.basename(s)
            domain = name.rsplit('_p', 1)[0] if '_p' in name else name.replace('.npy', '')
            self.shard_domains.append(domain)

        self.indices = list(range(self.total_chunks))
        random.shuffle(self.indices)
        self.current_pos = 0
        self.epoch_count = 1
        self.seen_in_epoch = 0

        self.teaching_weight = teaching_weight
        self.teaching_lessons = []
        self.teaching_path = teaching_path
        self.tokenizer = tokenizer
        self.teaching_mtime = 0
        self.teaching_triggers = 0
        self.teaching_log_interval = 20

        if self.teaching_path and self.tokenizer:
            self.refresh_teaching()

        self.yield_counts = Counter()

    def refresh_teaching(self):
        if not self.teaching_path or not os.path.exists(self.teaching_path): return
        current_mtime = os.path.getmtime(self.teaching_path)
        if current_mtime == self.teaching_mtime: return
        self.teaching_mtime = current_mtime

        new_lessons, seen = [], set()
        with open(self.teaching_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = (row.get('lesson') or '').strip()
                    if len(text) < 40 or text in seen: continue
                    seen.add(text)
                    ctx = row.get('context') or {}
                    header = (f"[MYCELIA TEACHING | step {row.get('step', '?')} | variable: {row.get('variable', '?')} | action: {row.get('direction', '?')}")
                    if ctx: header += (f" | loss {ctx.get('loss', '?')} | coh {ctx.get('coherence', '?')} | friction {ctx.get('friction', '?')} | mpc {ctx.get('mpc_intervention', '?')}")
                    header += "]\n"
                    toks = self.tokenizer.encode(header + text, add_special_tokens=False)
                    if self.tokenizer.eos_token_id is not None: toks.append(self.tokenizer.eos_token_id)
                    for i in range(0, len(toks) - self.target_len + 1, self.target_len):
                        chunk = toks[i:i + self.target_len]
                        if len(chunk) == self.target_len: new_lessons.append(torch.tensor(chunk, dtype=torch.long))
                except Exception: continue

        self.teaching_lessons = new_lessons
        if len(self.teaching_lessons) > 0: print(f"🧠 Refreshed Teaching Stream: {len(self.teaching_lessons)} lesson chunks loaded.")

    def _global_to_shard(self, global_idx):
        shard_idx = bisect.bisect_right(self.cumulative_sizes, global_idx)
        row_idx = global_idx if shard_idx == 0 else global_idx - self.cumulative_sizes[shard_idx - 1]
        return shard_idx, row_idx

    def __len__(self): return self.total_chunks

    def __getitem__(self, idx):
        if not hasattr(self, 'yield_counts'): self.yield_counts = Counter()
        if self.teaching_lessons and self.teaching_weight > 0 and random.random() < self.teaching_weight:
            if random.random() < 0.01: self.refresh_teaching()
            self.teaching_triggers += 1
            if self.teaching_triggers % self.teaching_log_interval == 0:
                print(f"🧠 Teaching trigger #{self.teaching_triggers} | epoch={self.epoch_count} | pos={self.seen_in_epoch}/{self.total_chunks:,} | lessons_loaded={len(self.teaching_lessons)}")
            self.yield_counts['🧠 TEACHING'] += 1
            return random.choice(self.teaching_lessons)

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

        domain = self.shard_domains[shard_idx]
        self.yield_counts[f"📁 {domain}"] += 1
        
        if self.seen_in_epoch % 50_000 == 0:
            pct = (self.seen_in_epoch / self.total_chunks) * 100
            print(f"📊 Epoch {self.epoch_count} | {pct:.1f}% seen | current domain: {domain} (shard {shard_idx}/{len(self.shards)})")

        return torch.tensor(chunk, dtype=torch.long)

# ============================================
# CONFIGURATION
# ============================================
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
ACCUM_STEPS = 32
WEIGHT_DECAY = 0.01
GRAD_CLIP = 2.0
SAVE_EVERY = 1000
LOG_EVERY = 20   
CACHE_CLEAN_EVERY = 1000

PEAK_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 500
TOTAL_TOKENS_TARGET = 50_000_000_000

ENABLE_LR_BURST = True
CONSENSUS_ROUNDS = 2
AUTO_TUNE_EVERY = 10000

CONTROL_GAIN_MIN = 0.5
CONTROL_GAIN_MAX = 1.5
CONTROL_GAIN_DEFAULT = 1.0

ADAPTIVE_TARGETS_ENABLED = True
USE_GRADUAL_TRANSITION = True
TRANSITION_DURATION = 1000
FFN_TARGET_START = 30.0
FFN_TARGET_END = 50.0
ALPHA_TARGET_START = 25.0
ALPHA_TARGET_END = 30.0

MAX_SIMULTANEOUS_GOVERNORS = 2
PRESSURE_CONCENTRATION_ALERT = 0.85

PH_STIR_START_RATIO = 0.01
PH_STIR_END_RATIO = 0.85
PH_STIR_TRANSITION_STEPS = 50_000

S3_BUCKET = "sagemaker-eu-central-1-119287771635"
FINEWEB_PREFIX = "fineweb_cache"
FINEWEB_EDU_PREFIX = "fineweb_edu_chunks_v1"

CKPT_DIR = os.path.join(os.environ.get('SM_MODEL_DIR', '/home/ec2-user/SageMaker'), 'mycelia_checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
LATEST_CKPT = os.path.join(CKPT_DIR, "mycelia_latest.pt")
BEST_CKPT = os.path.join(CKPT_DIR, "mycelia_best.pt")

LESSONS_PATH = os.path.join(CKPT_DIR, 'mycelia_lessons.jsonl')

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
    try:
        if hasattr(opt, 'adamw') and hasattr(opt.adamw, 'param_groups'): return opt.adamw.param_groups[0].get('weight_decay', 0.3)
        if hasattr(opt, 'param_groups') and len(opt.param_groups) > 1: return opt.param_groups[1].get('weight_decay', 0.3)
        if hasattr(opt, 'adamw_wd'): return opt.adamw_wd
    except Exception: pass
    return 0.3

def set_alpha_well_depth(opt, new_wd):
    try:
        if hasattr(opt, 'adamw') and hasattr(opt.adamw, 'param_groups'): opt.adamw.param_groups[0]['weight_decay'] = new_wd; return
        if hasattr(opt, 'param_groups') and len(opt.param_groups) > 1: opt.param_groups[1]['weight_decay'] = new_wd; return
        if hasattr(opt, 'adamw_wd'): opt.adamw_wd = new_wd
    except Exception: pass

# ============================================
# IMPORT ARCHITECTURE
# ============================================
try:
    from MYCELIA_architecture import MyceliaLM, MyceliaConfig
    print("🍄 Mycelia architecture loaded successfully - get ready for the ride!")
except ImportError:
    raise ImportError("MYCELIA_architecture not found!")

# ============================================
# PRESSURE TENSOR LOGGER
# ============================================
class PressureTensorLogger:
    def __init__(self):
        self.chi_history, self.alert_count, self.last_alert_step = [], 0, 0

    def update(self, info, step):
        chi = info.get('pressure_concentration', 0.0)
        self.chi_history.append(chi)
        if len(self.chi_history) > 1000: self.chi_history.pop(0)
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
        self.tokens_per_step, self.total_tokens = tokens_per_step, total_tokens
        self.start_time, self.last_time, self.last_step = time.time(), time.time(), -1
        self._cache = None
        self.window_tokens, self.window_times, self.window_size = [], [], 50
        self._first_call = True

    def update(self, step):
        if step == self.last_step: return self._cache
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
            if len(self.window_tokens) > self.window_size: self.window_tokens.pop(0); self.window_times.pop(0)
        smoothed = sum(self.window_tokens) / sum(self.window_times) if self.window_times else 0
        remaining = max(0, self.total_tokens - total_proc)
        eta = remaining / smoothed if smoothed > 0 else 0
        self.last_time, self.last_step = now, step
        raw_progress = (total_proc / self.total_tokens) * 100 if self.total_tokens > 0 else 0
        self._cache = {'step': step, 'smoothed_tps': smoothed, 'total_gb': total_proc / 1e9, 'target_gb': self.total_tokens / 1e9, 'progress': min(100.0, raw_progress), 'raw_progress': raw_progress, 'eta_h': eta / 3600, 'elapsed_h': elapsed / 3600}
        return self._cache

    def log(self, step):
        s = self.update(step)
        eta_str = str(timedelta(seconds=int(s['eta_h'] * 3600))) if s['eta_h'] > 0 else "N/A"
        elapsed_str = str(timedelta(seconds=int(s['elapsed_h'] * 3600)))
        progress_str = f"{s['progress']:.1f}%" if s['raw_progress'] <= 100 else f"{s['raw_progress']:.1f}% (>{s['target_gb']:.1f}B target)"
        print(f"\n⏱️ Step {s['step']:,} | {s['smoothed_tps']:.0f} tok/s | {s['total_gb']:.2f}/{s['target_gb']:.1f} Billion | {progress_str} | ETA {eta_str} | Elapsed {elapsed_str}")
        sys.stdout.flush()
        return s

def compute_spectral_concentration(hidden_states: torch.Tensor, top_k: int = 3) -> float:
    if hidden_states.dim() == 3:
        H = hidden_states.squeeze(0) if hidden_states.shape[0] == 1 else hidden_states.reshape(-1, hidden_states.shape[-1])
    else: H = hidden_states
    if H.shape[0] < 2 or H.shape[1] < 2: return 0.0
    H_centered = H - H.mean(dim=0, keepdim=True)
    try:
        S = torch.linalg.svdvals(H_centered)
        total = S.sum()
        if total < 1e-8: return 0.0
        return min(1.0, max(0.0, (S[:min(top_k, len(S))].sum() / total).item()))
    except Exception: return 0.0

# ============================================
# DATASETS
# ============================================
class TeachingDataset(IterableDataset):
    def __init__(self, path, tokenizer, max_seq_len=4096):
        self.path, self.tokenizer, self.target, self.lessons = path, tokenizer, max_seq_len + 1, []

    def refresh(self) -> int:
        self.lessons, seen = [], set()
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        text = (row.get('lesson') or '').strip()
                        if len(text) < 40 or text in seen: continue
                        seen.add(text)
                        ctx = row.get('context') or {}
                        header = f"[MYCELIA TEACHING | step {row.get('step', '?')} | variable: {row.get('variable', '?')} | action: {row.get('direction', '?')}"
                        if ctx: header += f" | loss {ctx.get('loss', '?')} | coh {ctx.get('coherence', '?')} | friction {ctx.get('friction', '?')} | mpc {ctx.get('mpc_intervention', '?')}"
                        self.lessons.append(header + "]\n" + text)
                    except Exception: continue
        return len(self.lessons)

    def __iter__(self):
        if self.refresh() == 0: return
        buffer, yielded = [], 0
        for _ in range(24):
            for lesson in self.lessons:
                try: toks = self.tokenizer.encode(lesson)
                except Exception: toks = self.tokenizer.encode(lesson, allowed_special="all")
                buffer.extend(toks)
                buffer.append(self.tokenizer.eos_token_id or 0)
                while len(buffer) >= self.target:
                    yield torch.tensor(buffer[:self.target], dtype=torch.long)
                    buffer = buffer[self.target:]
                    yielded += 1
            if yielded >= 4: break

class IterableWrapper(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        if hasattr(dataset, 'yield_counts'): self.yield_counts = dataset.yield_counts
    def __iter__(self):
        while True: yield self.dataset[random.randint(0, len(self.dataset) - 1)]

class CompositeIterableDataset(IterableDataset):
    def __init__(self, datasets, weights):
        self.datasets, total = datasets, sum(weights)
        self.weights = [w / total for w in weights]
        print(f"   🧱 CompositeIterableDataset initialized with {len(datasets)} streams.")
    def __iter__(self):
        iters = [iter(d) for d in self.datasets]
        while True:
            idx = random.choices(range(len(self.datasets)), weights=self.weights)[0]
            try: yield next(iters[idx])
            except StopIteration:
                iters[idx] = iter(self.datasets[idx])
                yield next(iters[idx])

class S3StreamingNpyDataset(IterableDataset):
    def __init__(self, bucket, prefix, max_seq_len=512, prefetch_size=15, source_tag="S3_UNKNOWN"):
        self.bucket, self.prefix, self.target = bucket, prefix, max_seq_len + 1
        from botocore.config import Config
        self.s3 = boto3.client('s3', region_name='eu-central-1', config=Config(connect_timeout=5, read_timeout=10, retries={'max_attempts': 2}))
        self.prefetch_size, self.source_tag = prefetch_size, source_tag 
        self.yield_counts = Counter() 
        self.queue, self.stop_event = queue.Queue(maxsize=prefetch_size), threading.Event()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True).start()
        print(f"   🌐 S3StreamingNpyDataset: s3://{bucket}/{prefix} | prefetch={prefetch_size} | tag={source_tag}")

    def _prefetch_worker(self):
        processed_keys = set()
        while not self.stop_event.is_set():
            current_keys, cont = [], None
            while True:
                kwargs = {'Bucket': self.bucket, 'Prefix': self.prefix}
                if cont: kwargs['ContinuationToken'] = cont
                try:
                    resp = self.s3.list_objects_v2(**kwargs)
                    current_keys.extend([o['Key'] for o in resp.get('Contents', []) if o['Key'].endswith('.npy')])
                    if not resp.get('IsTruncated'): break
                    cont = resp.get('NextContinuationToken')
                except Exception as e:
                    print(f"⚠️ S3 List error: {e}"); time.sleep(5); break
            new_keys = [k for k in current_keys if k not in processed_keys]
            if not new_keys: time.sleep(15); continue
            random.shuffle(new_keys)
            for key in new_keys:
                if self.stop_event.is_set(): break
                try:
                    arr = np.load(io.BytesIO(self.s3.get_object(Bucket=self.bucket, Key=key)['Body'].read()))
                    self.queue.put(arr, timeout=10.0)
                    processed_keys.add(key)
                except queue.Full: continue
                except Exception as e: print(f"⚠️ S3 prefetch error for {key}: {e}"); continue
        self.queue.put(None)

    def __iter__(self):
        buffer = np.array([], dtype=np.int32)
        while True:
            try: arr = self.queue.get(timeout=10.0)
            except queue.Empty:
                print("⚠️ Queue timeout. Restarting prefetch thread...")
                self.stop_event.set(); self.prefetch_thread.join(timeout=5.0); self.stop_event.clear()
                self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True).start()
                continue
            if arr is None:
                self.stop_event.set(); self.prefetch_thread.join(timeout=5.0); self.stop_event.clear()
                self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True).start()
                continue
            buffer = np.concatenate((buffer, arr)) if len(buffer) > 0 else arr
            while len(buffer) >= self.target:
                self.yield_counts[f"🌐 {self.source_tag}"] += 1
                yield torch.tensor(buffer[:self.target], dtype=torch.long)
                buffer = buffer[self.target:]

    def __del__(self): self.stop_event.set()

class PHStirredMixtureDataset(IterableDataset):
    def __init__(self, anchor_dataset, expansion_dataset, start_ratio=0.01, end_ratio=0.85, transition_steps=50000):
        self.anchor, self.expansion = anchor_dataset, expansion_dataset
        self.start_ratio, self.end_ratio, self.transition_steps = start_ratio, end_ratio, transition_steps
        self.current_step, self._last_logged_ratio = 0, -1.0
        print(f"   🧪 PHStirredMixtureDataset: ratio {start_ratio:.2f} → {end_ratio:.2f} over {transition_steps:,} steps")

    def update_step(self, step): self.current_step = step
    def get_current_ratio(self):
        if self.current_step >= self.transition_steps: return self.end_ratio
        return self.start_ratio + (self.end_ratio - self.start_ratio) * (self.current_step / self.transition_steps)

    def __iter__(self):
        anchor_iter, expansion_iter = iter(self.anchor), iter(self.expansion)
        while True:
            if random.random() < self.get_current_ratio():
                try: yield next(expansion_iter)
                except StopIteration:
                    expansion_iter = iter(self.expansion)
                    try: yield next(expansion_iter)
                    except StopIteration: yield next(anchor_iter)
            else:
                try: yield next(anchor_iter)
                except StopIteration:
                    anchor_iter = iter(self.anchor)
                    yield next(anchor_iter)

def collate(batch): return torch.stack(batch)

# ============================================
# CHECKPOINT HELPERS
# ============================================
def _add_checksum(data):
    try: data['_checkpoint_checksum'] = hashlib.sha256(str(sorted(data.keys())).encode()).hexdigest()[:16]
    except Exception: pass
    return data

def _safe_load_checkpoint(path):
    if not os.path.exists(path): return None
    file_size = os.path.getsize(path)
    print(f"   📂 File: {path}\n   💾 Size: {file_size/1e9:.2f} GB")
    if file_size < 1e6: print(f"   ⚠️  File too small ({file_size} bytes), skipping"); return None
    use_safe = file_size < 5e9
    try:
        t0 = time.time()
        print(f"   ⏳ Loading to CPU... (est. {max(30, int(file_size/3e8))}s)"); sys.stdout.flush()
        if use_safe:
            try: ckpt = torch.load(path, map_location='cpu', weights_only=True); print(f"   ✅ Safe CPU load")
            except Exception: print(f"   ⚠️  Safe load failed, using legacy..."); ckpt = torch.load(path, map_location='cpu', weights_only=False); print(f"   ✅ Legacy CPU load")
        else: ckpt = torch.load(path, map_location='cpu', weights_only=False); print(f"   ✅ Fast CPU load (weights_only=False)")
        print(f"   ⏱️  Load time: {time.time() - t0:.1f}s")
        return ckpt
    except Exception as e: print(f"   🚨 Failed to load: {type(e).__name__}: {str(e)[:200]}"); return None

def cleanup_checkpoints(ckpt_dir, keep=2):
    import glob
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "mycelia_step_*.pt")), key=os.path.getmtime)
    for old in ckpts[:-keep]:
        try: os.remove(old)
        except: pass

# ============================================
# MAIN
# ============================================
print("\n" + "="*70 + "\n🍄 MYCELIA TRAINING v12.9 (1.5B Muon + SNLI + MASSIF Observatory)\n" + "="*70)

print("\n📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id or 0
print(f"   Vocab: {tokenizer.vocab_size:,}")

print("\n🏗️ Building model...")
cfg = MyceliaConfig()
cfg.use_gradient_checkpointing = True
cfg.max_seq_len = MAX_SEQ_LEN
cfg.vocab_size = 151643
cfg.compress_window, cfg.compress_ratio, cfg.use_compression = 128, 8, False
cfg.consensus_rounds = CONSENSUS_ROUNDS
cfg.ffn_norm_target, cfg.alpha_norm_target, cfg.soft_cap = FFN_TARGET_START, ALPHA_TARGET_START, 400.0
cfg.instability_target, cfg.control_gain, cfg.control_factor_floor, cfg.predictive_scale = 0.80, CONTROL_GAIN_DEFAULT, 0.7, True
cfg.use_rate_governor, cfg.ffn_growth_ratio_max, cfg.residual_growth_ratio_max = False, 2.0, 1.5
cfg.use_gradual_transition, cfg.transition_duration = True, TRANSITION_DURATION
cfg.ffn_target_end, cfg.alpha_target_end = FFN_TARGET_END, ALPHA_TARGET_END
cfg.max_simultaneous_governors = MAX_SIMULTANEOUS_GOVERNORS

model = MyceliaLM(cfg)
if torch.cuda.is_available(): model = model.to(device='cuda', dtype=torch.bfloat16)
else: model = model.to('cpu')
device = next(model.parameters()).device

print(f"   {sum(p.numel() for p in model.parameters()):,} params on {device}")
auto_tuner = GovernorAutoTuner(model, interval=AUTO_TUNE_EVERY)
pressure_logger = PressureTensorLogger()

if not cfg.use_compression:
    if hasattr(model, 'compressor') and model.compressor is not None:
        print("🧹 Deleting MycelialCompressor to free 200MB VRAM (use_compression=False)...")
        del model.compressor
        model.compressor = None
        torch.cuda.empty_cache()

opt = make_mycelia_optimizer(model, muon_lr=PEAK_LR, adamw_lr=PEAK_LR, muon_wd=WEIGHT_DECAY, adamw_wd=0.01)
print(f"\n🔍 Optimizer: Muon + 8-bit AdamW hybrid | alpha_well_depth={get_alpha_well_depth(opt):.3f}\n" + "="*70 + "\n")

total_steps = TOTAL_TOKENS_TARGET // (BATCH_SIZE * ACCUM_STEPS * MAX_SEQ_LEN)
scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
if hasattr(opt, 'sync_adamw_lr'): opt.sync_adamw_lr(scheduler.get_last_lr()[0])
print(f"\n🔥 Scheduler: HF cosine | peak={PEAK_LR:.2e} | min={MIN_LR:.2e} | warmup={WARMUP_STEPS} | total={total_steps:,}")

# ============================================
# RESUME
# ============================================
start_epoch, best_loss, best_step, step, ckpt = 0, float('inf'), 0, 0, None

for path, label in [(BEST_CKPT, "🏆 BEST"), (LATEST_CKPT, "📂 LATEST")]:
    if os.path.exists(path):
        print(f"\n{'='*70}\n{label} CHECKPOINT\n{'='*70}")
        ckpt = _safe_load_checkpoint(path)
        if ckpt is not None: break

if ckpt is not None:
    print("   🔄 Filtering checkpoint state_dict for shape mismatches...")
    ckpt_state = ckpt['model_state_dict']
    model_state = model.state_dict()
    safe_state, skipped, legacy_buffers = {}, 0, 0
    for k, v in ckpt_state.items():
        if k in model_state and model_state[k].shape != v.shape:
            if '_prev_hidden' in k or '_prev_delta' in k: legacy_buffers += 1; continue
            print(f"      ⚠️  Skipping {k}"); skipped += 1
        else: safe_state[k] = v
    model.load_state_dict(safe_state, strict=False)
    if legacy_buffers > 0: print(f"   🧹 Silently reset {legacy_buffers} legacy geometric buffers")
    if skipped > 0: print(f"   ✅ Skipped {skipped} mismatched geometric buffers")

    if 'optimizer_state_dict' in ckpt:
        try: opt.load_state_dict(ckpt['optimizer_state_dict']); print("   ✅ Optimizer state restored")
        except Exception as e: print(f"   ⚠️  Optimizer state load failed: {e}")

    model = model.to(device)
    print("   ✅ Model loaded")
    step = ckpt.get('global_step', 0)
    start_epoch = ckpt.get('epoch', 0) + 1
    prev_loss = ckpt.get('loss', 'N/A')
    best_loss_ckpt = ckpt.get('best_loss', float('inf'))
    if isinstance(prev_loss, (int, float)) and prev_loss > 0: print(f"   📊 Resumed: step={step:,} | loss={prev_loss:.4f}")
    if isinstance(best_loss_ckpt, (int, float)) and best_loss_ckpt > 0:
        best_loss = best_loss_ckpt; print(f"   🏆 Best loss: {best_loss:.4f}")

    scheduler_alive = False
    if 'scheduler_state_dict' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            current_lr = scheduler.get_last_lr()[0]
            if current_lr > 0: print(f"   ✅ Scheduler restored | LR={current_lr:.2e}"); scheduler_alive = True
        except Exception as e: print(f"   ⚠️  Scheduler restore failed: {e}")

    if not scheduler_alive:
        new_total = max(step + 2_000_000, total_steps)
        scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=new_total)
        for _ in range(step): scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"   🔥 Scheduler rebuilt: total={new_total:,} | LR={current_lr:.2e}")

    for g in opt.param_groups:
        g['lr'] = scheduler.get_last_lr()[0]
        if hasattr(opt, 'sync_adamw_lr'): opt.sync_adamw_lr(scheduler.get_last_lr()[0])

    if 'auto_tuner_state' in ckpt:
        try: auto_tuner.load_state(ckpt['auto_tuner_state']); print("   ✅ Auto-tuner restored")
        except Exception as e: print(f"   ⚠️  Auto-tuner restore failed: {e}")

    if hasattr(auto_tuner, '_meta_governor'):
        del auto_tuner._meta_governor; print("   🧹 Old meta-governor state purged")

    if 'alpha_well_history' in ckpt:
        auto_tuner._well_history = ckpt['alpha_well_history']
        auto_tuner._best_loss_since_well = ckpt.get('alpha_best_loss_since_well', float('inf'))

    del ckpt_state, safe_state, ckpt
    gc.collect()

    with torch.no_grad():
        sample_ffn_norms, sample_contrib_norms = [], []
        for block in model.blocks:
            info = getattr(block, '_last_info', {}) or {}
            if 'mean_ffn_norm' in info: sample_ffn_norms.append(float(info['mean_ffn_norm']))
            if 'mean_contrib_norm' in info: sample_contrib_norms.append(float(info['mean_contrib_norm']))
        new_ffn_target = max(20.0, (sum(sample_ffn_norms) / len(sample_ffn_norms)) * 1.15) if sample_ffn_norms else 50.0
        new_alpha_target = max(20.0, (sum(sample_contrib_norms) / len(sample_contrib_norms)) * 1.15) if sample_contrib_norms else 45.0
        for block in model.blocks:
            block.ffn_norm_target = new_ffn_target
            block.alpha_norm_target = new_alpha_target
            block.soft_cap_target = max(25.0, new_ffn_target * 1.15)
        cfg.governor_recalibrated = True
        cfg.recal_ffn_target, cfg.recal_alpha_target, cfg.recal_softcap_target = new_ffn_target, new_alpha_target, max(25.0, new_ffn_target * 1.0)
        print(f"   🎯 Governor recalibration: FFN target → {new_ffn_target:.0f}, Alpha target → {new_alpha_target:.0f}")
else:
    scheduler._resurrected_count = 0
    auto_tuner.scheduler_resurrected_count = 0
    for block in model.blocks: block.config.transition_start_step = 0
    print(f"\n{'='*70}\n🚀 FRESH START — 1.5B de novo\n{'='*70}")
    RECAL_FFN, RECAL_ALPHA, RECAL_CAP = 90.0, 55.0, 120.0
    for block in model.blocks:
        block.ffn_norm_target = RECAL_FFN; block.alpha_norm_target = RECAL_ALPHA; block.soft_cap_target = RECAL_CAP
    cfg.governor_recalibrated = True
    cfg.recal_ffn_target, cfg.recal_alpha_target, cfg.recal_softcap_target = RECAL_FFN, RECAL_ALPHA, RECAL_CAP
    print(f"   🎯 Governor recalibration: FFN → {RECAL_FFN:.0f}, Alpha → {RECAL_ALPHA:.0f}, SoftCap → {RECAL_CAP:.0f}")

# ============================================
# COLD-START ARTIFACT SUPPRESSOR INIT
# ============================================
RESUME_STEP = step
ARTIFACT_SUPPRESSION_WINDOW = 5

if not hasattr(auto_tuner, '_friction_history'): auto_tuner._friction_history = []

# ============================================
# DATA LOADING
# ============================================   
print("\n📖 Loading Datasets (pH-Stirred Mixture Protocol)...")
anchor_local = UniversalNpyDataset("mycelia_s3_chunks", max_seq_len=MAX_SEQ_LEN, teaching_path=LESSONS_PATH, teaching_weight=0.06, tokenizer=tokenizer)
anchor_corpus = IterableWrapper(anchor_local)

exp_fineweb_edu = S3StreamingNpyDataset(bucket=S3_BUCKET, prefix=FINEWEB_EDU_PREFIX, max_seq_len=MAX_SEQ_LEN, prefetch_size=15, source_tag="FW_EDU")
exp_fineweb_cache = S3StreamingNpyDataset(bucket=S3_BUCKET, prefix=FINEWEB_PREFIX, max_seq_len=MAX_SEQ_LEN, prefetch_size=15, source_tag="FW_ORIG")
expansion_corpus = CompositeIterableDataset(datasets=[exp_fineweb_edu, exp_fineweb_cache], weights=[0.90, 0.10])

mixed = PHStirredMixtureDataset(anchor_dataset=anchor_corpus, expansion_dataset=expansion_corpus, start_ratio=PH_STIR_START_RATIO, end_ratio=PH_STIR_END_RATIO, transition_steps=PH_STIR_TRANSITION_STEPS)

loader = DataLoader(mixed, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
data_iter = iter(loader)
print("   ✅ pH-Stirred Mixture active.")

tokens_per_step = BATCH_SIZE * ACCUM_STEPS * MAX_SEQ_LEN
actual_total_tokens = max(total_steps * tokens_per_step, step * tokens_per_step)
tracker = ThroughputTracker(tokens_per_step, actual_total_tokens)
print(f"\n⏱️  Tracker: {tracker.tokens_per_step:,} tok/step | {tracker.total_tokens/1e9:.1f}B total")

# ============================================
# PHASE 2: INDEPENDENT SNLI EVALUATION SETUP
# ============================================
print("🔄 Loading canonical SNLI dataset for independent evaluation...")
snli_raw = load_dataset("stanfordnlp/snli", split="validation")
snli_filtered = snli_raw.filter(lambda example: example["label"] != -1)
print(f"✅ Loaded {len(snli_filtered)} clean SNLI examples.")

label_names = ["entailment", "contradiction", "neutral"]

def tokenize_snli(example):
    prompt = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nRelation:"
    prompt_enc = tokenizer(prompt, truncation=True, max_length=512)
    return {"prompt": prompt, "prompt_len": len(prompt_enc["input_ids"]), "label": example["label"], "label_text": label_names[example["label"]]}

snli_tokenized = snli_filtered.map(tokenize_snli, batched=False)
snli_tokenized.set_format(type="torch", columns=["prompt", "prompt_len", "label", "label_text"])
snli_dataloader = DataLoader(snli_tokenized, batch_size=16, shuffle=False)
print("✅ SNLI DataLoader ready.")

EVAL_INTERVAL = 5000
# Automatically version the CSV file with a timestamp to prevent overwriting
_run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EVAL_CSV_PATH = f"/home/ec2-user/SageMaker/mycelia_eval_logs/snli_audit_{_run_timestamp}.csv"
LEGACY_EVAL_CSV_PATH = "/home/ec2-user/SageMaker/mycelia_eval_logs/snli_independent_audit.csv"
EVAL_CSV_HEADER = ["step", "snli_accuracy", "snli_entropy", "total_samples", "internal_delta", "internal_pawula_ratio", "internal_rho_dir", "massif_instability_score", "massif_failure_flags"]
os.makedirs(os.path.dirname(EVAL_CSV_PATH), exist_ok=True)

def _backup_csv_without_overwriting(path):
    backup_path = f"{path}.bak"
    if os.path.exists(backup_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{path}.{timestamp}.bak"
        suffix = 1
        while os.path.exists(backup_path): backup_path = f"{path}.{timestamp}.{suffix}.bak"; suffix += 1
    os.replace(path, backup_path)
    return backup_path

def _read_csv_header(path):
    with open(path, "r", newline="") as f: return next(csv.reader(f), None)

if os.path.exists(LEGACY_EVAL_CSV_PATH):
    if _read_csv_header(LEGACY_EVAL_CSV_PATH) != EVAL_CSV_HEADER:
        print(f"⚠️ Archived incompatible evaluation CSV: {_backup_csv_without_overwriting(LEGACY_EVAL_CSV_PATH)}")

if os.path.exists(EVAL_CSV_PATH):
    if _read_csv_header(EVAL_CSV_PATH) != EVAL_CSV_HEADER:
        print(f"⚠️ Archived incompatible evaluation CSV: {_backup_csv_without_overwriting(EVAL_CSV_PATH)}")

if not os.path.exists(EVAL_CSV_PATH):
    with open(EVAL_CSV_PATH, 'w', newline='') as f: csv.writer(f).writerow(EVAL_CSV_HEADER)

def evaluate_snli_independent(model, tokenizer, snli_dataloader, device, step):
    """
    Evaluates SNLI accuracy, entropy, and 5-mode MASSIF Inference Instability.
    Batches the 3 labels in a single forward pass for maximum GPU efficiency.
    """
    model.eval()
    correct_predictions, total_samples, total_entropy, total_instability = 0, 0, 0.0, 0.0
    pad_id = tokenizer.pad_token_id
    label_names = ["entailment", "contradiction", "neutral"]
    all_triggered_flags = set()
    
    with torch.no_grad():
        for batch in snli_dataloader:
            prompts = batch["prompt"]
            prompt_lens = batch["prompt_len"].tolist()
            true_labels = batch["label"].tolist()
            true_label_texts = batch["label_text"]
            
            for i in range(len(prompts)):
                prompt = prompts[i]
                prompt_len = prompt_lens[i]
                true_label = true_labels[i]
                true_label_text = true_label_texts[i]
                true_idx = label_names.index(true_label_text)
                
                # --- 1. BATCH THE 3 LABELS (Single Forward Pass) ---
                texts = [f"{prompt} {lt}" for lt in label_names]
                encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                input_ids = encodings["input_ids"]
                attention_mask = encodings["attention_mask"]
                padding_mask = (input_ids == pad_id)
                
                hidden_states = {}
                def hook_fn(module, input, output):
                    hidden_states['last'] = output[0] if isinstance(output, tuple) else output
                handle = model.blocks[-1].register_forward_hook(hook_fn)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits_out = model(input_ids, padding_mask=padding_mask, use_compression=False, log_during_train=False)
                handle.remove()
                
                logits = torch.cat(logits_out, dim=1) if isinstance(logits_out, list) else logits_out
                
                # --- 2. COMPUTE NLL & MEAN TRAJECTORY ENTROPY ---
                nlls = []
                entropies = []
                for j in range(3):
                    seq_len = attention_mask[j].sum().item()
                    if seq_len > prompt_len:
                        shift_logits = logits[j, prompt_len-1:seq_len-1, :].contiguous().unsqueeze(0)
                        shift_labels = input_ids[j, prompt_len:seq_len].contiguous().unsqueeze(0)
                        loss_fct = torch.nn.CrossEntropyLoss()
                        nlls.append(loss_fct(shift_logits.float().view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item())
                        
                        # Mean trajectory entropy for this label's generated tokens
                        probs = F.softmax(shift_logits.float(), dim=-1)
                        token_entropies = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                        entropies.append(token_entropies.mean().item())
                    else:
                        nlls.append(float('inf'))
                        entropies.append(float('inf'))
                
                best_label = int(np.argmin(nlls))
                if best_label == true_label: correct_predictions += 1
                total_samples += 1
                
                if len(entropies) == 3 and entropies[true_idx] != float('inf'):
                    total_entropy += entropies[true_idx]
                
                # --- 3. MASSIF LATENT STABILITY OBSERVATORY ---
                h = hidden_states.get('last')
                if h is not None:
                    gen_lens = [attention_mask[j].sum().item() - prompt_len for j in range(3)]
                    I_ts = []
                    
                    for j in range(3):
                        gl = gen_lens[j]
                        if gl > 1:
                            h_j = h[j, prompt_len:prompt_len+gl, :].float()
                            h_norm = F.normalize(h_j, p=2, dim=-1)
                            v = h_norm[1:, :] - h_norm[:-1, :]
                            if v.shape[0] > 1:
                                I_t = F.cosine_similarity(v[:-1, :], v[1:, :], dim=-1)
                                I_ts.append(I_t)
                            
                    if I_ts:
                        max_i_len = max([len(it) for it in I_ts])
                        I_padded = torch.full((len(I_ts), max_i_len), float('nan'), device=device, dtype=torch.float32)
                        for k, it in enumerate(I_ts):
                            I_padded[k, :len(it)] = it
                        
                        mean_I = torch.nanmean(I_padded).item()
                        kappa = torch.acos(torch.clamp(torch.tensor(mean_I), -1.0+1e-7, 1.0-1e-7)).item()
                        sigma_t = torch.nanstd(I_padded, dim=0)
                        max_sigma = torch.nanmax(sigma_t).item()
                        
                        if len(sigma_t) > 1:
                            phi_t = torch.diff(sigma_t)
                            max_phi = torch.nanmax(torch.abs(phi_t)).item()
                        else:
                            max_phi = 0.0
                        psi_t = torch.abs(phi_t[-3:]).mean().item() if len(sigma_t) >= 4 else max_phi

                        failures = []
                        if mean_I > 0.0: failures.append("attractor_lockin")
                        if max_sigma > 0.35: failures.append("brittle_oscillation")
                        if kappa < 0.2: failures.append("curvature_collapse")
                        
                        # Loop Degeneration Check (Entropy Collapse + High Tension)
                        if max_sigma > 0.25 and len(entropies) == 3:
                            true_traj_entropy = entropies[true_idx] if entropies[true_idx] != float('inf') else 1.0
                            if true_traj_entropy < 0.5:
                                failures.append("loop_degeneration")
                                
                        if len(sigma_t) >= 4:
                            early_sigma = torch.nanmean(sigma_t[:2]).item()
                            late_sigma = torch.nanmean(sigma_t[-2:]).item()
                            if early_sigma > 0.01 and late_sigma > (early_sigma * 1.5):
                                failures.append("stress_accumulation")

                        total_instability += len(failures) / 5.0
                        if failures:
                            all_triggered_flags.update(failures)
                        
                        if 'I_padded' in locals(): del I_padded
                        if 'sigma_t' in locals(): del sigma_t
                        if 'phi_t' in locals(): del phi_t

                # 🧹 SAFE CLEANUP
                if 'logits_out' in locals(): del logits_out
                if 'logits' in locals(): del logits
                if 'h' in locals(): del h
                if 'hidden_states' in locals(): del hidden_states
                if 'I_ts' in locals(): del I_ts
                if 'input_ids' in locals(): del input_ids
                if 'attention_mask' in locals(): del attention_mask
                if 'padding_mask' in locals(): del padding_mask
                if 'probs' in locals(): del probs
                if 'shift_logits' in locals(): del shift_logits
                if 'shift_labels' in locals(): del shift_labels
                torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    model.train()
    
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    avg_entropy = total_entropy / total_samples if total_samples > 0 else 0.0
    avg_instability = total_instability / total_samples if total_samples > 0 else 0.0
    summary_flags = ",".join(sorted(list(all_triggered_flags))) if all_triggered_flags else "none"
    
    return {
        "step": step, "snli_accuracy": accuracy, "snli_entropy": avg_entropy,
        "total_samples": total_samples, "massif_instability_score": avg_instability,
        "massif_failure_flags": summary_flags
    }
# ============================================
# TRAINING LOOP
# ============================================
print("\n" + "="*70 + f"\n🚀 EPOCH {start_epoch} — STEP {step:,}\n" + "="*70 + "\n")

if hasattr(model, 'consensus_stats'): model.consensus_stats.clear()
if hasattr(model, 'dubito_history'): model.dubito_history.clear()
torch.cuda.empty_cache()

_sde_hidden_states, _fiber_q = {}, {}

def _sde_hook(layer_idx):
    def hook(module, input, output): _sde_hidden_states[layer_idx] = (output[0] if isinstance(output, tuple) else output).detach()[:, :64, :]
    return hook

def _fiber_q_hook(layer_idx):
    def hook(module, input, output): _fiber_q[layer_idx] = output.detach()[:, :64, :]
    return hook

_sde_lattice = [4, 8, 12, 16, 20, 23]
for _idx in _sde_lattice:
    if _idx < len(model.blocks):
        model.blocks[_idx].register_forward_hook(_sde_hook(_idx))
        if hasattr(model.blocks[_idx].attn, 'qkv'): model.blocks[_idx].attn.qkv.register_forward_hook(_fiber_q_hook(_idx))
print(f"🌊 SDE & 🧶 Fiber Observatories: hooks registered on layers {_sde_lattice}")

def compute_sde_telemetry():
    layers = sorted(_sde_hidden_states.keys())
    if len(layers) < 2: return {"sde_drift_norm": 0.0, "sde_noise_trace": 0.0, "sde_snr": 0.0, "sde_drift_dir": 0.0, "sde_noise_dir": 0.0, "sde_snr_dir": 0.0, "directional_coherence_layers": [], "directional_coherence_global": 0.0, "pawula_ratio": 0.0, "pawula_safe": False, "var_dir": 0.0, "d4_dir": 0.0}
    drift_raw, noise_raw, drift_dir, noise_dir, coherence_per_layer, d4_dir = [], [], [], [], [], []
    for idx in range(len(layers) - 1):
        l_curr, l_next = layers[idx], layers[idx + 1]
        h_curr, h_next = _sde_hidden_states[l_curr], _sde_hidden_states[l_next]
        delta_raw = h_next - h_curr
        D1_raw = delta_raw.mean(dim=[0, 1])
        drift_raw.append(torch.norm(D1_raw).item())
        c_raw = delta_raw - D1_raw.unsqueeze(0).unsqueeze(0)
        noise_raw.append(0.5 * (c_raw ** 2).mean().item())
        u_curr = h_curr / (torch.norm(h_curr, dim=-1, keepdim=True) + 1e-8)
        u_next = h_next / (torch.norm(h_next, dim=-1, keepdim=True) + 1e-8)
        v_dir = u_next - u_curr
        v_hat = v_dir / (torch.norm(v_dir, dim=-1, keepdim=True) + 1e-8)
        coherence_per_layer.append(torch.norm(v_hat.mean(dim=[0, 1])).item())
        D1_dir = v_dir.mean(dim=[0, 1])
        drift_dir.append(torch.norm(D1_dir).item())
        c_dir = v_dir - D1_dir.unsqueeze(0).unsqueeze(0)
        c_dir_f64 = c_dir.to(torch.float64)
        d2_term, d4_term = (c_dir_f64 ** 2).mean().item(), (c_dir_f64 ** 4).mean().item()
        noise_dir.append(0.5 * d2_term); d4_dir.append(d4_term)
    avg_drift_raw = sum(drift_raw) / len(drift_raw)
    avg_noise_raw = sum(noise_raw) / len(noise_raw)
    rho_raw = avg_drift_raw / (avg_noise_raw ** 0.5 + 1e-8)
    avg_drift_dir = sum(drift_dir) / len(drift_dir)
    avg_noise_dir = sum(noise_dir) / len(noise_dir)
    avg_d4_dir = sum(d4_dir) / len(d4_dir)
    rho_dir = avg_drift_dir / (avg_noise_dir ** 0.5 + 1e-8)
    avg_var_dir = 2.0 * avg_noise_dir
    pawula_ratio = avg_d4_dir / (avg_var_dir ** 2 + 1e-12)
    return {"sde_drift_norm": avg_drift_raw, "sde_noise_trace": avg_noise_raw, "sde_snr": rho_raw, "sde_drift_dir": avg_drift_dir, "sde_noise_dir": avg_noise_dir, "sde_snr_dir": rho_dir, "directional_coherence_layers": coherence_per_layer, "directional_coherence_global": sum(coherence_per_layer) / len(coherence_per_layer) if coherence_per_layer else 0.0, "pawula_ratio": pawula_ratio, "pawula_safe": pawula_ratio < 50.0, "var_dir": avg_var_dir, "d4_dir": avg_d4_dir}

def compute_attention_entropy_variance():
    layers = sorted(_fiber_q.keys())
    if len(layers) == 0: return 0.0
    H, D_head, layer_vars = 32, 2048 // 32, []
    for l in layers:
        qkv_raw = _fiber_q[l]
        B, T, _ = qkv_raw.shape
        q, k, _ = qkv_raw.chunk(3, dim=-1)
        q_heads = q.view(B, T, H, D_head).transpose(1, 2)
        k_heads = k.view(B, T, H, D_head).transpose(1, 2)
        T_slice = min(T, 64)
        logits = torch.matmul(q_heads[:, :, :T_slice, :], k_heads[:, :, :T_slice, :].transpose(-2, -1)) / (D_head ** 0.5)
        probs = F.softmax(logits, dim=-1).clamp(min=1e-9)
        layer_vars.append(torch.var(-torch.sum(probs * torch.log(probs), dim=-1).mean(dim=[0, 2])).item())
    return sum(layer_vars) / len(layer_vars) if layer_vars else 0.0

gci_ema = {'sigma_head': 0.70, 'attn_entropy_var': 0.001, 'R_pawula': 20.0}
GCI_ALPHA = 0.01 

def update_gci_ema(sigma_head, attn_entropy_var, R_pawula):
    gci_ema['sigma_head'] = (1 - GCI_ALPHA) * gci_ema['sigma_head'] + GCI_ALPHA * sigma_head
    gci_ema['attn_entropy_var'] = (1 - GCI_ALPHA) * gci_ema['attn_entropy_var'] + GCI_ALPHA * attn_entropy_var
    gci_ema['R_pawula'] = (1 - GCI_ALPHA) * gci_ema['R_pawula'] + GCI_ALPHA * R_pawula

def compute_gci_snapshot(sigma_head, sigma_dot_head, attn_entropy_var, W_gov, R_pawula, R_t):
    update_gci_ema(sigma_head, attn_entropy_var, R_pawula)
    score = 0.0
    if abs(sigma_head - gci_ema['sigma_head']) < 0.01: score += 1/6
    if abs(sigma_dot_head) < 0.005: score += 1/6
    if abs(attn_entropy_var - gci_ema['attn_entropy_var']) < 0.0005: score += 1/6
    if W_gov < 1.0: score += 1/6
    if abs(R_pawula - gci_ema['R_pawula']) < 1.0: score += 1/6
    if R_t > 1.5: score += 1/6
    return score

def compute_fiber_curvature():
    layers = sorted(_fiber_q.keys())
    if len(layers) == 0: return 0.0
    variances, H, D_head = [], 32, 2048 // 32
    for l in layers:
        qkv_raw = _fiber_q[l]
        B, T, _ = qkv_raw.shape
        q, k, v = qkv_raw.chunk(3, dim=-1)
        q_heads = q.view(B, T, H, D_head).transpose(1, 2)
        k_heads = k.view(B, T, H, D_head).transpose(1, 2)
        T_slice = min(T, 64)
        logits = torch.matmul(q_heads[:, :, :T_slice, :], k_heads[:, :, :T_slice, :].transpose(-2, -1)) / (D_head ** 0.5)
        variances.append(torch.var(logits, dim=1).mean().item())
    return sum(variances) / len(variances) if variances else 0.0

def compute_attractor_basin_radius():
    layers = sorted(_sde_hidden_states.keys())
    if not layers: return 0.0
    h = _sde_hidden_states[layers[-1]]
    return torch.var(h - h.mean(dim=1, keepdim=True), dim=1).sum().item()

def compute_lyapunov_proxy():
    layers = sorted(_sde_hidden_states.keys())
    if len(layers) < 2: return 0.0
    expansion_rates = []
    for idx in range(len(layers) - 1):
        h_curr, h_next = _sde_hidden_states[layers[idx]], _sde_hidden_states[layers[idx + 1]]
        ratio = (torch.norm(h_next.mean(dim=1), dim=-1) + 1e-8) / (torch.norm(h_curr.mean(dim=1), dim=-1) + 1e-8)
        expansion_rates.append(torch.log(ratio).mean().item())
    return sum(expansion_rates) / len(expansion_rates)

def compute_critical_slowing_down(history):
    if len(history) < 10: return 0.0
    arr = np.array(history[-50:]) 
    if np.std(arr) < 1e-6: return 1.0 
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])

model.train()
losses_window, nan_count, accum_counter, mean_alpha_grad, max_alpha_grad = [], 0, 0, 0.0, 0.0

for b in model.blocks: b.mycelia.reset_stats()

PROFILE_STEP = 17_873
prof = None

for step in tqdm(range(step, total_steps), desc="Training", initial=step):
    mixed.update_step(step)

    if step == PROFILE_STEP and prof is None:
        print(f"\n🔬 Starting profiler for step {step}...")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True).start()

    if _shutdown_requested:
        print("\n🛑 Graceful shutdown, saving checkpoint...")
        try:
            _lineage_receipt = compute_lineage_receipt(model, step=step, epoch=start_epoch, cfg=cfg, batch_size=BATCH_SIZE, accum_steps=ACCUM_STEPS, max_seq_len=MAX_SEQ_LEN, peak_lr=PEAK_LR, min_lr=MIN_LR, warmup_steps=WARMUP_STEPS, grad_clip=GRAD_CLIP, weight_decay=WEIGHT_DECAY, ffn_target_start=FFN_TARGET_START, ffn_target_end=FFN_TARGET_END, alpha_target_start=ALPHA_TARGET_START, alpha_target_end=ALPHA_TARGET_END, transition_duration=TRANSITION_DURATION)
            emergency_ckpt = _add_checksum({'epoch': start_epoch, 'global_step': step, 'model_state_dict': model.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'auto_tuner_state': auto_tuner.get_state(), 'loss': float(losses_window[-1]) if losses_window else None, 'best_loss': float(best_loss), 'best_step': best_step, 'timestamp': datetime.now().isoformat(), 'lineage_receipt': _lineage_receipt})
            torch.save(emergency_ckpt, LATEST_CKPT)
            print(f"\n💾 Emergency save: step {step:,} → {LATEST_CKPT}")
        except Exception as e: print(f"\n🚨 Emergency save failed: {e}")
        break

    try: batch = next(data_iter)
    except StopIteration: data_iter = iter(loader); batch = next(data_iter)

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
        if progress < 1.0:
            block.ffn_norm_target = current_ffn_target
            block.alpha_norm_target = current_alpha_target
            if getattr(cfg, 'governor_recalibrated', False):
                block.ffn_norm_target = cfg.recal_ffn_target
                block.alpha_norm_target = cfg.recal_alpha_target
                block.soft_cap_target = cfg.recal_softcap_target
        block.use_rate_governor = use_rate

    with autocast(dtype=torch.bfloat16):
        logits_out = model(input_ids, padding_mask=(input_ids == PAD_ID), use_compression=False, log_during_train=False)
        if isinstance(logits_out, list):
            ce_loss = 0.0
            for i, chunk_logits in enumerate(logits_out):
                chunk_targets = targets[:, i*128:(i+1)*128]
                ce_loss += F.cross_entropy(chunk_logits.reshape(-1, chunk_logits.size(-1)), chunk_targets.reshape(-1), ignore_index=PAD_ID)
            ce_loss = (ce_loss / len(logits_out)) / ACCUM_STEPS
        else:
            ce_loss = F.cross_entropy(logits_out.reshape(-1, logits_out.size(-1)), targets.reshape(-1), ignore_index=PAD_ID) / ACCUM_STEPS
        alpha_loss = model.alpha_regularization_loss() / ACCUM_STEPS    
        loss = ce_loss + alpha_loss

    if torch.isnan(loss) or torch.isinf(loss):
        nan_count += 1
        print(f"\n⚠️  NaN/Inf at step {step} (count: {nan_count})")
        if nan_count >= 2:
            if hasattr(opt, 'halve_all_lr'): opt.halve_all_lr()
            else: 
                for g in opt.param_groups: g['lr'] *= 0.5
            print(f"   🚨 LR halved to {opt.param_groups[0]['lr']:.2e}")
        if nan_count >= 3:
            print("   🚨🚨 Persistent NaN, rebuilding Muon hybrid optimizer")
            current_lr = opt.param_groups[0]['lr'] if hasattr(opt, 'param_groups') else PEAK_LR
            opt = make_mycelia_optimizer(model, muon_lr=current_lr * 2, adamw_lr=current_lr * 2, muon_wd=WEIGHT_DECAY, adamw_wd=get_alpha_well_depth(opt))
            nan_count = 0; opt.zero_grad(); continue

    nan_count = 0
    torch.cuda.empty_cache()
    loss.backward()
    accum_counter += 1
    loss_val = loss.item() * ACCUM_STEPS
    del loss, logits_out, ce_loss, alpha_loss
    torch.cuda.empty_cache()

    if accum_counter >= ACCUM_STEPS:
        alpha_grad_norms = model.alpha_gradient_norms()
        mean_alpha_grad = sum(alpha_grad_norms) / len(alpha_grad_norms) if alpha_grad_norms else 0.0
        max_alpha_grad = max(alpha_grad_norms) if alpha_grad_norms else 0.0
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"\n⚠️  Bad gradients at step {step}, skipping step"); opt.zero_grad()
        else:
            opt.step(); opt.zero_grad(); scheduler.step()
            if step % 100 == 0 and scheduler.get_last_lr()[0] <= 0: print(f"\n🚨 DEBUG: LR hit zero at step {step}!")
        accum_counter = 0
        torch.cuda.empty_cache()

    losses_window.append(loss_val)
    if len(losses_window) > 1000: losses_window.pop(0)

    if hasattr(model, '_last_info') and model._last_info: auto_tuner.update_telemetry_emas(model._last_info)

    if step % LOG_EVERY == 0 and step > 0:
        current_avg_loss = float(np.mean(losses_window[-100:])) if losses_window else float('inf')
        current_lr = opt.param_groups[0]['lr'] if hasattr(opt, 'param_groups') and opt.param_groups else PEAK_LR

        if step == 208080:
            for block in model.blocks: block.alpha_norm_target = 25.0
            print(f"\n🚨 MANUAL INTERVENTION: Depressed tau_alpha to 25.0 to force macro-state condensation.\n")

        _info_src = getattr(model, '_last_info', None) or {}
        sde_metrics = compute_sde_telemetry()
        
        _pawula_r = sde_metrics.get('pawula_ratio', 20.0)
        print(f"   🛡️ PAWULA CHECK: Var={sde_metrics.get('var_dir', 0.0):.2e} | D4={sde_metrics.get('d4_dir', 0.0):.2e} | Ratio={_pawula_r:.2f} ({'✅ FOKKER-PLANCK' if sde_metrics.get('pawula_safe', True) else '⚠️ HEAVY TAILS'})")   
        print(f"   🔬 SDE AUDIT: drift={sde_metrics.get('sde_drift_norm', 0.0):.4f} | diffusion={sde_metrics.get('sde_noise_trace', 0.0):.4f} | raw_snr={sde_metrics.get('sde_snr', 0.0):.4f}")        

        _info_src["sde_drift_norm"], _info_src["sde_noise_trace"], _info_src["sde_snr"], _info_src["sde_snr_dir"] = sde_metrics["sde_drift_norm"], sde_metrics["sde_noise_trace"], sde_metrics["sde_snr"], sde_metrics["sde_snr_dir"]
        _info_src["fiber_curvature_head_var"] = compute_fiber_curvature()

        TELEMETRY_LATTICE = {4, 8, 12, 16, 20, 23}
        fiber_vars = [block._last_fiber_curvature for idx, block in enumerate(model.blocks) if idx in TELEMETRY_LATTICE and hasattr(block, '_last_fiber_curvature')]
        if fiber_vars:
            with torch.no_grad():
                _info_src['fiber_curvature_head_var'] = torch.stack(fiber_vars).mean().item()
                _info_src['fiber_curvature_layers'] = sorted(TELEMETRY_LATTICE)
        if _info_src:
            alpha_attn_vals, alpha_ffn_vals = [], []
            for block in model.blocks:
                if hasattr(block, 'raw_alpha_attn'): alpha_attn_vals.append(float((1.0 + block.raw_alpha_attn).item()))
                elif hasattr(block, 'alpha_attn'): alpha_attn_vals.append(float(block.alpha_attn.item()))
                if hasattr(block, 'raw_alpha_ffn'): alpha_ffn_vals.append(float((1.0 + block.raw_alpha_ffn).item()))
                elif hasattr(block, 'alpha_ffn'): alpha_ffn_vals.append(float(block.alpha_ffn.item()))
            if alpha_attn_vals: _info_src['alpha_attn_min'], _info_src['alpha_attn_max'], _info_src['alpha_attn_mean'] = min(alpha_attn_vals), max(alpha_attn_vals), sum(alpha_attn_vals) / len(alpha_attn_vals)
            if alpha_ffn_vals: _info_src['alpha_ffn_min'], _info_src['alpha_ffn_max'], _info_src['alpha_ffn_mean'] = min(alpha_ffn_vals), max(alpha_ffn_vals), sum(alpha_ffn_vals) / len(alpha_ffn_vals)
            _info_src['alpha_well_depth'], _info_src['alpha_well_target'], _info_src['alpha_well_history'] = get_alpha_well_depth(opt), 1.0, getattr(auto_tuner, '_well_history', [])
            _info_src['alpha_grad_norm_mean'], _info_src['alpha_grad_norm_max'] = mean_alpha_grad, max_alpha_grad

        pressure = PressureState.from_telemetry(_info_src) if _info_src else None
        tune_decision = auto_tuner.tune(step, _info_src)

        if tune_decision.lr_multiplier is not None:
            for pg in opt.param_groups: pg['lr'] *= tune_decision.lr_multiplier

        if pressure and tune_decision.r_action:
            regime_icon = "🔴" if pressure.ccr < 1.0 else "🟡" if pressure.ccr < 2.5 else "🟢"
            print(f"   🔥 R={pressure.ccr:.3f} {regime_icon} | phase={auto_tuner.phase:.3f} | action={tune_decision.r_action}")
        if tune_decision.phase_action: print(f"   🌊 Phase action: {tune_decision.phase_action}")
        if tune_decision.actions:
            for action in tune_decision.actions: print(f"   ⚙️  {action}")

        if _info_src:
            coherence, pressure_conc, mpc_ratio, dominant = _info_src.get('coherence', 0.5), _info_src.get('pressure_concentration', 0.5), _info_src.get('mpc_intervention_ratio', 0.0), _info_src.get('dominant_governor', 'none')
            _alpha_ffn_max, _alpha_attn_max = _info_src.get('alpha_ffn_max', 1.0), _info_src.get('alpha_attn_max', 1.0)
            _alpha_drift = (abs(_alpha_ffn_max - 1.0) > 1.0) or (abs(_alpha_attn_max - 1.0) > 1.0)
            if _alpha_drift and (pressure_conc > 0.7 or mpc_ratio > 0.25 or dominant == 'mpc'): target_wd, wd_reason = 0.5, "deep_well (alpha_drift)"
            elif coherence > 0.6 and pressure_conc < 0.5: target_wd, wd_reason = 0.05, "shallow_well (healthy)"
            elif hasattr(auto_tuner, '_last_loss') and current_avg_loss < auto_tuner._last_loss * 0.99: target_wd, wd_reason = 0.0, "flat_well (loss_dropping)"
            else: target_wd, wd_reason = 0.01, "shallow_well (learning)"
            if hasattr(auto_tuner, '_well_history') and len(auto_tuner._well_history) > 0:
                if sum(auto_tuner._well_history[-10:]) / len(auto_tuner._well_history[-10:]) > 0.4 and current_avg_loss < getattr(auto_tuner, '_best_loss_since_well', float('inf')) * 0.98:
                    target_wd, wd_reason = -0.1, "INVERTED (slingshot)"; auto_tuner._best_loss_since_well = current_avg_loss
            else:
                if not hasattr(auto_tuner, '_well_history'): auto_tuner._well_history = []
                auto_tuner._best_loss_since_well = current_avg_loss
            current_wd = get_alpha_well_depth(opt)
            set_alpha_well_depth(opt, max(-0.2, min(0.6, 0.9 * current_wd + 0.1 * target_wd)))
            auto_tuner._well_history.append(float(get_alpha_well_depth(opt)))
            if len(auto_tuner._well_history) > 50: auto_tuner._well_history.pop(0)
            if step % (LOG_EVERY * 4) == 0:
                print(f"   🌍 Alpha well: WD={get_alpha_well_depth(opt):.3f} ({wd_reason}) | target={target_wd:.2f} | coh={coherence:.2f} | χ={pressure_conc:.2f}")
                print(f"   🍄 Alphas: attn_mean={_info_src.get('alpha_attn_mean', 1.0):.3f} ffn_mean={_info_src.get('alpha_ffn_mean', 1.0):.3f} ffn_max={_info_src.get('alpha_ffn_max', 1.0):.3f}")

        scheduler_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else current_lr
        if scheduler_lr <= 0:
            print(f"\n   🚨🚨🚨 SCHEDULER FROZEN at step {step}: LR={scheduler_lr:.2e}")
            new_total = max(step + 1_000_000, getattr(scheduler, 'num_training_steps', step) + 500_000)
            scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=new_total)
            for _ in range(step): scheduler.step()
            recovered_lr = scheduler.get_last_lr()[0]
            if hasattr(opt, 'sync_adamw_lr'): opt.sync_adamw_lr(recovered_lr)
            scheduler._resurrected_count = getattr(scheduler, '_resurrected_count', 0) + 1
            auto_tuner.scheduler_resurrected_count = scheduler._resurrected_count
            print(f"   🔥 Scheduler RESURRECTED #{scheduler._resurrected_count}: total={new_total:,} | LR={recovered_lr:.2e}")
            if recovered_lr <= 0: raise RuntimeError(f"CRITICAL: Scheduler resurrection failed. LR={recovered_lr:.2e}")

        stats = tracker.log(step)

        coherence = 0.0
        early_var, late_var, delta, friction, cap_hit_ratio, max_raw_norm, mean_raw_norm, info = 0.0, 0.0, 0.0, "", 0.0, 0.0, 0.0, {}

        if hasattr(model, '_last_info') and model._last_info:
            info = model._last_info
            if not hasattr(auto_tuner, '_mpc_dormant_since'): auto_tuner._mpc_dormant_since = None
            if not hasattr(auto_tuner, '_control_gain_locked'): auto_tuner._control_gain_locked = False
            mpc_recent = info.get('mpc_intervention_ratio', 0.0)
            if mpc_recent < 0.05:
                if auto_tuner._mpc_dormant_since is None:
                    auto_tuner._mpc_dormant_since = step; print(f"   🎯 MPC dormancy detected at step {step}")
                elif step - auto_tuner._mpc_dormant_since > 2000 and not auto_tuner._control_gain_locked:
                    auto_tuner._control_gain_locked = True; print(f"   🔒 control_gain LOCKED: predictor recalibrating (MPC dormant since {auto_tuner._mpc_dormant_since})")
            else:
                if auto_tuner._control_gain_locked: print(f"   🔓 control_gain UNLOCKED: MPC reactivated")
                auto_tuner._mpc_dormant_since = None; auto_tuner._control_gain_locked = False
            if not hasattr(auto_tuner, '_i_field_ema'): auto_tuner._i_field_ema = 0.70
            i_field_now = info.get('mean_instability_field', 0.0)
            if i_field_now > 0: auto_tuner._i_field_ema = 0.98 * auto_tuner._i_field_ema + 0.02 * i_field_now
            adaptive_floor = min(0.95, max(0.45, auto_tuner._i_field_ema * 1.12))
            if getattr(model.blocks[0], 'instability_target', cfg.instability_target) < adaptive_floor * 0.98:
                for block in model.blocks: block.instability_target = adaptive_floor
                cfg.instability_target = adaptive_floor
                print(f"   🎯 MPC auto-recalibrated: instability_target → {adaptive_floor:.3f} (I-EMA={auto_tuner._i_field_ema:.3f}")            
            coherence = float(info.get('coherence', 0.0)); early_var = float(info.get('early_var', 0.0)); late_var = float(info.get('late_var', 0.0))
            delta = float(info.get('variance_delta', 0.0)); cap_hit_ratio = float(info.get('soft_cap_hit_ratio', 0.0)); max_raw_norm = float(info.get('max_raw_norm', 0.0)); mean_raw_norm = float(info.get('mean_raw_norm', 0.0))
            if delta > 1.0: friction = "✅ DISSIPATED"
            elif delta < -1.0: friction = "🌋 DEEP DRIFT"
            elif early_var < 2.0 and late_var < 2.0: friction = "🟢 HARMONIZED"
            else: friction = "🟡 PROCESSING"

        coh_icon = "📈" if coherence > 0.8 else "📉" if coherence < 0.5 else "➡️"

        if hasattr(auto_tuner, '_alpha_wake_step') and auto_tuner._alpha_wake_step > 0:
            wake_progress = min(1.0, (step - auto_tuner._alpha_wake_step) / 5000.0)
            ease = 0.5 - 0.5 * math.cos(math.pi * wake_progress)
            new_alpha_target = 150.0 - ease * (150.0 - 35.0)
            for block in model.blocks: block.alpha_norm_target = new_alpha_target
            if step % LOG_EVERY == 0: print(f"   🍄 Alpha wake: progress {wake_progress*100:.1f}% | target={new_alpha_target:.1f}")

        if USE_GRADUAL_TRANSITION:
            progress = min(1.0, (step - cfg.transition_start_step) / TRANSITION_DURATION)
            transition_status = f" | 📈 Transition: {progress*100:.0f}% (FFN={current_ffn_target:.0f} α={current_alpha_target:.0f})"
        else: transition_status = ""

        _ph_ratio = mixed.get_current_ratio()
        _ph_progress = min(1.0, step / PH_STIR_TRANSITION_STEPS) if PH_STIR_TRANSITION_STEPS > 0 else 1.0
        print(f"   🧪 pH-Stir: expansion={_ph_ratio:.1%} | anchor={1-_ph_ratio:.1%} | ramp={_ph_progress:.1%} (step {step:,}/{PH_STIR_TRANSITION_STEPS:,})")

        _tapestry_counts = Counter()
        if hasattr(mixed.anchor, 'yield_counts'): _tapestry_counts.update(mixed.anchor.yield_counts)
        _exp = mixed.expansion; _exp_list = _exp if isinstance(_exp, list) else _exp.datasets if hasattr(_exp, 'datasets') else [_exp]
        for _ds in _exp_list:
            if hasattr(_ds, 'yield_counts'): _tapestry_counts.update(_ds.yield_counts)
        _total_yielded = sum(_tapestry_counts.values())
        if _total_yielded > 0:
            _audit_strs = [f"{_src}: {(_count / _total_yielded) * 100:5.2f}%" for _src, _count in _tapestry_counts.most_common()]
            print(f"   🌍 TAPESTRY AUDIT (Exact Source Attribution | N={_total_yielded:,} batches):")
            for i in range(0, len(_audit_strs), 4): print(f"      {' | '.join(_audit_strs[i:i+4])}")
        else: print(f"   🌍 TAPESTRY AUDIT: Waiting for first batches...")
        print(f"\n📊 Step {step:,} | Loss: {current_avg_loss:.4f} | LR: {current_lr:.2e} | 📉 Annealing{transition_status}")
        print(f"   Coherence: {coherence:.4f} {coh_icon}")
        if friction: print(f"   Friction: {friction} | early={early_var:.2f} late={late_var:.2f} Δ={delta:+.2f}")

        auto_tuner._friction_history.append(delta)
        if len(auto_tuner._friction_history) > 100: auto_tuner._friction_history.pop(0)
        if step % (LOG_EVERY * 5) == 0:
            print(f"   🌡️ STABILITY OBSERVATORY: | 🎯 Basin Radius: {compute_attractor_basin_radius():.3f}")
            print(f"      🌪️ Lyapunov Proxy (λ): {compute_lyapunov_proxy():+.4f} {'(Exploring)' if compute_lyapunov_proxy() > 0 else '(Converging)'}")
            print(f"      🧊 Critical Slowing Down (Autocorr): {compute_critical_slowing_down(auto_tuner._friction_history):.3f}")

        if not hasattr(auto_tuner, '_prev_drift_dir'): auto_tuner._prev_drift_dir, auto_tuner._prev_pi_alpha, auto_tuner._prev_sigma2_head = None, 0.0, 0.0
        _current_drift = sde_metrics.get('sde_drift_norm', 0.0)
        sbti = 1.0 - min(1.0, max(-1.0, _current_drift / (auto_tuner._prev_drift_dir + 1e-8))) if auto_tuner._prev_drift_dir is not None and _current_drift > 0 and auto_tuner._prev_drift_dir > 0 else 0.0
        auto_tuner._prev_drift_dir = _current_drift

        _current_pi_alpha = info.get('pressure_by_governor', {}).get('alpha', 0.0)
        _current_sigma2 = info.get('fiber_curvature_head_var', 0.0)

        IS_RESUME_WINDOW = (step <= RESUME_STEP + 5) and (step > RESUME_STEP)
        if IS_RESUME_WINDOW:
            w_gov_safe, sigma_head_dot_safe = 0.0, 0.0
            if step == RESUME_STEP + 1: print(f"⚠️ TELEMETRY SAFEGUARD: Steps {RESUME_STEP+1} to {RESUME_STEP+5} flagged as post-resume. Suppressing cold-start artifacts.")
        else:
            w_gov_safe = abs(_current_pi_alpha - getattr(auto_tuner, '_prev_pi_alpha', 0.0))
            sigma_head_dot_safe = (_current_sigma2 - getattr(auto_tuner, '_prev_sigma2_head', _current_sigma2))
            auto_tuner._prev_pi_alpha = _current_pi_alpha; auto_tuner._prev_sigma2_head = _current_sigma2

        if not hasattr(auto_tuner, 'losses_window'): auto_tuner.losses_window = []
        auto_tuner.losses_window.append(current_avg_loss)
        if len(auto_tuner.losses_window) > 10: auto_tuner.losses_window.pop(0)

        _entropy_var = compute_attention_entropy_variance()
        print(f"   🔬 KINETIC TRIGGERS & HIDDEN DESCENT: | 🧠 Attention Entropy Var: {_entropy_var:.4f}")
        print(f"      👆 Sub-Basin Jump (SBTI): {sbti:.3f} | ⚡ Control Effort (Ẇ_gov): {w_gov_safe:.2f}")
        print(f"      🧬 Decoupling Vel (σ̇²_head): {sigma_head_dot_safe:+.3f} | 🧶 Fiber Curv: {_current_sigma2:.4f}")

        _macro_R = 0.0 
        gci_score = 0.50 if IS_RESUME_WINDOW else compute_gci_snapshot(sigma_head=_current_sigma2, sigma_dot_head=sigma_head_dot_safe, attn_entropy_var=_entropy_var, W_gov=w_gov_safe, R_pawula=_pawula_r, R_t=_macro_R)
        gci_phase = "CRYSTALLIZED (Horizon Reached)" if gci_score >= 0.95 else "LATE DESCENT (Stabilizing)" if gci_score >= 0.50 else "ACTIVE DESCENT (Specializing)" if gci_score >= 0.10 else "EARLY PLASTICITY (High Variance)"
        print(f"   🧊 GCI: {gci_score:.2f} | Phase: {gci_phase}")        

        if hasattr(model.blocks[-1], '_hidden_state'):
            h = model.blocks[-1]._hidden_state
            spectral_concentration = compute_spectral_concentration(h, top_k=3)
            if spectral_concentration > 0.0:
                print(f"   🔬 Spectral concentration: {spectral_concentration:.3f}")
                if spectral_concentration > 0.95: print(f"   🚨 EARLY WARNING: Dimensional collapse detected!")

        ffn_veto_ratio, mean_ffn_norm, max_ffn_norm = info.get('ffn_veto_ratio', 0.0), info.get('mean_ffn_norm', 0.0), info.get('max_ffn_norm', 0.0)
        if ffn_veto_ratio > 0 or mean_ffn_norm > 0: print(f"   FFNVeto: {ffn_veto_ratio*100:.1f}% mean_norm={mean_ffn_norm:.1f} max_norm={max_ffn_norm:.1f} | target={current_ffn_target:.0f}")

        alpha_scale_ratio, mean_alpha_scale, mean_contrib_norm = info.get('alpha_scale_ratio', 0.0), info.get('mean_alpha_scale', 1.0), info.get('mean_contrib_norm', 0.0)
        if alpha_scale_ratio > 0 or mean_contrib_norm > 0:
            _surfer_target = getattr(model.blocks[0], 'alpha_norm_target', 30.0)
            if hasattr(_surfer_target, 'item'): _surfer_target = _surfer_target.item()
            print(f"   🏄‍♂️ SURFER_TARGET={_surfer_target:.2f}")

        if cap_hit_ratio > 0 or max_raw_norm > 0: print(f"   SoftCap: hit={cap_hit_ratio*100:.1f}% max_raw={max_raw_norm:.1f} mean_raw={mean_raw_norm:.1f}")

        mpc_intervention_ratio, mean_control_factor, mean_instability_field = info.get('mpc_intervention_ratio', 0.0), info.get('mean_control_factor', 1.0), info.get('mean_instability_field', 0.0)
        if mpc_intervention_ratio > 0 or mean_instability_field > 0:
            print(f"\n   🔮 MPC: intervene={mpc_intervention_ratio*100:.1f}% control={mean_control_factor:.3f} I={mean_instability_field:.3f}")
            print(f"   📊 Pred: {info.get('mean_prediction', 0):.3f} | Conf: {info.get('mean_confidence', 1):.3f}")
            print(f"   📈 Dynamics: v={info.get('instability_velocity', 0):+.4f} a={info.get('instability_acceleration', 0):+.4f}")
            print(f"   🎯 Forecast Error: {info.get('forecast_error', 0):.3f}")

        instability_history, confidence_history = info.get('instability_field_history', []), info.get('confidence_history', [])
        if len(instability_history) >= 6: print(f"   I-field:  {' '.join([f'{v:.2f}' for v in instability_history])}")
        if len(confidence_history) >= 6: print(f"   Conf-field:{' '.join([f'{v:.2f}' for v in confidence_history])}")

        total_pressure, pressure_conc, dominant = info.get('total_pressure', 0.0), info.get('pressure_concentration', 0.0), info.get('dominant_governor', 'none')
        if total_pressure > 0:
            print(f"\n   🔥 Π={total_pressure:.1f} | χ={pressure_conc:.2f} | dominant={dominant}")
            print(f"   🔥 Π breakdown: {' '.join([f'{k}={v:.1f}' for k, v in info.get('pressure_by_governor', {}).items()])}")
            pressure_alert = pressure_logger.update(info, step)
            if pressure_alert: print(f"   {pressure_alert}")

        rate_governor_hit, rate_scale_mean, ffn_growth, res_growth = info.get('rate_governor_hit', 0.0), info.get('rate_scale_mean', 1.0), info.get('ffn_growth_ratio', 1.0), info.get('residual_growth_ratio', 1.0)
        if use_rate and (rate_governor_hit > 0.01 or rate_scale_mean < 0.99): print(f"   📐 Rate Governor: hit={rate_governor_hit*100:.1f}% scale={rate_scale_mean:.3f} | ffn_growth={ffn_growth:.2f}x res_growth={res_growth:.2f}x")
        elif not use_rate: print(f"   📐 Rate Governor: DISABLED (transition progress < 50%)")

        sys.stdout.flush()

        active_govs = sum([1 if ffn_veto_ratio > 0.5 else 0, 1 if alpha_scale_ratio > 0.5 else 0, 1 if cap_hit_ratio > 0.5 else 0, 1 if mpc_intervention_ratio > 0.5 else 0, 1 if (use_rate and rate_governor_hit > 0.5) else 0])
        if active_govs > MAX_SIMULTANEOUS_GOVERNORS: print(f"   ⚠️  GOVERNOR INTERACTION GUARD: {active_govs} governors active, limit is {MAX_SIMULTANEOUS_GOVERNORS}")

        if USE_GRADUAL_TRANSITION and progress >= 1.0:
            if not hasattr(model, '_transition_complete_logged'):
                model._transition_complete_logged = True
                print(f"\n{'='*70}\n✅ TRANSITION COMPLETE at step {step:,}\n{'='*70}\n")
                sys.stdout.flush()

        model_pbg = (getattr(model, '_last_info', None) or {}).get('pressure_by_governor', {})
        total_alpha_work, total_ffn_work, total_mpc_work, fb_alpha, fb_ffn, fb_mpc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for block in model.blocks:
            if hasattr(block, '_last_info') and block._last_info:
                ib = block._last_info
                aw, fw, mw = ib.get('alpha_work', 0.0), ib.get('ffn_work', 0.0), ib.get('mpc_work', 0.0)
                total_alpha_work += aw; total_ffn_work += fw; total_mpc_work += mw
                fb_alpha += aw * ib.get('mean_contrib_norm', 0.0); fb_ffn += fw * ib.get('mean_ffn_norm', 0.0); fb_mpc += mw * ib.get('mean_instability_field', 0.0)
        if model_pbg: pi_alpha, pi_ffn, pi_mpc = float(model_pbg.get('alpha', 0.0)), float(model_pbg.get('ffn', 0.0)), float(model_pbg.get('mpc', 0.0))
        else: pi_alpha, pi_ffn, pi_mpc = fb_alpha, fb_ffn, fb_mpc
        denominator = pi_ffn + pi_mpc
        R = pi_alpha / denominator if denominator > 1e-8 else (float('inf') if pi_alpha > 1e-8 else 0.0)
        regime_state = "CONSTRUCTIVE" if R > 1.5 else "MARGINAL" if R > 0.8 else "COMPENSATORY"
        R_print = f"{R:.3f}" if R != float('inf') else "∞"
        print(f"   🔭 Macroscopic R={R_print} | Regime: {regime_state} (Π_α={pi_alpha:.2f}, Π_FFN={pi_ffn:.2f}, Π_MPC={pi_mpc:.2f} | raw: α={total_alpha_work:.2f} ffn={total_ffn_work:.2f} mpc={total_mpc_work:.2f})")

        current_R_chi = float(R) if R != float('inf') else 999.0
        current_lambda_w = get_alpha_well_depth(opt)
        if getattr(auto_tuner, '_chi_state', None) is None:
            auto_tuner._chi_state = {'prev_R': current_R_chi, 'prev_loss': float(current_avg_loss) if np.isfinite(current_avg_loss) else 99.0, 'prev_lambda_w': current_lambda_w, 'chi_R_ema': 0.0, 'prev_chi_R_ema': None}
        s = auto_tuner._chi_state
        delta_R, delta_loss, delta_lambda = current_R_chi - s['prev_R'], (float(current_avg_loss) if np.isfinite(current_avg_loss) else 99.0) - s['prev_loss'], current_lambda_w - s['prev_lambda_w']
        chi_R_raw = delta_loss / delta_R if abs(delta_R) > 0.01 else 0.0 
        gauge_sens = delta_R / delta_lambda if abs(delta_lambda) > 1e-6 else 0.0
        s['chi_R_ema'] = 0.9 * s['chi_R_ema'] + 0.1 * chi_R_raw
        info['optimization_response_chi_R'], info['gauge_sensitivity_dR_dlambda'] = s['chi_R_ema'], gauge_sens
        if s['prev_chi_R_ema'] is not None:
            if s['prev_chi_R_ema'] < 0 and s['chi_R_ema'] > 0: print(f"🌌 SLT PHASE TRANSITION: chi_R flipped POSITIVE ({s['prev_chi_R_ema']:+.4f} -> {s['chi_R_ema']:+.4f}). Entering Grokking Window.")
            elif s['prev_chi_R_ema'] > 0 and s['chi_R_ema'] < 0: print(f"⚠️  SLT REGRESSION: chi_R flipped NEGATIVE ({s['prev_chi_R_ema']:+.4f} -> {s['chi_R_ema']:+.4f}). Entering Compensatory Degeneracy.")
        s['prev_R'], s['prev_loss'], s['prev_lambda_w'], s['prev_chi_R_ema'] = current_R_chi, float(current_avg_loss) if np.isfinite(current_avg_loss) else 99.0, current_lambda_w, s['chi_R_ema']

        _chi_R = info.get('optimization_response_chi_R', 0.0)
        rho_raw, rho_dir = _info_src.get('sde_snr', 0.0), _info_src.get('sde_snr_dir', 0.0)
        regime_raw = "🟢 BALLISTIC" if rho_raw > 1.5 else "🔴 STOCHASTIC"
        regime_dir = "🟢 BALLISTIC" if rho_dir > 1.5 else "🔴 STOCHASTIC"
        _fiber_var = info.get('fiber_curvature_head_var', 0.0)
        print(f"   🌌 chi_R={_chi_R:.4f} | 🌊 SDE ρ_raw={rho_raw:.3f} ({regime_raw}) | ρ_dir={rho_dir:.3f} ({regime_dir})")
        print(f"   🧭 Cross-token coherence c_l={sde_metrics.get('directional_coherence_global', 0.0):.3f}")
        print(f"   🧶 Fiber Curvature (σ²_head): {_fiber_var:.4f} | Heads {'SPECIALIZING' if _fiber_var > 0.5 else 'COLLAPSED'}")

        if step > 0 and step % EVAL_INTERVAL == 0:
            print(f"\n🔬 [SCHEDULED EVAL] Running independent SNLI audit at Step {step}...")
            eval_results = evaluate_snli_independent(model=model, tokenizer=tokenizer, snli_dataloader=snli_dataloader, device=device, step=step)
            current_telemetry = {"internal_delta": delta, "internal_pawula_ratio": _pawula_r, "internal_rho_dir": rho_dir}
            with open(EVAL_CSV_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([eval_results["step"], f"{eval_results['snli_accuracy']:.4f}", f"{eval_results['snli_entropy']:.4f}", eval_results["total_samples"], f"{current_telemetry['internal_delta']:.4f}", f"{current_telemetry['internal_pawula_ratio']:.2f}", f"{current_telemetry['internal_rho_dir']:.2f}", f"{eval_results['massif_instability_score']:.4f}", eval_results["massif_failure_flags"]])
            print(f"✅ [SCHEDULED EVAL] Complete. Accuracy: {eval_results['snli_accuracy']:.4f} | Entropy: {eval_results['snli_entropy']:.4f}")
            print(f"🌡️ [MASSIF OBSERVATORY] Inference Instability: {eval_results['massif_instability_score']:.4f} | Flags: {eval_results['massif_failure_flags']}")
            # 🧹 Final safety net to guarantee zero fragmentation before the next backward pass
            gc.collect()
            torch.cuda.empty_cache()
            sys.stdout.flush()

    if R < 0.1 and regime_state == "COMPENSATORY" and not getattr(auto_tuner, '_lr_dampened_this_episode', False):
        dampened_lr = current_lr * 0.60
        for pg in opt.param_groups: pg['lr'] = dampened_lr
        if hasattr(opt, 'sync_adamw_lr'): opt.sync_adamw_lr(dampened_lr)
        current_lr = dampened_lr; auto_tuner._lr_dampened_this_episode = True
        print(f"   🎛️ R-guard LR dampened (ONE-TIME): {current_lr:.2e} (R={R:.3f}, compensatory)")
    elif R >= 0.1:
        auto_tuner._lr_dampened_this_episode = False
        if regime_state == "COMPENSATORY" and cfg.instability_target < 0.80:
            for block in model.blocks: block.instability_target = 0.80
            cfg.instability_target = 0.80
            print(f"   🎯 MPC compensatory relief: instability_target → 0.80 (R={R_print})")
        _mpc_recent, _forecast_err = _info_src.get('mpc_intervention_ratio', 0.0), _info_src.get('forecast_error', 0.0)

    if not hasattr(auto_tuner, '_mpc_release_step'): auto_tuner._mpc_release_step = 0
    _in_grace = False
    if _mpc_recent < 0.05:
        if auto_tuner._mpc_release_step == 0:
            auto_tuner._mpc_release_step = step; print(f"   🎯 MPC predictor recalibration started at step {step}")
        elif step - auto_tuner._mpc_release_step < 15000: _in_grace = True
        else: auto_tuner._mpc_release_step = 0

        model._meta_enrichment = {'predictor_recalibrating': _in_grace, 'recalibration_age': step - auto_tuner._mpc_release_step if _in_grace else 0, 'forecast_error': _forecast_err, 'mpc_intervention_ratio': _mpc_recent, 'R_regime': regime_state, 'R_value': 999.0 if R == float('inf') else float(R), 'control_gain_locked': _in_grace and _forecast_err > 0.5, 'control_gain_floor': 0.85 if (R < 0.1) else 0.01, 'alpha_wake_active': (hasattr(auto_tuner, '_alpha_wake_step') and (step - auto_tuner._alpha_wake_step) > 1000), 'alpha_norm_target': getattr(model.blocks[0], 'alpha_norm_target', 150.0), 'instability_target': cfg.instability_target, 'well_depth': get_alpha_well_depth(opt), 'locked_levers': (['control_gain'] if (_in_grace and _forecast_err > 0.5) else []), 'recommended_levers': ['alpha_well_depth', 'instability_target', 'alpha_norm_target'] + ([] if (_in_grace and _forecast_err > 0.5) else ['control_gain']), 'fiber_curvature_head_var': _info_src.get('fiber_curvature_head_var', 0.0), 'optimization_response_chi_R': info.get('optimization_response_chi_R', 0.0), 'gauge_sensitivity_dR_dlambda': info.get('gauge_sensitivity_dR_dlambda', 0.0), 'mean_alpha_scale': _info_src.get('mean_alpha_scale', 1.0), 'mean_contrib_norm': _info_src.get('mean_contrib_norm', 0.0)}

        meta_actions = integrate_meta_governor(model=model, auto_tuner=auto_tuner, step=step, current_loss=current_avg_loss, current_lr=current_lr, scheduler=scheduler, log_every=LOG_EVERY, local_only=False)

        if meta_actions:
            if "COMMAND: EXPAND_SOFTCAP_WINDOW" in meta_actions:
                for block in model.blocks:
                    if hasattr(block, 'soft_cap'): block.soft_cap = max(getattr(block, 'soft_cap', 15.0), 18.0)
                print("   ️ FRICTION RAIL: SoftCap window temporarily expanded.")
            if "COMMAND: DROPLET_LR_10_PERCENT" in meta_actions:
                for param_group in opt.param_groups: param_group['lr'] *= 0.90
                print(f"   🛡️ FRICTION RAIL: Learning rate dropped to {opt.param_groups[0]['lr']:.2e}")
            for _action in meta_actions:
                if "SURFER" in str(_action): print(f"   🏄‍♂️ {_action}")
        if meta_actions and meta_actions[0] not in ["CIRCUIT_BREAKER_ACTIVE", "NO_CONSENSUS", "LOCAL_RULE: no_action", "META_RATE_LIMITED", "META_INTERACTION_GUARD"]:
            print(f"   🛠  Meta-Governor: {meta_actions}")
            for action in meta_actions:
                if action.startswith("ALPHA_WELL_DEPTH_"):
                    parts = action.split("_")
                    if len(parts) >= 4:
                        direction = parts[3]
                        try: value = float(parts[4]) if len(parts) > 4 else None
                        except (ValueError, IndexError): value = None
                        current_wd = get_alpha_well_depth(opt)
                        if direction == "raise" and value is not None: new_wd = min(0.6, current_wd + value)
                        elif direction == "lower" and value is not None: new_wd = max(-0.2, current_wd - value)
                        elif direction == "set" and value is not None: new_wd = max(-0.2, min(0.6, value))
                        else: continue
                        set_alpha_well_depth(opt, new_wd)
                        print(f"   🌍 Meta-Gov alpha well: {current_wd:.3f} → {new_wd:.3f} ({direction})")
                elif action.startswith("ALPHA_WELL_INVERT_"):
                    parts = action.split("_")
                    if len(parts) >= 4:
                        direction = parts[3]
                        try: value = float(parts[4]) if len(parts) > 4 else -0.1
                        except (ValueError, IndexError): value = -0.1
                        if direction in ("raise", "set"): set_alpha_well_depth(opt, max(-0.2, value)); print(f"   🌍 Meta-Gov alpha well INVERTED: {value:.3f} (slingshot)")
                        elif direction in ("lower", "release"): set_alpha_well_depth(opt, 0.3); print(f"   🌍 Meta-Gov alpha well restored: 0.3")
                elif action.startswith("SUGGEST_control_gain_"):
                    parts = action.split("_")
                    if len(parts) >= 5:
                        direction = parts[-2]
                        try: multiplier = float(parts[-1])
                        except (ValueError, IndexError): multiplier = None
                        if multiplier is not None:
                            mpc_recent, forecast_err = _info_src.get('mpc_intervention_ratio', 0.0), _info_src.get('forecast_error', 0.0)
                            in_grace = False
                            if not hasattr(auto_tuner, '_mpc_release_step'): auto_tuner._mpc_release_step = 0
                            if mpc_recent < 0.05:
                                if auto_tuner._mpc_release_step == 0:
                                    auto_tuner._mpc_release_step = step; print(f"   🎯 MPC predictor recalibration started at step {step}")
                                elif step - auto_tuner._mpc_release_step < 15000: in_grace = True
                            else: auto_tuner._mpc_release_step = 0
                            if direction == "lower" and in_grace and forecast_err > 0.5:
                                print(f"   🎛️ Meta-Gov REJECTED: control_gain lower blocked (predictor recalibrating: forecast_err={forecast_err:.3f}, grace={(step - auto_tuner._mpc_release_step)} steps)"); continue
                            for block in model.blocks:
                                current = getattr(block, 'control_gain', CONTROL_GAIN_DEFAULT)
                                new_val = multiplier if direction == "set" else current / multiplier if direction == "raise" else current * multiplier
                                block.control_gain = max(0.85 if R < 0.1 else 0.01, min(10.0, new_val))
                            cfg.control_gain = model.blocks[-1].control_gain
                            print(f"   🎛️ Meta-Gov APPLIED: control_gain → {cfg.control_gain:.3f} ({direction} {'÷' if direction == 'raise' else '×'}{multiplier:.3f})" + (" [GRACE]" if in_grace else ""))
                elif action.startswith("SUGGEST_instability_target_"):
                    parts = action.split("_")
                    if len(parts) >= 5:
                        direction = parts[-2]
                        try: multiplier = float(parts[-1])
                        except (ValueError, IndexError): multiplier = None
                        if multiplier is not None:
                            for block in model.blocks:
                                current = getattr(block, 'instability_target', 0.45)
                                new_val = multiplier if direction == "set" else current / multiplier if direction == "raise" else current * multiplier
                                block.instability_target = max(0.01, min(1.0, new_val))
                            cfg.instability_target = model.blocks[-1].instability_target
                            print(f"   🎯 Meta-Gov APPLIED: instability_target → {cfg.instability_target:.3f} ({direction} {'÷' if direction == 'raise' else '×'}{multiplier:.3f})")

        if hasattr(auto_tuner, '_meta_governor'):
            status = auto_tuner._meta_governor.get_status()
            print(f"   📡 Meta-Gov: CB={'🔴' if status.get('circuit_breaker', False) else '🟢'} | memory={status.get('memory_size', 0)} | pending={status.get('pending_verifications', 0)} | ε={status.get('exploration_rate', 0.0):.2f}")

        current_R, current_lambda_w = R, get_alpha_well_depth(opt)
        prev_R, prev_lambda_w, prev_chi_R = getattr(auto_tuner, '_prev_R', None), getattr(auto_tuner, '_prev_lambda_w', None), getattr(auto_tuner, '_prev_chi_R', 0.0)
        if prev_R is not None and prev_lambda_w is not None:
            delta_R, delta_lambda = current_R - prev_R, current_lambda_w - prev_lambda_w
            chi_R = delta_R / delta_lambda if abs(delta_lambda) > 1e-6 else prev_chi_R
            if prev_chi_R < 0 and chi_R > 0: print("   🌌 SLT PHASE TRANSITION: chi_R flipped POSITIVE → Grokking Window")
            elif prev_chi_R > 0 and chi_R < 0: print("   ⚠️ SLT REGRESSION: chi_R flipped NEGATIVE → Compensatory Degeneracy")
            auto_tuner._prev_chi_R = chi_R
        auto_tuner._prev_R, auto_tuner._prev_lambda_w = current_R, current_lambda_w

        if current_avg_loss < best_loss:
            best_loss, best_step = current_avg_loss, step
            _lineage_receipt = compute_lineage_receipt(model, step=step, epoch=start_epoch, cfg=cfg, batch_size=BATCH_SIZE, accum_steps=ACCUM_STEPS, max_seq_len=MAX_SEQ_LEN, peak_lr=PEAK_LR, min_lr=MIN_LR, warmup_steps=WARMUP_STEPS, grad_clip=GRAD_CLIP, weight_decay=WEIGHT_DECAY, ffn_target_start=FFN_TARGET_START, ffn_target_end=FFN_TARGET_END, alpha_target_start=ALPHA_TARGET_START, alpha_target_end=ALPHA_TARGET_END, transition_duration=TRANSITION_DURATION)
            ckpt_data = _add_checksum({'epoch': start_epoch, 'global_step': step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'auto_tuner_state': auto_tuner.get_state(), 'loss': float(current_avg_loss), 'best_loss': float(best_loss), 'coherence': float(coherence), 'friction': friction, 'early_var': float(early_var), 'late_var': float(late_var), 'delta': float(delta), 'timestamp': datetime.now().isoformat(), 'lineage_receipt': _lineage_receipt})
            try:
                torch.save(ckpt_data, BEST_CKPT)
                if os.path.getsize(BEST_CKPT) / 1e9 > 1.0: print(f"\n🏆 BEST SAVED: {best_loss:.4f} at step {step:,} ({os.path.getsize(BEST_CKPT) / 1e9:.2f} GB)")
            except Exception as e: print(f"\n🚨 BEST save FAILED at step {step}: {e}")
            sys.stdout.flush()

        old_val = cfg.instability_target
        for block in model.blocks:
            if hasattr(block, 'instability_target'): block.instability_target = min(0.85, max(0.40, block.instability_target))
        cfg.instability_target = min(0.85, max(0.40, cfg.instability_target))
        if abs(cfg.instability_target - old_val) > 1e-6: print(f"   🔒 instability_target clamped: {old_val:.3f} → {cfg.instability_target:.3f}")

    if step % SAVE_EVERY == 0 and step > 0:
        _lineage_receipt = compute_lineage_receipt(model, step=step, epoch=start_epoch, cfg=cfg, batch_size=BATCH_SIZE, accum_steps=ACCUM_STEPS, max_seq_len=MAX_SEQ_LEN, peak_lr=PEAK_LR, min_lr=MIN_LR, warmup_steps=WARMUP_STEPS, grad_clip=GRAD_CLIP, weight_decay=WEIGHT_DECAY, ffn_target_start=FFN_TARGET_START, ffn_target_end=FFN_TARGET_END, alpha_target_start=ALPHA_TARGET_START, alpha_target_end=ALPHA_TARGET_END, transition_duration=TRANSITION_DURATION)
        print(f"   📜 Lineage receipt: {_lineage_receipt.get('checkpoint_hash', 'N/A')[:16]}... (arch={_lineage_receipt.get('architecture_hash', 'N/A')[:8]}, loop={_lineage_receipt.get('training_loop_hash', 'N/A')[:8]})")
        ckpt_data = _add_checksum({'epoch': start_epoch, 'global_step': step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'auto_tuner_state': auto_tuner.get_state(), 'scheduler_resurrected_count': getattr(scheduler, '_resurrected_count', 0), 'loss': float(losses_window[-1]) if losses_window else None, 'avg_loss_100': float(np.mean(losses_window[-100:])) if len(losses_window) >= 100 else None, 'best_loss': float(best_loss), 'best_step': best_step, 'timestamp': datetime.now().isoformat(), 'lineage_receipt': _lineage_receipt, 'alpha_well_history': getattr(auto_tuner, '_well_history', []), 'alpha_best_loss_since_well': getattr(auto_tuner, '_best_loss_since_well', float('inf'))})
        path = os.path.join(CKPT_DIR, f"mycelia_step_{step:05x}.pt")
        try:
            torch.save(ckpt_data, path); torch.save(ckpt_data, LATEST_CKPT)
            print(f"\n💾 Checkpoint: step {step:,} → {path}")
            cleanup_checkpoints(CKPT_DIR)
        except Exception as e: print(f"\n🚨 Checkpoint save failed: {e}")
        sys.stdout.flush()
        _log_path = "/home/ec2-user/SageMaker/training.log"
        if os.path.exists(_log_path):
            try:
                with open(_log_path, "r") as _lf: _lines = _lf.readlines()[-2000:] 
                with open(f"/home/ec2-user/SageMaker/mycelia_checkpoints/training_log_step_{step}.txt", "w") as _sf: _sf.writelines(_lines)
            except Exception as _e: print(f"   ⚠️ Log snapshot failed: {_e}")

    if step % CACHE_CLEAN_EVERY == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache(); gc.collect()
    if prof is not None:
        prof.stop()
        print("\n" + "="*80 + "\n🔬 TOP 25 OPS BY CUDA TIME\n" + "="*80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        prof.export_chrome_trace(f"mycelia_trace_step_{step}.json")
        print(f"\n💾 Trace saved → mycelia_trace_step_{step}.json\n" + "="*80 + "\n")
        prof = None 

if hasattr(auto_tuner, '_meta_governor'):
    if hasattr(auto_tuner._meta_governor, 'shutdown'): auto_tuner._meta_governor.shutdown(); print("   📡 Meta-Governor shutdown complete")

print("\n" + "="*70 + "\n💾 Final save...")
_lineage_receipt = compute_lineage_receipt(model, step=step, epoch=start_epoch, cfg=cfg, batch_size=BATCH_SIZE, accum_steps=ACCUM_STEPS, max_seq_len=MAX_SEQ_LEN, peak_lr=PEAK_LR, min_lr=MIN_LR, warmup_steps=WARMUP_STEPS, grad_clip=GRAD_CLIP, weight_decay=WEIGHT_DECAY, ffn_target_start=FFN_TARGET_START, ffn_target_end=FFN_TARGET_END, alpha_target_start=ALPHA_TARGET_START, alpha_target_end=ALPHA_TARGET_END, transition_duration=TRANSITION_DURATION)
print(f"   📜 Lineage receipt: {_lineage_receipt.get('checkpoint_hash', 'N/A')[:16]}... (arch={_lineage_receipt.get('architecture_hash', 'N/A')[:8]}, loop={_lineage_receipt.get('training_loop_hash', 'N/A')[:8]})")
final = _add_checksum({'epoch': start_epoch, 'global_step': step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'auto_tuner_state': auto_tuner.get_state(), 'scheduler_resurrected_count': getattr(scheduler, '_resurrected_count', 0), 'loss': float(losses_window[-1]) if losses_window else None, 'avg_loss_100': float(np.mean(losses_window[-100:])) if len(losses_window) >= 100 else None, 'best_loss': float(best_loss), 'best_step': best_step, 'timestamp': datetime.now().isoformat(), 'lineage_receipt': _lineage_receipt, 'alpha_well_history': getattr(auto_tuner, '_well_history', []), 'alpha_best_loss_since_well': getattr(auto_tuner, '_best_loss_since_well', float('inf'))})
for ckpt_path, label in [(LATEST_CKPT, "LATEST"), (BEST_CKPT, "BEST")]:
    try: torch.save(final, ckpt_path); print(f"   ✅ {label} saved")
    except Exception as e: print(f"   🚨 {label} save failed: {e}")

print("\n" + "="*70 + "\n🍄 MYCELIA TRAINING v12.9 (1.5B Muon + pH-Stirred Composite + MASSIF Observatory) COMPLETE\n" + "="*70)