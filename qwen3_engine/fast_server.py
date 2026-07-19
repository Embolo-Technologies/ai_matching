import os
os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
import sys
import time
import re
import json
import subprocess
import queue
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from flask_cors import CORS
from rapidfuzz import process, fuzz
from qwen3_engine.searcher import FuzzySearcher
from qwen3_engine.pharma_data import formulations_compatible

# ─── GPU Concurrency & Engine Pool Helpers ──────────────────────────────────────────
def get_gpu_info() -> tuple:
    """
    Returns (num_gpus, vram_mb_per_gpu) where vram_mb_per_gpu is a list.
    Falls back to (0, []) if nvidia-smi is unavailable.
    """
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5, check=True
        )
        lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip().isdigit()]
        vram_list = [int(x) for x in lines]
        return len(vram_list), vram_list
    except Exception:
        return 0, []


def calculate_optimal_workers(model_key: str) -> int:
    """
    Auto-detects number of GPUs and VRAM per GPU.
    Returns total safe worker count across all GPUs.
    Each worker needs ~1.82 GB (model weights + KV cache at n_ctx=1024).
    Leaves ~4 GB buffer per GPU for safety.
    """
    num_gpus, vram_list = get_gpu_info()
    if not vram_list:
        return 16  # safe CPU-only fallback

    total_workers = 0
    for vram_mb in vram_list:
        safe_workers = max(1, int((vram_mb - 4000) // 1900))
        total_workers += safe_workers

    print(f"[Workers] {num_gpus} GPU(s) detected: {vram_list} MB each → {total_workers} total workers")
    return total_workers


def get_num_gpus() -> int:
    """Returns number of available GPUs (0 if none)."""
    num_gpus, _ = get_gpu_info()
    return num_gpus


class EnginePool:
    """Thread-safe pool of independent model instances spread across all available GPUs."""
    def __init__(self, model_key: str, size: int):
        self.model_key = model_key
        self.size = size
        self.pool = queue.Queue()
        self.num_gpus = max(1, get_num_gpus())

    def populate(self):
        import os as _os
        from qwen3_engine.vllm_engine import VLLMEngine
        backend_urls = [u.strip() for u in _os.environ.get(
            "LLAMA_BACKEND_URLS", "http://localhost:8000/v1"
        ).split(",") if u.strip()]

        ready_backends = []
        for url in backend_urls:
            probe = VLLMEngine(base_url=url)
            if probe.check_ready(timeout=2.0):
                ready_backends.append(url)
                print(f"[EnginePool] llama_cpp.server detected at {url} — ready.")
            else:
                print(f"[EnginePool] WARNING: backend {url} not ready, skipping.")

        if ready_backends:
            for i in range(self.size):
                url = ready_backends[i % len(ready_backends)]
                engine = VLLMEngine(base_url=url)
                engine._ready = True
                self.pool.put(engine)
            print(f"[EnginePool] ✓ {self.size} workers wired across {len(ready_backends)} backend(s): {ready_backends}")
            return

        from qwen3_engine.engine import Qwen3Engine
        from qwen3_engine.config import N_CTX_MATCHER
        print(f"[EnginePool] llama_cpp.server not detected. Loading local {self.size} models across {self.num_gpus} GPU(s)...")
        for i in range(self.size):
            gpu_id = i % self.num_gpus
            t0 = time.time()
            engine = Qwen3Engine(model_key=self.model_key, gpu_id=gpu_id)
            engine.load(n_ctx=N_CTX_MATCHER, num_gpus=self.num_gpus)
            self.pool.put(engine)
            print(f"  ✓ Worker {i+1}/{self.size} on GPU {gpu_id} ready in {time.time()-t0:.1f}s")

    def lease(self):
        return self.pool.get()
        
    def release(self, engine):
        self.pool.put(engine)

# ─── Batch Match Helper Functions ────────────────────────────────────────────────
def extract_strength(name_str):
    if not isinstance(name_str, str):
        return ""
    # Compound doses like "500/125MG" or "875/125" — return full slash expression
    # so both components end up in c_nums during strength matching.
    compound = re.search(r'\b\d+(?:\.\d+)?/\d+(?:\.\d+)?\s*(?:mg|ml|gm|g|mcg)?\b', name_str, re.IGNORECASE)
    if compound:
        return compound.group(0)
    match = re.search(r'\b\d+(?:\.\d+)?\s*(?:mg|ml|gm|g|mcg|cap|tab)\b', name_str, re.IGNORECASE)
    if match:
        return match.group(0)
    match_num = re.search(r'\b\d+(?:\.\d+)?\b', name_str)
    if match_num:
        val = match_num.group(0)
        try:
            if float(val) <= 10.0 and '.' not in val:
                return ""
        except ValueError:
            pass
        return val
    return ""

def compute_confidence(query_name, matched_name, matched_brand, query_pack, matched_pack):
    if not matched_name:
        return 0
    q = query_name.lower().strip()
    m = matched_name.lower().strip()
    q_words = re.sub(r'[^a-z0-9\s]', '', q).split()
    m_words = re.sub(r'[^a-z0-9\s]', '', m).split()
    q_brand = q_words[0] if q_words else ""
    m_brand = m_words[0] if m_words else ""
    if len(q_brand) <= 2 and len(q_words) > 1:
        q_brand = q_words[1]
    brand_score = fuzz.ratio(q_brand, m_brand)
    full_score = fuzz.token_sort_ratio(q, m)
    q_nums = set(re.findall(r'\d+\.?\d*', q))
    m_nums = set(re.findall(r'\d+\.?\d*', m))
    PACK_NUMS = {'1','2','3','4','5','6','7','8','9','10'}
    q_strengths = q_nums - PACK_NUMS
    m_strengths = m_nums - PACK_NUMS
    strength_bonus = 0
    if q_strengths and m_strengths:
        if q_strengths & m_strengths:
            strength_bonus = 10
        else:
            strength_bonus = -15
    confidence = (brand_score * 0.5) + (full_score * 0.4) + strength_bonus
    return max(0, min(100, round(confidence)))

# ─── Company Compatibility Helper Functions ──────────────────────────────────────────
def clean_company_name(name_str: str) -> str:
    if not name_str:
        return ""
    name_str = name_str.lower().strip()
    if name_str in {"", "--", "none", "null", "nan", "undefined", "unknown", "-"}:
        return ""
    # Remove common punctuation
    name_str = re.sub(r'[^a-z0-9\s]', ' ', name_str)
    # Remove common business suffixes
    stopwords = {
        'laboratories', 'laboratory', 'labs', 'lab', 'pharma', 'pharmaceuticals', 'pharmaceutical', 'therapeutics', 
        'healthcare', 'lifesciences', 'life', 'sciences', 'pvt', 'ltd', 'private', 'limited', 'india', 'inc', 'corp', 
        'corporation', 'co', 'gmbh', 'sa', 'ag', 'limited', 'ltd'
    }
    words = name_str.split()
    cleaned_words = [w for w in words if w not in stopwords]
    if not cleaned_words:
        cleaned_words = words
    return " ".join(cleaned_words).strip()

def _split_camel_case_words(word):
    """Split 'SmithKline' -> ['Smith', 'Kline']."""
    parts = re.findall(r'[A-Z][a-z]*|[a-z]+|[A-Z]+(?![a-z])', word)
    return [p for p in parts if p]

def is_company_acronym(abbrev, full_name):
    """
    General acronym check: does `abbrev` (e.g. 'GSK') match the initials of
    `full_name`'s significant words (e.g. 'Glaxo SmithKline' -> G+S+K)?
    Works for any company, not hardcoded to a specific one.
    """
    abbrev = abbrev.strip().lower()
    if not abbrev.isalpha() or not (2 <= len(abbrev) <= 6):
        return False

    stopwords = {
        'laboratories', 'laboratory', 'labs', 'lab', 'pharma', 'pharmaceuticals', 'pharmaceutical',
        'therapeutics', 'healthcare', 'lifesciences', 'life', 'sciences', 'pvt', 'ltd', 'private',
        'limited', 'india', 'inc', 'corp', 'corporation', 'co', 'gmbh', 'sa', 'ag', 'and', 'the', 'of'
    }
    raw_words = re.findall(r"[A-Za-z]+", full_name)
    sub_words = []
    for w in raw_words:
        if w.lower() in stopwords:
            continue
        sub_words.extend(_split_camel_case_words(w))

    if not sub_words:
        return False

    initials = ''.join(w[0].lower() for w in sub_words if w)
    if initials == abbrev:
        return True
    if abbrev in initials:
        return True
    return False

def companies_compatible(q_comp: str, c_comp: str) -> bool:
    q_clean = clean_company_name(q_comp)
    c_clean = clean_company_name(c_comp)
    
    if not q_clean or not c_clean:
        return True  # If either side is missing, it's compatible (no info to conflict)
        
    if q_clean == c_clean:
        return True
        
    q_words = set(q_clean.split())
    c_words = set(c_clean.split())
    
    # Overlap of non-trivial words (length >= 3)
    non_trivial_overlap = {w for w in (q_words & c_words) if len(w) >= 3}
    if non_trivial_overlap:
        return True
        
    # General acronym check: does either short side match the initials
    # of the other's significant words? Catches 'GSK'='Glaxo SmithKline' etc.
    if is_company_acronym(q_comp, c_comp) or is_company_acronym(c_comp, q_comp):
        return True

    # Fuzzy ratio match
    score = fuzz.ratio(q_clean, c_clean)
    if score >= 75:
        return True
        
    # Partial ratio match for substrings
    if len(q_clean) >= 4 and len(c_clean) >= 4:
        partial = fuzz.partial_ratio(q_clean, c_clean)
        if partial >= 85:
            return True
            
    return False

BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)

_searcher = None
_llm_engines = {}   # Lazy-loaded AI engines for fallback: model_key -> Qwen3Engine

# ─── Diagnostic timing accumulator (answers: where does the per-item time go?) ──
_timing_lock = threading.Lock()
_timing_stats = {
    "n_total": 0, "n_llm": 0,
    "search_ms_sum": 0.0, "heur_ms_sum": 0.0, "llm_ms_sum": 0.0,
    "search_ms_max": 0.0, "llm_ms_max": 0.0,
}
_inflight_lock = threading.Lock()
_inflight_count = 0
_inflight_max = 0

def _record_timing(search_ms, heur_ms, llm_ms, llm: bool):
    with _timing_lock:
        _timing_stats["n_total"] += 1
        _timing_stats["search_ms_sum"] += search_ms
        _timing_stats["heur_ms_sum"] += heur_ms
        _timing_stats["search_ms_max"] = max(_timing_stats["search_ms_max"], search_ms)
        if llm:
            _timing_stats["n_llm"] += 1
            _timing_stats["llm_ms_sum"] += llm_ms
            _timing_stats["llm_ms_max"] = max(_timing_stats["llm_ms_max"], llm_ms)

def _inflight_enter():
    global _inflight_max
    with _inflight_lock:
        global _inflight_count
        _inflight_count += 1
        _inflight_max = max(_inflight_max, _inflight_count)

def _inflight_exit():
    with _inflight_lock:
        global _inflight_count
        _inflight_count -= 1

@app.route("/timing", methods=["GET", "POST"])
def timing_stats():
    """GET: return accumulated per-item timing stats. POST: reset them."""
    if request.method == "POST":
        with _timing_lock:
            for k in _timing_stats:
                _timing_stats[k] = 0 if isinstance(_timing_stats[k], int) else 0.0
        global _inflight_max
        with _inflight_lock:
            _inflight_max = 0
        return jsonify({"reset": True})
    with _timing_lock:
        s = dict(_timing_stats)
    n = max(1, s["n_total"])
    n_llm = max(1, s["n_llm"])
    return jsonify({
        "n_total": s["n_total"],
        "n_llm": s["n_llm"],
        "llm_fraction": round(s["n_llm"] / n, 3),
        "avg_search_ms": round(s["search_ms_sum"] / n, 2),
        "max_search_ms": round(s["search_ms_max"], 2),
        "avg_heur_ms": round(s["heur_ms_sum"] / n, 2),
        "avg_llm_ms": round(s["llm_ms_sum"] / n_llm, 2) if s["n_llm"] else 0,
        "max_llm_ms": round(s["llm_ms_max"], 2),
        "max_concurrent_llm_calls": _inflight_max,
    })

def is_abbreviation(abbrev: str, full: str) -> bool:
    abbrev = abbrev.lower().strip()
    full = full.lower().strip()
    if len(abbrev) < 3 or not full:
        return False
    if abbrev[0] != full[0]:
        return False
    it = iter(full)
    return all(char in it for char in abbrev)

class FastMatcher:
    def __init__(self, searcher_instance, model_key: str = "gemma4_2b"):
        self.searcher = searcher_instance
        self.catalog = []
        self.model_key = model_key
        
    def load_catalog(self):
        self.catalog = []
        for cand in self.searcher.catalog:
            self.catalog.append({
                "code": cand["code"],
                "name": cand["name"],
                "brand": cand["compname"],
                "pack": cand["pack"],
                "strength": cand["strength"],
            })

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        # Replace separators with spaces
        text = text.replace("-", " ").replace("/", " ").replace(",", " ")
        # Split stuck numbers from words: PAN40 -> PAN 40, DOLO650 -> DOLO 650
        text = re.sub(r'([a-z])([0-9])', r'\1 \2', text)
        text = re.sub(r'([0-9])([a-z])', r'\1 \2', text)
        # Remove formulation keywords
        text = re.sub(r'\b(tabs?|tablets?|inj(ection)?s?|caps?(ules?)?|susp(ension)?|syp|syrup|drops?|cream|ointment|gel|liq(uid)?|sol(ution)?)\b', '', text)
        # Remove all non-alphanumeric except spaces and dots
        text = re.sub(r'[^a-z0-9\s\.]', '', text)
        tokens = text.split()
        # Strip leading pure-numeric vendor codes (e.g. "001 PAN 40" → "pan 40")
        # BUT stop if next token is short (≤2 alpha chars) — that means the number IS part of
        # the brand name, e.g. "36 D" or "4 PH" (after "4ph" is split by the regex above).
        while tokens and re.match(r'^\d+$', tokens[0]):
            if len(tokens) > 1 and sum(c.isalpha() for c in tokens[1]) >= 3:
                tokens.pop(0)
            else:
                break
        return ' '.join(tokens)

    def _clean_raw_query(self, text: str) -> str:
        """Clean the raw query for strength extraction: strip vendor codes, fix separators."""
        text = text.replace(",", " ").replace("/", " ").replace("-", " ")
        # Split stuck numbers: PAN40 -> PAN 40
        text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)
        text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
        # Remove trailing dots from numbers (650. -> 650)
        text = re.sub(r'(\d)\.(?!\d)', r'\1', text)
        tokens = text.split()
        # Strip leading pure-numeric vendor codes, same guard as normalize_text:
        # only strip if the next token is a real word (≥3 alpha chars).
        while tokens and re.match(r'^\d+$', tokens[0]) and len(tokens[0]) <= 3:
            if len(tokens) > 1 and sum(c.isalpha() for c in tokens[1]) >= 3:
                tokens.pop(0)
            else:
                break
        return ' '.join(tokens)

    def _has_unit_suffix(self, text: str, num: str) -> bool:
        """Check if a number in the text is followed by a unit suffix (mg, ml, etc.).
        If so, it's a strength, not a pack count."""
        pattern = re.escape(num) + r'\s*(?:mg|mcg|ml|gm|g|iu|units?)\b'
        return bool(re.search(pattern, text, re.IGNORECASE))

    def validate_match(self, query: str, candidate: dict, name: str = "", compname: str = "", strict: bool = False) -> bool:
        """Validate a proposed match. In strict mode (AI fallback), thresholds are higher."""
        q_clean = self.normalize_text(query)
        c_name_clean = self.normalize_text(candidate['name'])
        c_brand_clean = self.normalize_text(candidate['brand']) if candidate.get('brand') else ""
        
        q_words = q_clean.split()
        if not q_words:
            return False
        q_brand = q_words[0]
        UNIT_AND_FORMULATION = {
            'tab', 'tabs', 'tablet', 'tablets', 'cap', 'caps', 'capsule', 'capsules',
            'inj', 'injection', 'syp', 'syrup', 'susp', 'suspension', 'ml', 'mg', 'gm', 'g', 'mcg',
            'drop', 'drops', 'cream', 'oint', 'ointment', 'gel', 'lotion', 'vial', 'vials', 'amp', 'amps'
        }
        if re.match(r'^\d+$', q_brand) and len(q_words) > 1:
            next_w = q_words[1]
            alpha_count = sum(c.isalpha() for c in next_w)
            if alpha_count <= 2 and next_w not in UNIT_AND_FORMULATION:
                # Compound brand like "36 D" or "4 PH" — fuse into one token
                q_brand = q_brand + next_w
            elif alpha_count >= 3:
                # Leading number was a vendor code; real brand is the next word
                q_brand = next_w
        elif len(q_brand) <= 2 and len(q_words) > 1:
            next_word = q_words[1].lower()
            is_num = bool(re.match(r'^\d', next_word))
            if not (is_num or next_word in UNIT_AND_FORMULATION):
                q_brand = q_words[1]
            
        q_brand_flat = q_brand.replace(" ", "")
        c_name_flat = c_name_clean.replace(" ", "")
        c_brand_flat = c_brand_clean.replace(" ", "")
        
        # Check brand match — strict mode uses higher thresholds for AI fallback safety
        brand_cutoff = 88 if strict else 85
        min_substr_len = 4 if strict else 1
        
        is_abbrev_match = (
            is_abbreviation(q_brand, c_name_clean) or
            is_abbreviation(q_brand, c_brand_clean)
        )
        
        brand_match = (
            is_abbrev_match or
            (len(q_brand) >= min_substr_len and q_brand in c_name_clean) or 
            (len(q_brand) >= min_substr_len and q_brand in c_brand_clean) or 
            (len(q_brand_flat) >= min_substr_len and q_brand_flat in c_name_flat) or
            (len(c_name_flat) >= min_substr_len and c_name_flat in q_brand_flat) or
            (c_brand_flat and len(q_brand_flat) >= min_substr_len and (q_brand_flat in c_brand_flat or c_brand_flat in q_brand_flat)) or
            process.extractOne(q_brand, c_name_clean.split(), scorer=fuzz.ratio, score_cutoff=brand_cutoff) is not None or
            (c_brand_clean and process.extractOne(q_brand, c_brand_clean.split(), scorer=fuzz.ratio, score_cutoff=brand_cutoff) is not None)
        )
        # Fallback for vendor prefix codes (e.g. "RS-PAN 40" -> brand "PAN" is in query words)
        if not brand_match:
            c_first_word = c_name_clean.split()[0] if c_name_clean.split() else ""
            c_brand_first = c_brand_clean.split()[0] if c_brand_clean.split() else ""
            if (c_first_word and len(c_first_word) >= 2 and c_first_word.isalpha() and c_first_word in q_words) or (c_brand_first and len(c_brand_first) >= 2 and c_brand_first.isalpha() and c_brand_first in q_words):
                brand_match = True
            # In strict mode, also try the reverse: query first word in candidate
            if strict and not brand_match:
                q_first = q_words[0] if q_words else ""
                if q_first and len(q_first) >= 3 and q_first in c_name_clean:
                    brand_match = True

        if not brand_match:
            return False
            
        # Combo Suffix Validation Check
        COMBO_SUFFIXES = {
            'd', 'dsr', 'sr', 'xr', 'er', 'cr', 'mr', 'xl', 'la', 'plus', 'xt', 'tz', 'oz', 'h', 'at', 'am', 'lp', 'sp', 
            'ap', 'force', 'forte', 'fort', 'pd', 'kid', 'dc', 'as', 'sl', 'dt', 'ct', 'rt', 'mps', 'df', 'dx', 'ds', 
            'cv', 'lb', 'xp', 'f', 'e', 'k2', 'gold', 'mb', 'hs', 'at', 'aa', 'pr', 'od', 'fb', 'l', 'ax', 'snq', 'dp', 
            'th', 'pg', 'nt', 'm', 'beta', 'tri', 'ch', 'ln', 't', 'o', 'av', 'd3', 'p', 'rapid'
        }
        
        def get_suffixes(text: str) -> set:
            words = re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split()
            return {w for w in words if w in COMBO_SUFFIXES}
            
        q_suffixes = get_suffixes(name if name else query)
        c_suffixes = get_suffixes(candidate['name'])
        
        if q_suffixes != c_suffixes:
            return False

        # Smart Strength Validation Check
        def clean_nums(text: str, pack_text: str = "") -> set:
            text_lower = text.lower() + " " + pack_text.lower()
            # Split stuck number+letter (e.g. "125mg" -> "125 mg") to guarantee word boundaries
            text_lower = re.sub(r'([0-9])([a-z])', r'\1 \2', text_lower)
            text_lower = re.sub(r'([a-z])([0-9])', r'\1 \2', text_lower)
            text_clean = re.sub(r'\b\d+\s*(?:tab|tabs|tablet|tablets|cap|caps|capsule|capsules|vial|vials|amp|amps|ampoule|ampoules|sachet|sachets|pc|pcs|piece|pieces|strip|strips|pack|packs|packet|packets|bottle|bottles|bot|tube|tubes|pair|pairs|set|sets|roll|rolls|s)\b', ' ', text_lower)
            nums = re.findall(r'\b\d+(?:\.\d+)?\b', text_clean)
            normalized = set()
            for n in nums:
                try:
                    val = float(n)
                    # Ignore very common small values like 1 or 2 if they are not followed by units
                    if val in (1.0, 2.0) and not re.search(r'\b' + re.escape(n) + r'\s*(?:mg|ml|gm|g|mcg|iu|%)\b', text_lower):
                        continue
                    if val.is_integer():
                        normalized.add(str(int(val)))
                    else:
                        normalized.add(str(val))
                except ValueError:
                    normalized.add(n)
            return normalized

        q_strengths = clean_nums(name if name else query)
        c_strengths = clean_nums(candidate['name'], candidate.get('strength', '') + ' ' + candidate.get('pack', ''))
        
        if q_strengths and c_strengths:
            # ALL query strength components must appear in candidate.
            # issubset catches compound-dose mismatches like "500/125" vs "250/125" —
            # both share "125" but differ on "500" vs "250". A plain intersection would
            # pass; issubset correctly rejects because "500" is missing from {"250","125"}.
            if not q_strengths.issubset(c_strengths):
                return False

        # Comprehensive formulation compatibility check
        if not formulations_compatible(query, candidate['name'] + " " + candidate['pack']):
            return False

        # Smart Company Compatibility check (always checked if input company is provided)
        if compname and compname.strip() and compname.strip() not in ('--', 'none', 'null', 'nan'):
            c_brand = candidate.get('brand', '') or ''
            if c_brand and not companies_compatible(compname, c_brand):
                return False

        return True

    def heuristic_match(self, query: str, candidates: list, name: str = "", compname: str = "") -> tuple:
        """Try to find a match using heuristic rules.
        Returns (match_dict, confidence_score) or (None, 0).
        confidence_score is 0-100 based on brand similarity."""
        if not candidates:
            return None, 0
            
        for cand in candidates:
            q_clean = self.normalize_text(query)
            c_name_clean = self.normalize_text(cand['name'])
            c_brand_clean = self.normalize_text(cand['brand']) if cand.get('brand') else ""
            
            strength_query = self._clean_raw_query(name if name else query)
            q_nums = re.findall(r'\d+\.?\d*', strength_query)
            c_nums = re.findall(r'\d+\.?\d*', cand['name'] + " " + cand['strength'])
            
            q_words = q_clean.split()
            if not q_words:
                continue
            q_brand = q_words[0]
            if re.match(r'^\d+$', q_brand) and len(q_words) > 1:
                next_w = q_words[1]
                alpha_count = sum(c.isalpha() for c in next_w)
                if alpha_count <= 2 and next_w.isalpha():
                    q_brand = q_brand + next_w  # compound brand: "36d", "4ph"
                else:
                    q_brand = next_w             # vendor code, use real brand word
            elif len(q_brand) <= 2 and len(q_words) > 1:
                q_brand = q_words[1]

            c_words = c_name_clean.split()
            c_brand_words = c_brand_clean.split()

            brand_exact = False
            if c_words:
                UNIT_SUFFIXES = {'mg', 'ml', 'mcg', 'gm', 'g', 'cap', 'tab', 'tabs', 'caps'}
                # Detect compound num+letter brand in candidate (e.g. master "36 D CAPS" → "36d")
                c_compound_brand = ""
                if (re.match(r'^\d+$', c_words[0]) and len(c_words) > 1
                        and len(c_words[1]) <= 2 and c_words[1].isalpha()):
                    c_compound_brand = c_words[0] + c_words[1]
                c_brand_words_no_num = [w for w in c_words if not w.isdigit() and w not in UNIT_SUFFIXES]
                c_brand_flat = "".join(c_brand_words_no_num[:2]) if len(c_brand_words_no_num) >= 2 else "".join(c_brand_words_no_num)
                
                q_brand_words_no_num = [w for w in q_words if not w.isdigit() and w not in UNIT_SUFFIXES]
                q_brand_flat = "".join(q_brand_words_no_num[:2]) if len(q_brand_words_no_num) >= 2 else "".join(q_brand_words_no_num)
                
                def _brands_similar(q_flat: str, c_flat: str) -> bool:
                    if q_flat == c_flat:
                        return True
                    if fuzz.ratio(q_flat, c_flat) >= 78:
                        return True
                    # partial_ratio only if both brands are long enough (>=5 chars)
                    # and with a stricter threshold to prevent cross-brand leaks
                    if len(q_flat) >= 5 and len(c_flat) >= 5 and fuzz.partial_ratio(q_flat, c_flat) >= 90:
                        return True
                    return False

                if (_brands_similar(q_brand_flat, c_brand_flat)
                        or (c_brand_words_no_num and _brands_similar(q_brand_flat, c_brand_words_no_num[0]))
                        or (c_compound_brand and _brands_similar(q_brand_flat, c_compound_brand))):
                    brand_exact = True
                elif c_brand_words:
                    c_brand_words2_no_num = [w for w in c_brand_words if not w.isdigit() and w not in UNIT_SUFFIXES]
                    c_brand_flat2 = "".join(c_brand_words2_no_num[:2]) if len(c_brand_words2_no_num) >= 2 else "".join(c_brand_words2_no_num)
                    # Skip company brand check if flat string is too short (e.g. '3M' -> 'm')
                    if len(c_brand_flat2) >= 3 and (_brands_similar(q_brand_flat, c_brand_flat2) or (c_brand_words2_no_num and len(c_brand_words2_no_num[0]) >= 3 and _brands_similar(q_brand_flat, c_brand_words2_no_num[0]))):
                        brand_exact = True
                        
            # Fallback for vendor prefix codes (e.g. "RS-PAN 40" -> brand "PAN" is in query words)
            if not brand_exact:
                c_first_word = c_words[0] if c_words else ""
                c_brand_first = c_brand_words[0] if c_brand_words else ""
                if (c_first_word and len(c_first_word) >= 2 and c_first_word.isalpha() and c_first_word in q_words) or (c_brand_first and len(c_brand_first) >= 2 and c_brand_first.isalpha() and c_brand_first in q_words):
                    brand_exact = True
                        
            # Strength matching — numbers with unit suffixes are ALWAYS strengths
            PACK_COUNT_NUMS = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}
            raw_q = name if name else query
            raw_c = cand['name'] + ' ' + cand['strength']
            nums_match = False
            if q_nums:
                matched_num = False
                for q_num in q_nums:
                    if q_num in PACK_COUNT_NUMS and '.' not in q_num and not self._has_unit_suffix(raw_q, q_num):
                        continue
                    if q_num in c_nums:
                        matched_num = True
                        break
                q_strengths = [n for n in q_nums if n not in PACK_COUNT_NUMS or '.' in n or self._has_unit_suffix(raw_q, n)]
                if q_strengths:
                    if matched_num:
                        nums_match = True
                else:
                    c_strengths = [n for n in c_nums if n not in PACK_COUNT_NUMS or '.' in n or self._has_unit_suffix(raw_c, n)]
                    if not c_strengths:
                        nums_match = True
            else:
                c_strengths = [n for n in c_nums if n not in PACK_COUNT_NUMS or '.' in n or self._has_unit_suffix(raw_c, n)]
                if not c_strengths:
                    nums_match = True

            # Comprehensive formulation compatibility check
            formulation_match = formulations_compatible(query, cand['name'] + " " + cand['pack'])

            if brand_exact and nums_match and formulation_match:
                if self.validate_match(query, cand, name=name, compname=compname):
                    # Calculate confidence score from brand similarity
                    brand_score = fuzz.ratio(q_brand_flat, c_brand_flat)
                    if c_brand_words_no_num:
                        word_score = fuzz.ratio(q_brand_flat, c_brand_words_no_num[0])
                        brand_score = max(brand_score, word_score)
                    
                    # Company mismatch penalty on confidence
                    if compname and compname.strip() and compname.strip() != '--':
                        c_company = self.normalize_text(cand.get('brand', '') or '')
                        q_comp_norm = self.normalize_text(compname)
                        if q_comp_norm and c_company:
                            comp_sim = fuzz.partial_ratio(q_comp_norm, c_company)
                            if comp_sim < 30:
                                brand_score = max(0, brand_score - 20)
                    
                    return cand, brand_score
        return None, 0

    def llm_rerank(self, query: str, candidates: list, name: str = "", compname: str = "", engine=None) -> dict:
        """AI fallback: ask the local LLM to pick the correct match."""
        if engine is None:
            global _llm_engines
            engine = _llm_engines.get(self.model_key)
            if engine is None:
                from qwen3_engine.vllm_engine import VLLMEngine
                probe = VLLMEngine()
                if probe.check_ready(timeout=2.0):
                    print(f"[AI] Using VLLMEngine (llama_cpp.server on port 8000) for fallback matching...")
                    engine = VLLMEngine()
                    engine._ready = True
                    _llm_engines[self.model_key] = engine
                else:
                    try:
                        from qwen3_engine.engine import Qwen3Engine
                        from qwen3_engine.config import N_CTX_MATCHER
                        engine = Qwen3Engine(model_key=self.model_key)
                        print(f"[AI] Loading local {self.model_key} LLM for fallback matching...")
                        engine.load(n_ctx=N_CTX_MATCHER if 'N_CTX_MATCHER' in dir() else 2048)
                        print(f"[AI] ✓ LLM {self.model_key} loaded successfully.")
                        _llm_engines[self.model_key] = engine
                    except Exception as e:
                        print(f"[AI] ⚠ Could not load LLM {self.model_key}: {e}")
                        return None

        if not engine or not engine.is_loaded():
            return None

        # Set system prompt unconditionally
        if self.model_key == "qwen3":
            engine.system_prompt = (
                "You are a strict data-matching engine. Match the wholesaler input to a candidate number (1-5) or return null. Respond ONLY with a JSON block, no explanations."
            )
        else:
            engine.system_prompt = (
                "You are a strict, deterministic medical data-matching engine.\n"
                "Match a 'Wholesaler Input' to a candidate from the 'Master Candidates List'.\n\n"
                "RULES (ALL must be satisfied):\n"
                "1. BRAND must match (typos OK: 'PNTOP'='PANTOP'). Different brands = null.\n"
                "2. FORMULATION/PACK must match. Oral solids (TAB, CAP, S10, STRIP, 10TAB, 15TAB, etc.) are compatible. Incompatible formulations are mismatches.\n"
                "3. STRENGTH must match. 40≠400, 650≠500.\n"
                "4. Respond ONLY with a JSON block in the format: {\"match_number\": X} or {\"match_number\": null}. Do not write any analysis or explanations."
            )

        if self.model_key == "qwen3":
            # Simple, short prompt for Qwen3 0.6B to maximize speed and efficiency
            prompt = (
                "Wholesaler Input: PAN 40 TAB\n"
                "Master Candidates List:\n"
                "1. PAN 40 (Pack: 10 TAB, code: 101)\n"
                "2. PENTAB 40 (Pack: 1 PC, code: 102)\n"
                "Result:\n"
                "```json\n"
                "{\"match_number\": 1}\n"
                "```\n\n"
                "Wholesaler Input: CALPOL 650\n"
                "Master Candidates List:\n"
                "1. CALPOL 500 (Pack: 15 TAB, code: 201)\n"
                "2. DOLO 650 (Pack: 15 TAB, code: 202)\n"
                "Result:\n"
                "```json\n"
                "{\"match_number\": null}\n"
                "```\n\n"
                f"Wholesaler Input: {query}\n"
                "Master Candidates List:\n"
            )
            for i, cand in enumerate(candidates, 1):
                prompt += f"{i}. {cand['name']} (Pack: {cand['pack']}, code: {cand['code']})\n"
            prompt += "\nResult:"
        else:
            # Trimmed chain-of-thought (matches validated matcher.py version):
            # 1 example, no 'code' field shown to the model (code isn't needed
            # for its decision — only used internally via list position).
            prompt = (
                "Wholesaler Input: CALPOL 650 (Co: GSK)\n"
                "Candidates:\n"
                "1. CALPOL 500 (Pack: 15 TAB, Co: GSK)\n"
                "2. CALPOL 650 (Pack: 15 TAB, Co: GSK)\n"
                "Analysis: Cand 1 strength mismatch (500 vs 650). Cand 2 matches all. Match.\n"
                "Result:\n```json\n{\"match_number\": 2}\n```\n\n"
                f"Wholesaler Input: {query}"
            )
            if compname and compname.strip() and compname.strip() != '--':
                prompt += f" (Co: {compname.strip()})"
            prompt += "\nCandidates:\n"
            for i, cand in enumerate(candidates, 1):
                cand_company = cand.get('brand', '') or ''
                prompt += f"{i}. {cand['name']} (Pack: {cand['pack']}, Co: {cand_company})\n"

            prompt += (
                "\nWrite 'Analysis:' with brief step-by-step comparisons, then 'Result:' with the final match JSON block.\n\n"
                f"Analysis:\n"
                f"- Wholesaler input is '{query}'."
            )
        engine.clear_history()
        raw_response = engine.generate(prompt, max_tokens=700, temperature=0.0)
        if self.model_key != "qwen3":
            response = f"- Wholesaler input is '{query}'." + raw_response.strip()
        else:
            response = raw_response.strip()
        print(f"[AI] LLM Output:\n{response}")

        try:
            match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if not match:
                match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group(1) if len(match.groups()) > 0 else match.group(0))
                match_num = data.get("match_number")
                if match_num is not None and isinstance(match_num, int):
                    match_idx = match_num - 1
                    if 0 <= match_idx < len(candidates):
                        cand = candidates[match_idx]
                        if self.validate_match(query, cand, name=name, compname=compname, strict=True):
                            return cand
                        else:
                            print(f"[AI] Guardrail blocked: {cand['name']}")
        except Exception as e:
            print(f"[AI] Parse error: {e}")
        return None

    def match(self, query: str, name: str = "", pack: str = "", compname: str = "", engine=None) -> dict:
        search_query = name if name else query
        _t_search = time.time()
        results, _ = self.searcher.search(search_query, pack=pack, compname=compname, top_k=5)

        # Second-pass: if first search returned fewer than 4 results, retry with
        # just the first substantive word (catches abbreviations like "AMC" → "AMOXICLAV")
        if len(results) < 4:
            words = [w for w in re.sub(r'[^a-zA-Z0-9\s]', ' ', search_query).split()
                     if len(w) >= 3 and not w.isdigit()]
            if words:
                extra, _ = self.searcher.search(words[0], pack=pack, compname=compname, top_k=5)
                seen = {r["code"] for r in results}
                for r in extra:
                    if r["code"] not in seen:
                        results.append(r)
                        seen.add(r["code"])
                results = results[:8]
        _search_ms = (time.time() - _t_search) * 1000.0

        candidates = []
        for r in results:
            candidates.append({
                "code": r["code"],
                "name": r["name"],
                "brand": r["compname"],
                "pack": r["pack"],
                "strength": r["strength"],
            })

        # Stage 1: Heuristic (fast, handles 85%+ correctly)
        _t_heur = time.time()
        heur_match, confidence = self.heuristic_match(query, candidates, name=name, compname=compname)
        _heur_ms = (time.time() - _t_heur) * 1000.0

        if heur_match and confidence >= 95:
            # HIGH confidence → accept directly
            _record_timing(_search_ms, _heur_ms, 0.0, llm=False)
            return heur_match

        # Medium/no-confidence heuristic matches go straight to AI fallback.
        # (AI Verify used to run first here, but it rejected ~99% of medium-
        # confidence matches and fallback re-checks the same candidate anyway
        # with a stricter, guardrail-backed prompt — verify was pure overhead.)

        # Stage 2: AI fallback (slow but accurate for edge cases)
        if candidates:
            print(f"[AI Fallback] Heuristic failed for '{query}'. Asking LLM...")
            _t_llm = time.time()
            _inflight_enter()
            try:
                ai_match = self.llm_rerank(query, candidates, name=name, compname=compname, engine=engine)
            finally:
                _inflight_exit()
            _llm_ms = (time.time() - _t_llm) * 1000.0
            _record_timing(_search_ms, _heur_ms, _llm_ms, llm=True)
            if ai_match:
                return ai_match
            return None

        _record_timing(_search_ms, _heur_ms, 0.0, llm=False)
        return None

_matcher = None
_engine_pool = None   # pre-loaded at startup; None until ready

# A single big batch landing on ONE gunicorn process can only use that
# process's threads — and Python's GIL means those threads can't truly run
# the CPU-bound heuristic matching in parallel. Splitting a large batch into
# sub-chunks and firing them at our own /match endpoint lets gunicorn's
# shared listen socket spread them across ALL worker processes instead,
# giving genuine multi-core parallelism. Sub-chunk requests carry
# "_internal_subchunk" so they're processed directly, not split again.
SELF_SPLIT_THRESHOLD  = 150
SELF_SPLIT_CHUNK_SIZE = 150


def _dispatch_self_split(data: dict, products: list):
    """Split a large batch across this server's own worker processes."""
    backend_url   = (data.get("backend_url") or "").rstrip("/")
    job_id        = (data.get("job_id") or "").strip()
    signal_secret = (data.get("signal_secret") or "").strip()
    progress_url  = (
        f"{backend_url}/medicine-matching/admin/gpu-match/{job_id}/gpu-progress"
        if backend_url and job_id and signal_secret else None
    )

    sub_chunks = [products[i:i + SELF_SPLIT_CHUNK_SIZE]
                  for i in range(0, len(products), SELF_SPLIT_CHUNK_SIZE)]
    total   = len(products)
    t_start = time.time()

    _state       = {"processed": 0, "matched": 0}
    _done_event  = threading.Event()
    counter_lock = threading.Lock()

    def _http_push(processed, matched):
        if not progress_url:
            return
        elapsed = time.time() - t_start
        rate    = processed / elapsed if elapsed > 0 else 0
        eta_s   = (total - processed) / rate if rate > 0 else 0
        try:
            requests.post(
                progress_url,
                json={
                    "job_id": job_id, "secret": signal_secret,
                    "processed": processed, "total": total, "matched": matched,
                    "eta_seconds": round(eta_s, 1),
                },
                timeout=5,
            )
        except Exception as _e:
            print(f"[Progress] HTTP push failed: {_e}")

    def _reporter_thread():
        INTERVAL = 30
        while not _done_event.wait(INTERVAL):
            _http_push(_state["processed"], _state["matched"])
        _http_push(_state["processed"], _state["matched"])

    reporter = threading.Thread(target=_reporter_thread, daemon=True)
    reporter.start()

    all_mappings  = []
    all_unmatched = []
    errors        = []

    def _send_sub_chunk(sub_products):
        sub_payload = dict(data)
        sub_payload["products"] = sub_products
        sub_payload["_internal_subchunk"] = True
        try:
            # 300s timeout: 150 items @ 5.8 items/sec = 26s processing + queue time
            resp = requests.post("http://127.0.0.1:8080/match", json=sub_payload, timeout=300)
            resp.raise_for_status()
            result = resp.json()
        except Exception as exc:
            errors.append(str(exc))
            return
        with counter_lock:
            all_mappings.extend(result.get("mappings", []))
            all_unmatched.extend(result.get("unmatched", []))
            _state["processed"] += len(sub_products)
            _state["matched"]    = len(all_mappings)

    try:
        # Fire sub-chunks in waves matching gunicorn worker count (10) so every
        # worker stays busy. Timeout (300s) ensures stuck requests fail fast.
        max_workers = min(10, len(sub_chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_send_sub_chunk, sub_chunks))
    finally:
        _done_event.set()
        reporter.join(timeout=10)

    if errors:
        print(f"[Self-Split] {len(errors)}/{len(sub_chunks)} sub-chunks failed: {errors[:3]}")

    return jsonify({"mappings": all_mappings, "unmatched": all_unmatched})


@app.route("/match", methods=["POST"])
def match_item():
    data = request.get_json(silent=True) or {}

    if "products" in data:
        # ── Batch match mode ──────────────────────────────────────────────────
        products      = data.get("products", [])

        if not data.get("_internal_subchunk") and len(products) > SELF_SPLIT_THRESHOLD:
            return _dispatch_self_split(data, products)

        # Progress callback info (injected by backend)
        backend_url   = (data.get("backend_url") or "").rstrip("/")
        job_id        = (data.get("job_id") or "").strip()
        signal_secret = (data.get("signal_secret") or "").strip()
        progress_url  = (
            f"{backend_url}/medicine-matching/admin/gpu-match/{job_id}/gpu-progress"
            if backend_url and job_id and signal_secret else None
        )

        # ── Use pre-loaded global searcher and engine pool ──────────────────
        if _searcher is None or _matcher is None:
            return jsonify({"error": "Server still initialising — searchers not ready."}), 503

        # Wait up to 180 seconds for engine pool to finish populating in the background
        wait_elapsed = 0
        while _engine_pool is None and wait_elapsed < 180:
            time.sleep(0.5)
            wait_elapsed += 0.5

        if _engine_pool is None:
            return jsonify({"error": "Server still initialising — pool not ready."}), 503

        temp_matcher = _matcher

        pool_size  = _engine_pool.size
        engine_pool = _engine_pool

        mappings  = []   # matched items
        unmatched = []   # products with no master match
        total     = len(products)
        t_start   = time.time()

        # ── Shared state for the daemon reporter thread ───────────────────
        _state       = {"processed": 0, "matched": 0}
        _done_event  = threading.Event()
        counter_lock = threading.Lock()

        def _http_push(processed, matched):
            """Fire a single HTTP POST to the backend progress endpoint."""
            if not progress_url:
                return
            import urllib.request as _req, json as _json
            elapsed = time.time() - t_start
            rate    = processed / elapsed if elapsed > 0 else 0
            eta_s   = (total - processed) / rate if rate > 0 else 0
            body = _json.dumps({
                "job_id":      job_id,
                "secret":      signal_secret,
                "processed":   processed,
                "total":       total,
                "matched":     matched,
                "eta_seconds": round(eta_s, 1),
            }).encode()
            try:
                req = _req.Request(
                    progress_url, data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with _req.urlopen(req, timeout=5):
                    pass
            except Exception as _e:
                print(f"[Progress] HTTP push failed: {_e}")

        def _reporter_thread():
            """
            Daemon thread: fires a progress callback every 30 s.
            Never touches the match loop — reads shared counters only.
            Wakes early when _done_event is set (job finished/failed).
            """
            INTERVAL = 30
            while not _done_event.wait(INTERVAL):
                _http_push(_state["processed"], _state["matched"])
            _http_push(_state["processed"], _state["matched"])

        reporter = threading.Thread(target=_reporter_thread, daemon=True)
        reporter.start()

        def process_product(prod):
            prod_name = prod.get("name", "").strip()
            prod_pack = prod.get("pack", "").strip()
            prod_comp = prod.get("company", "").strip()
            prod_code = prod.get("code", "").strip()

            try:
                engine = engine_pool.lease()
                try:
                    res_match = temp_matcher.match(
                        f"{prod_name} {prod_pack}".strip(),
                        name=prod_name, pack=prod_pack, compname=prod_comp,
                        engine=engine
                    )
                finally:
                    engine_pool.release(engine)

                if res_match:
                    conf = compute_confidence(
                        prod_name,
                        res_match['name'],
                        res_match.get('brand', ''),
                        prod_pack,
                        res_match.get('pack', '')
                    )
                    mappings.append({
                        "vendor_code":      prod_code,
                        "master_id":        res_match["code"],
                        "product_name":     prod_name,
                        "company":          prod_comp,
                        "pack":             prod_pack,
                        "matched_name":     res_match["name"],
                        "matched_pack":     res_match.get("pack", ""),
                        "matched_company":  res_match.get("brand", ""),
                        "source":           "gpu_ai",
                        "confidence":       float(conf)
                    })
                else:
                    cand_results, _ = _searcher.search(
                        f"{prod_name} {prod_pack}".strip(), pack=prod_pack, compname=prod_comp, top_k=5
                    )
                    unmatched.append({
                        "vendor_code":  prod_code,
                        "product_name": prod_name,
                        "company":      prod_comp,
                        "pack":         prod_pack,
                        "candidates": [
                            {
                                "master_id": r["code"],
                                "name": r["name"],
                                "company": r.get("compname", ""),
                                "pack": r.get("pack", ""),
                                "score": r.get("final_score", 0),
                            }
                            for r in cand_results
                        ],
                    })
            except Exception as _proc_exc:
                print(f"[ProcessError] '{prod_name} {prod_pack}': {type(_proc_exc).__name__}: {_proc_exc}")
                unmatched.append({
                    "vendor_code":  prod_code,
                    "product_name": prod_name,
                    "company":      prod_comp,
                    "pack":         prod_pack,
                    "candidates": [],
                    "error": f"{type(_proc_exc).__name__}: {_proc_exc}",
                })

            with counter_lock:
                _state["processed"] += 1
                _state["matched"]    = len(mappings)

        # ── Main matching loop (Parallel ThreadPoolExecutor) ──────────────
        try:
            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                executor.map(process_product, products)
        finally:
            _done_event.set()
            reporter.join(timeout=10)

        return jsonify({"mappings": mappings, "unmatched": unmatched})

    # Single match mode (legacy/original compatibility)
    name     = data.get("name", "").strip()
    pack     = data.get("pack", "").strip()
    compname = data.get("company", "").strip() or data.get("compname", "").strip()
    query    = data.get("query", "").strip()

    if not query:
        query = f"{name} {pack}".strip()

    if not query:
        return jsonify({"error": "Field 'query' or 'name' is required."}), 400

    t0 = time.time()
    result = _matcher.match(query, name=name, pack=pack, compname=compname)
    elapsed = time.time() - t0

    return jsonify({
        "query": query,
        "match": result,
        "latency_ms": round(elapsed * 1000, 2)
    })

@app.route("/search", methods=["POST"])
def search_item():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Field 'name' is required."}), 400
    pack = data.get("pack", "").strip()
    top_k = int(data.get("top_k", 5))
    
    results, elapsed_ms = _searcher.search(name, pack=pack, top_k=top_k)
    return jsonify({
        "query": {"name": name, "pack": pack},
        "results": results,
        "count": len(results),
        "search_ms": elapsed_ms,
    })

@app.route("/health", methods=["GET"])
def health():
    if _searcher is None or _matcher is None:
        return jsonify({"status": "initialising"}), 503
    pool_size = _engine_pool.size if _engine_pool else 0
    if pool_size == 0:
        # Pool still loading — tell startup script to keep waiting
        return jsonify({"status": "initialising", "mode": "standalone_fast", "workers": 0}), 503
    return jsonify({"status": "ok", "mode": "standalone_fast", "workers": pool_size})


def get_gce_metadata(attribute_name: str) -> str:
    url = f"http://metadata.google.internal/computeMetadata/v1/instance/attributes/{attribute_name}"
    import urllib.request as _req
    req = _req.Request(url)
    req.add_header("Metadata-Flavor", "Google")
    try:
        with _req.urlopen(req, timeout=2) as resp:
            return resp.read().decode('utf-8').strip()
    except Exception:
        return ""

def initialize_on_import():
    global _searcher, _matcher
    # Fetch parameters from environment or GCE metadata
    backend_url = os.environ.get("BACKEND_URL") or get_gce_metadata("backend_url")
    secret = os.environ.get("GPU_SIGNAL_SECRET") or get_gce_metadata("gpu_signal_secret")

    catalog_loaded = False
    if backend_url and secret:
        import urllib.request as _req, json as _json, tempfile, csv
        catalog_url = f"{backend_url.rstrip('/')}/medicine-matching/admin/gpu-match/catalog?secret={secret}"
        print(f"[Startup] Fetching master medicines from backend: {catalog_url}...")
        try:
            with _req.urlopen(catalog_url, timeout=120) as resp:
                raw = _json.loads(resp.read())
            medicines = raw.get("catalog", [])
            print(f"[Startup] Received {len(medicines):,} master medicines from backend.")

            # Write to a temp CSV so FuzzySearcher can load it
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
            try:
                with open(tmp_fd, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    fieldnames = ['code', 'name', 'Compname', 'Pack', 'strength']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for m in medicines:
                        name_str = (m.get("name") or "").strip()
                        writer.writerow({
                            'code':     m.get("master_id") or m.get("code") or "",
                            'name':     name_str,
                            'Compname': (m.get("company") or "").strip(),
                            'Pack':     (m.get("pack") or "").strip(),
                            'strength': extract_strength(name_str),
                        })
                print(f"[Startup] Building FuzzySearcher index from {len(medicines):,} medicines...")
                _searcher = FuzzySearcher(csv_path=tmp_path)
                _searcher.load()
                catalog_loaded = True
            finally:
                import os as _os
                try: _os.unlink(tmp_path)
                except Exception: pass
        except Exception as e:
            print(f"[Startup] WARN: Catalog fetch failed ({e}). Falling back to local CSV...")

    if not catalog_loaded:
        # Fallback to local CSV
        user_home = os.path.expanduser("~")
        candidate_paths = [
            os.path.join(BASE_DIR, "Item_export_2026-05-29_17-33-30.csv"),
            os.path.join(os.path.dirname(BASE_DIR), "Item_export_2026-05-29_17-33-30.csv"),
            os.path.join(user_home, "Downloads", "Item_export_2026-05-29_17-33-30.csv"),
            os.path.join(user_home, "Downloads", "item_export_2026-05-30_09-08-22.csv"),
            os.path.join(os.path.dirname(sys.executable), "Item_export_2026-05-29_17-33-30.csv"),
            os.path.join(os.getcwd(), "Item_export_2026-05-29_17-33-30.csv"),
        ]
        csv_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                csv_path = p
                break
        if not csv_path or not os.path.exists(csv_path):
            print(f"Error: Unified CSV database not found. Pass BACKEND_URL and GPU_SIGNAL_SECRET env variables.")
            sys.exit(1)

        print(f"Loading search database from: {csv_path}...")
        _searcher = FuzzySearcher(csv_path=csv_path)
        _searcher.load()

    _matcher = FastMatcher(_searcher)
    _matcher.load_catalog()

    # Under gunicorn --preload the app (and this catalog load) runs ONCE in
    # the master before forking, so all workers inherit it via copy-on-write
    # instead of each of the N workers redundantly re-fetching/re-indexing
    # the full catalog themselves — that redundant work was the dominant
    # cost in pod boot time. The engine pool holds live HTTP connections to
    # llama-server though — those don't survive being copied into a forked
    # child — so it must be created AFTER fork, once per worker.
    # FAST_SERVER_DEFER_POOL_INIT=1 signals that a gunicorn `post_fork` hook
    # (see qwen3_engine/gunicorn_conf.py) will call
    # init_engine_pool_for_worker() itself instead of doing it here.
    if os.environ.get("FAST_SERVER_DEFER_POOL_INIT") != "1":
        import threading as _threading
        _threading.Thread(target=init_engine_pool_for_worker, daemon=True).start()

def init_engine_pool_for_worker():
    """
    Creates and populates this process's EnginePool. Must run once per
    gunicorn worker AFTER fork — the pool holds live HTTP connections to
    llama-server, which do not survive being copied into a forked child.
    Runs automatically at import when not deferred (see initialize_on_import
    above), or via a gunicorn `post_fork` hook when FAST_SERVER_DEFER_POOL_INIT
    is set (see qwen3_engine/gunicorn_conf.py).
    """
    global _engine_pool
    from qwen3_engine.config import N_CTX_MATCHER
    # Pool size = concurrent requests to llama-server. Each pool slot is a
    # lightweight HTTP client connection (no local model weights), so size
    # it to the server's actual concurrency limit (--parallel N) rather
    # than a hardcoded guess — reading /props keeps this correct even if
    # --parallel changes.
    pool_size = 30
    try:
        import os as _os2, requests as _rq
        from qwen3_engine.config import LLAMA_SERVER_BASE_URL as _base
        _urls = [u.strip() for u in _os2.environ.get("LLAMA_BACKEND_URLS", _base).split(",") if u.strip()]
        _total_slots = 0
        for _u in _urls:
            _props_url = _u.rstrip("/").removesuffix("/v1") + "/props"
            try:
                _s = int(_rq.get(_props_url, timeout=3).json().get("total_slots") or 0)
                _total_slots += _s
            except Exception:
                pass
        if _total_slots > 0:
            pool_size = _total_slots
    except Exception:
        pass
    # Running under gunicorn with N worker PROCESSES: each process gets
    # its own independent pool, so sizing every one to the full slot count
    # oversubscribes llama-server by Nx (measured: 10 workers x 32 each =
    # 320 connections fighting over 32 real slots -> timeouts, dropped
    # items, throughput collapse). Divide the total capacity across
    # workers instead so the combined pool size matches llama-server's
    # real concurrency limit.
    worker_count = int(os.environ.get("GUNICORN_WORKER_COUNT", "1") or "1")
    if worker_count > 1:
        divided = max(1, pool_size // worker_count)
        print(f"[Startup] Gunicorn worker count={worker_count}: dividing pool "
              f"{pool_size} -> {divided} per worker ({divided * worker_count} total).")
        pool_size = divided
    print(f"[Startup] Sizing engine pool to {pool_size} workers (llama-server slots).")
    pool = EnginePool(model_key="gemma4_2b", size=pool_size)
    pool.populate()
    _engine_pool = pool

# Initialize automatically on import if NOT running via CLI command-line tool with arguments
import sys
if not (len(sys.argv) > 1 and ("--backend-url" in sys.argv or "--csv" in sys.argv)):
    initialize_on_import()

def main():
    global _searcher, _matcher
    import argparse

    parser = argparse.ArgumentParser(description="Standalone REST API Server")
    parser.add_argument("--csv",         type=str, default="", help="Path to catalog CSV (legacy fallback)")
    parser.add_argument("--backend-url", type=str, default="", help="Backend base URL to fetch master medicines from")
    parser.add_argument("--secret",      type=str, default="", help="GPU_SIGNAL_SECRET shared with backend")
    args = parser.parse_args()

    # ── Option A: fetch master medicines from backend API ─────────────────────
    if args.backend_url and args.secret:
        import urllib.request as _req, json as _json, tempfile, csv
        catalog_url = f"{args.backend_url.rstrip('/')}/medicine-matching/admin/gpu-match/catalog?secret={args.secret}"
        print(f"[Startup] Fetching master medicines from backend: {catalog_url}...")
        try:
            with _req.urlopen(catalog_url, timeout=120) as resp:
                raw = _json.loads(resp.read())
            medicines = raw.get("catalog", [])
            print(f"[Startup] Received {len(medicines):,} master medicines from backend.")
        except Exception as e:
            print(f"[Startup] ERROR: Failed to fetch catalog from backend: {e}")
            sys.exit(1)

        # Write to a temp CSV so FuzzySearcher can load it
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
        try:
            with open(tmp_fd, 'w', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = ['code', 'name', 'Compname', 'Pack', 'strength']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for m in medicines:
                    name_str = (m.get("name") or "").strip()
                    writer.writerow({
                        'code':     m.get("master_id") or m.get("code") or "",
                        'name':     name_str,
                        'Compname': (m.get("company") or "").strip(),
                        'Pack':     (m.get("pack") or "").strip(),
                        'strength': extract_strength(name_str),
                    })
            print(f"[Startup] Building FuzzySearcher index from {len(medicines):,} medicines...")
            _searcher = FuzzySearcher(csv_path=tmp_path)
            _searcher.load()
        finally:
            import os as _os
            try: _os.unlink(tmp_path)
            except Exception: pass

    # ── Option B: legacy CSV path ─────────────────────────────────────────────
    else:
        csv_path = args.csv
        if not csv_path:
            user_home = os.path.expanduser("~")
            candidate_paths = [
                os.path.join(BASE_DIR, "Item_export_2026-05-29_17-33-30.csv"),
                os.path.join(os.path.dirname(BASE_DIR), "Item_export_2026-05-29_17-33-30.csv"),
                os.path.join(user_home, "Downloads", "Item_export_2026-05-29_17-33-30.csv"),
                os.path.join(user_home, "Downloads", "item_export_2026-05-30_09-08-22.csv"),
                os.path.join(os.path.dirname(sys.executable), "Item_export_2026-05-29_17-33-30.csv"),
                os.path.join(os.getcwd(), "Item_export_2026-05-29_17-33-30.csv"),
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    csv_path = p
                    break

        if not csv_path or not os.path.exists(csv_path):
            print(f"Error: Unified CSV database not found. Pass --backend-url and --secret to fetch from backend.")
            sys.exit(1)

        print(f"Loading search database from: {csv_path}...")
        _searcher = FuzzySearcher(csv_path=csv_path)
        _searcher.load()

    _matcher = FastMatcher(_searcher)
    _matcher.load_catalog()

    # Check if AI model is available
    from qwen3_engine.config import MODEL_REGISTRY, DEFAULT_MODEL
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    model_file = os.path.join(model_dir, MODEL_REGISTRY[DEFAULT_MODEL]["filename"])
    if os.path.exists(model_file):
        print("✓ AI model found — LLM fallback enabled.")
    else:
        print("⚠ AI model not found — running heuristic-only mode.")

    # ── Pre-load engine pool in background so health returns 503 until ready ──
    def _preload_pool():
        global _engine_pool
        from qwen3_engine.config import N_CTX_MATCHER, MODEL_REGISTRY, DEFAULT_MODEL
        pool_size = calculate_optimal_workers("gemma4_2b")
        # Native llama-server mode: pool entries are lightweight HTTP clients,
        # not local model instances — size the pool to the server's slot count
        # so every GPU slot can stay busy. The VRAM formula above only applies
        # when loading local instances.
        try:
            import os as _os2, requests as _rq
            from qwen3_engine.config import LLAMA_SERVER_BASE_URL as _base
            _urls = [u.strip() for u in _os2.environ.get("LLAMA_BACKEND_URLS", _base).split(",") if u.strip()]
            _total_slots = 0
            for _u in _urls:
                _props_url = _u.rstrip("/").removesuffix("/v1") + "/props"
                try:
                    _s = int(_rq.get(_props_url, timeout=3).json().get("total_slots") or 0)
                    _total_slots += _s
                except Exception:
                    pass
            if _total_slots > 0:
                pool_size = _total_slots
        except Exception:
            pass
        print(f"[Startup] Pre-loading engine pool: {pool_size} workers (n_ctx={N_CTX_MATCHER})...")

        # ── Pre-warm: read model file into OS RAM cache once ──
        # This makes all workers load from RAM (1.6s each) instead of cold disk (137s for first worker)
        if os.path.exists(model_file):
            model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"[Startup] Pre-warming disk cache: reading {model_size_mb:.0f}MB model into RAM...")
            t_warm = time.time()
            with open(model_file, 'rb') as f:
                while f.read(64 * 1024 * 1024):  # Read in 64MB chunks
                    pass
            print(f"[Startup] Disk cache warm in {time.time()-t_warm:.1f}s — all workers will load from RAM.")

        pool = EnginePool(model_key="gemma4_2b", size=pool_size)
        pool.populate()
        _engine_pool = pool
        print(f"[Startup] Engine pool ready — {pool_size} workers loaded. Health now returns 200.")

    import threading as _threading
    _threading.Thread(target=_preload_pool, daemon=True).start()

    print("✓ Standalone fast server listening on port 8080 (pool loading in background)...")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)

if __name__ == "__main__":
    main()
