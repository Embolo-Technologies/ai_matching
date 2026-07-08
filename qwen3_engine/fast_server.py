import os
os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
import sys
import time
import re
import json
import subprocess
import queue
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from flask_cors import CORS
from rapidfuzz import process, fuzz
from qwen3_engine.searcher import FuzzySearcher
from qwen3_engine.pharma_data import formulations_compatible

# ─── GPU Concurrency & Engine Pool Helpers ──────────────────────────────────────────
def get_free_gpu_memory() -> int:
    """Returns free GPU memory in MB, or 0 if no GPU/nvidia-smi is available."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        out = res.stdout.strip()
        if out:
            lines = [int(x) for x in out.splitlines() if x.strip().isdigit()]
            return sum(lines) if lines else 0
    except Exception:
        pass
    return 0

def calculate_optimal_workers(model_key: str) -> int:
    """Returns fixed 40 workers — tuned for A100 40GB GPU with n_ctx=1024 and mmap=True."""
    return 40


class EnginePool:
    """Thread-safe pool of independent model instances to prevent concurrency race conditions."""
    def __init__(self, model_key: str, size: int):
        self.model_key = model_key
        self.size = size
        self.pool = queue.Queue()
        
    def populate(self):
        from qwen3_engine.engine import Qwen3Engine
        from qwen3_engine.config import N_CTX_MATCHER
        print(f"[EnginePool] Loading {self.size} workers sequentially (n_ctx={N_CTX_MATCHER})...")
        for i in range(self.size):
            t0 = time.time()
            engine = Qwen3Engine(model_key=self.model_key)
            engine.load(n_ctx=N_CTX_MATCHER)
            self.pool.put(engine)
            print(f"  ✓ Worker {i+1}/{self.size} ready in {time.time()-t0:.1f}s")
            
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
            'th', 'pg', 'nt', 'm', 'beta', 'tri', 'ch', 'ln', 't', 'o', 'av', 'd3'
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
            # Full detailed prompt for 1.7B and larger models, but without analysis output
            prompt = (
                # Example 1: Correct match
                "Wholesaler Input: PAN 40 TAB (Company: SUN PHARMA)\n"
                "Master Candidates List:\n"
                "1. PAN 40 (Pack: 10 TAB, Company: SUN PHAR, code: 101)\n"
                "2. PENTAB 40 (Pack: 1 PC, Company: TORQUE, code: 102)\n"
                "Result:\n"
                '```json\n{"match_number": 1}\n```\n\n'
                # Example 2: Strength mismatch → null
                "Wholesaler Input: CALPOL 650 (Company: GSK)\n"
                "Master Candidates List:\n"
                "1. CALPOL 500 (Pack: 15 TAB, Company: GSK, code: 201)\n"
                "2. DOLO 650 (Pack: 15 TAB, Company: MICRO, code: 202)\n"
                "Result:\n"
                '```json\n{"match_number": null}\n```\n\n'
                # Example 3: Formulation mismatch → null
                "Wholesaler Input: PAN 40 INJ (Company: SUN PHARMA)\n"
                "Master Candidates List:\n"
                "1. PAN 40 (Pack: 10 TAB, Company: SUN PHAR, code: 301)\n"
                "2. PAN D (Pack: 10 CAP, Company: SUN PHAR, code: 302)\n"
                "Result:\n"
                '```json\n{"match_number": null}\n```\n\n'
                # Example 4: Oral Pack matching (abbreviation compatibility)
                "Wholesaler Input: LIOFEN 10MG 10TAB (Company: INTAS)\n"
                "Master Candidates List:\n"
                "1. LIOFEN 10 (Pack: S10, Company: INTAS, code: M_3218)\n"
                "2. LIOFEN LIQUID (Pack: BL1, Company: INTAS, code: M_16166)\n"
                "Result:\n"
                '```json\n{"match_number": 1}\n```\n\n'
                # Now the real query
                f"Wholesaler Input: {query}"
            )
            if compname and compname.strip() and compname.strip() != '--':
                prompt += f" (Company: {compname.strip()})"
            prompt += "\nMaster Candidates List:\n"
            for i, cand in enumerate(candidates, 1):
                cand_company = cand.get('brand', '') or ''
                prompt += f"{i}. {cand['name']} (Pack: {cand['pack']}, Company: {cand_company}, code: {cand['code']})\n"

            prompt += "\nResult:"

        engine.clear_history()
        response = engine.generate(prompt, max_tokens=50, temperature=0.0).strip()
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

    def ai_verify(self, query: str, candidate: dict, name: str = "", compname: str = "", engine=None) -> bool:
        """AI verification: ask the LLM if a heuristic match is correct (yes/no).
        Used for medium-confidence matches (80-94% brand similarity)."""
        if engine is None:
            global _llm_engines
            engine = _llm_engines.get(self.model_key)
            if engine is None:
                try:
                    from qwen3_engine.engine import Qwen3Engine
                    from qwen3_engine.config import N_CTX_MATCHER
                    engine = Qwen3Engine(model_key=self.model_key)
                    print(f"[AI] Loading local {self.model_key} LLM for verification...")
                    engine.load(n_ctx=N_CTX_MATCHER if 'N_CTX_MATCHER' in dir() else 2048)
                    print(f"[AI] ✓ LLM {self.model_key} loaded successfully.")
                    _llm_engines[self.model_key] = engine
                except Exception as e:
                    print(f"[AI] ✗ Failed to load LLM: {e}")
                    return False

        if not engine or not engine.is_loaded():
            return False

        # Set system prompt unconditionally
        engine.system_prompt = "You verify if two medicine descriptions refer to the same medicine. Answer ONLY with YES or NO."

        cand_str = f"{candidate['name']}"
        if candidate.get('strength'):
            cand_str += f" {candidate['strength']}"
        if candidate.get('pack'):
            cand_str += f" ({candidate['pack']})"
        if candidate.get('brand'):
            cand_str += f" [Company: {candidate['brand']}]"

        q_display = name if name else query
        if compname and compname.strip() and compname.strip() != '--':
            q_display += f" [Company: {compname}]"

        prompt = (
            "You are a medical data verification assistant. Decide if the Chemist name and the Wholesaler name refer to the exact same medicine (same brand/drug, same strength, compatible formulation, same company).\n"
            "Trailing unit symbols like 'MG' can be omitted in one of them. Oral forms like tablets (TAB) and capsules (CAP) match each other.\n"
            "If the companies are clearly specified and different, the answer must be NO.\n\n"
            "Examples:\n"
            "Chemist: CLOFRANIL 50MG TAB 10TAB [Company: INTAS]\n"
            "Wholesaler: CLOFRANIL 50 (S10) [Company: INTAS]\n"
            "Answer: YES\n\n"
            "Chemist: PAN 40 TAB [Company: ALKEM]\n"
            "Wholesaler: PAN D (10 CAP) [Company: ALKEM]\n"
            "Answer: NO\n\n"
            "Chemist: CALPOL 650 [Company: GSK]\n"
            "Wholesaler: CALPOL 500 (15 TAB) [Company: GSK]\n"
            "Answer: NO\n\n"
            "Chemist: BRIV SYRUP 100ML [Company: DR REDDY]\n"
            "Wholesaler: BRIVATAB 100ML ORAL SOLUTION (100ML) [Company: HETERO]\n"
            "Answer: NO\n\n"
            f"Chemist: {q_display}\n"
            f"Wholesaler: {cand_str}\n"
            "Answer:"
        )

        engine.clear_history()
        response = engine.generate(prompt, max_tokens=20, temperature=0.0).strip().upper()
        print(f"[AI Verify] '{q_display}' vs '{candidate['name']}' → {response}")

        # Accept if AI says YES and guardrail also passes
        if 'YES' in response:
            return True
        return False


    def match(self, query: str, name: str = "", pack: str = "", compname: str = "", engine=None) -> dict:
        search_query = name if name else query
        results, _ = self.searcher.search(search_query, pack=pack, compname=compname, top_k=8)

        # Second-pass: if first search returned fewer than 4 results, retry with
        # just the first substantive word (catches abbreviations like "AMC" → "AMOXICLAV")
        if len(results) < 4:
            words = [w for w in re.sub(r'[^a-zA-Z0-9\s]', ' ', search_query).split()
                     if len(w) >= 3 and not w.isdigit()]
            if words:
                extra, _ = self.searcher.search(words[0], pack=pack, compname=compname, top_k=8)
                seen = {r["code"] for r in results}
                for r in extra:
                    if r["code"] not in seen:
                        results.append(r)
                        seen.add(r["code"])
                results = results[:8]

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
        heur_match, confidence = self.heuristic_match(query, candidates, name=name, compname=compname)
        
        if heur_match:
            if confidence >= 95:
                # HIGH confidence → accept directly
                return heur_match
            else:
                # MEDIUM confidence → ask AI to verify
                print(f"[AI Verify] Heuristic matched '{query}' → '{heur_match['name']}' (confidence={confidence}%). Verifying with AI...")
                if self.ai_verify(query, heur_match, name=name, compname=compname, engine=engine):
                    return heur_match
                else:
                    print(f"[AI Verify] Rejected: '{heur_match['name']}' is NOT the same as '{query}'")

        # Stage 2: AI fallback (slow but accurate for edge cases)
        if candidates:
            print(f"[AI Fallback] Heuristic failed for '{query}'. Asking LLM...")
            ai_match = self.llm_rerank(query, candidates, name=name, compname=compname, engine=engine)
            if ai_match:
                return ai_match

        return None

_matcher = None
_engine_pool = None   # pre-loaded at startup; None until ready

@app.route("/match", methods=["POST"])
def match_item():
    data = request.get_json(silent=True) or {}

    if "products" in data:
        # ── Batch match mode ──────────────────────────────────────────────────
        products      = data.get("products", [])

        # Progress callback info (injected by backend)
        backend_url   = (data.get("backend_url") or "").rstrip("/")
        job_id        = (data.get("job_id") or "").strip()
        signal_secret = (data.get("signal_secret") or "").strip()
        progress_url  = (
            f"{backend_url}/medicine-matching/admin/gpu-match/{job_id}/gpu-progress"
            if backend_url and job_id and signal_secret else None
        )

        # ── Use pre-loaded global searcher and engine pool ──────────────────
        if _searcher is None or _matcher is None or _engine_pool is None:
            return jsonify({"error": "Server still initialising — pool not ready."}), 503

        temp_matcher = _matcher

        import threading

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
    if _searcher is None or _matcher is None or _engine_pool is None:
        return jsonify({"status": "initialising"}), 503
    return jsonify({"status": "ok", "mode": "standalone_fast", "workers": _engine_pool.size})

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
        from qwen3_engine.config import N_CTX_MATCHER
        pool_size = calculate_optimal_workers("gemma4_2b")
        print(f"[Startup] Pre-loading engine pool: {pool_size} workers (n_ctx={N_CTX_MATCHER})...")
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
