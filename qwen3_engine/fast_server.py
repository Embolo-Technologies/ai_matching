import os
import sys
import time
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from rapidfuzz import process, fuzz
from qwen3_engine.searcher import FuzzySearcher
from qwen3_engine.pharma_data import formulations_compatible

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
    def __init__(self, searcher_instance, model_key: str = "qwen3"):
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
        # Strip leading pure-numeric tokens (vendor codes like "001", "123")
        while tokens and re.match(r'^\d+$', tokens[0]):
            tokens.pop(0)
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
        # Strip leading pure-numeric tokens (vendor codes like 001, 123)
        while tokens and re.match(r'^\d+$', tokens[0]) and len(tokens[0]) <= 3:
            tokens.pop(0)
        return ' '.join(tokens)

    def _has_unit_suffix(self, text: str, num: str) -> bool:
        """Check if a number in the text is followed by a unit suffix (mg, ml, etc.).
        If so, it's a strength, not a pack count."""
        pattern = re.escape(num) + r'\s*(?:mg|mcg|ml|gm|g|iu|units?)\b'
        return bool(re.search(pattern, text, re.IGNORECASE))

    def validate_match(self, query: str, candidate: dict, name: str = "", strict: bool = False) -> bool:
        """Validate a proposed match. In strict mode (AI fallback), thresholds are higher."""
        q_clean = self.normalize_text(query)
        c_name_clean = self.normalize_text(candidate['name'])
        c_brand_clean = self.normalize_text(candidate['brand']) if candidate.get('brand') else ""
        
        q_words = q_clean.split()
        if not q_words:
            return False
        q_brand = q_words[0]
        # If the first token is a short vendor prefix (length <= 2), try using the second token
        if len(q_brand) <= 2 and len(q_words) > 1:
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
            
        # Strength checks — use cleaned raw query to avoid vendor code numbers
        strength_query = self._clean_raw_query(name if name else query)
        q_nums = re.findall(r'\d+\.?\d*', strength_query)
        c_nums = re.findall(r'\d+\.?\d*', candidate['name'] + " " + candidate['strength'])
        
        # Skip common pack-count numbers — but numbers with unit suffixes are ALWAYS strengths
        PACK_COUNT_NUMS = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}
        raw_q = name if name else query
        raw_c = candidate['name'] + ' ' + candidate['strength']
        if q_nums:
            matched_num = False
            for q_num in q_nums:
                if q_num in PACK_COUNT_NUMS and '.' not in q_num and not self._has_unit_suffix(raw_q, q_num):
                    continue
                if q_num in c_nums:
                    matched_num = True
                    break
            strength_q_nums = [n for n in q_nums if n not in PACK_COUNT_NUMS or '.' in n or self._has_unit_suffix(raw_q, n)]
            # Strict: if query specifies strength, candidate MUST match
            if strength_q_nums and not matched_num:
                return False

        # Comprehensive formulation compatibility check
        if not formulations_compatible(query, candidate['name'] + " " + candidate['pack']):
            return False

        return True

    def heuristic_match(self, query: str, candidates: list, name: str = "") -> tuple:
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
            # If the first token is a short vendor prefix (length <= 2), try using the second token
            if len(q_brand) <= 2 and len(q_words) > 1:
                q_brand = q_words[1]
                
            c_words = c_name_clean.split()
            c_brand_words = c_brand_clean.split()
            
            brand_exact = False
            if c_words:
                UNIT_SUFFIXES = {'mg', 'ml', 'mcg', 'gm', 'g', 'cap', 'tab', 'tabs', 'caps'}
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

                if _brands_similar(q_brand_flat, c_brand_flat) or (c_brand_words_no_num and _brands_similar(q_brand_flat, c_brand_words_no_num[0])):
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
                if self.validate_match(query, cand, name=name):
                    # Calculate confidence score from brand similarity
                    brand_score = fuzz.ratio(q_brand_flat, c_brand_flat)
                    if c_brand_words_no_num:
                        word_score = fuzz.ratio(q_brand_flat, c_brand_words_no_num[0])
                        brand_score = max(brand_score, word_score)
                    return cand, brand_score
        return None, 0

    def llm_rerank(self, query: str, candidates: list, name: str = "") -> dict:
        """AI fallback: ask the local LLM to pick the correct match."""
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
                "You are a strict data-matching engine. Match the wholesaler input to a candidate number (1-5) or return null."
            )
        else:
            engine.system_prompt = (
                "You are a strict, deterministic medical data-matching engine.\n"
                "Match a 'Wholesaler Input' to a candidate from the 'Master Candidates List'.\n\n"
                "RULES (ALL must be satisfied):\n"
                "1. BRAND must match (typos OK: 'PNTOP'='PANTOP'). Different brands = null.\n"
                "2. FORMULATION/PACK must match. Oral solids (TAB, CAP, S10, STRIP, 10TAB, 15TAB, etc.) are compatible. Incompatible formulations (e.g. liquid SYRUP/drops/suspension vs solid TAB/CAP, injection INJ vs tablet TAB, topical CREAM/ointment/gel vs oral TAB) are mismatches and must be matched to null.\n"
                "3. STRENGTH must match. 40≠400, 650≠500. If query has strength but candidate doesn't = null.\n"
                "4. If NO candidate satisfies ALL rules, match_number = null. Never guess."
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
            # Full detailed prompt for 1.7B and larger models
            prompt = (
                # Example 1: Correct match
                "Wholesaler Input: PAN 40 TAB\n"
                "Master Candidates List:\n"
                "1. PAN 40 (Pack: 10 TAB, code: 101)\n"
                "2. PENTAB 40 (Pack: 1 PC, code: 102)\n\n"
                "Analysis:\n"
                "- Candidate 1: brand 'PAN'=match, strength '40'=match, pack 'TAB'=match. CORRECT.\n"
                "- Candidate 2: brand 'PENTAB'≠'PAN'. REJECT.\n"
                "Result:\n"
                '```json\n{"match_number": 1}\n```\n\n'
                # Example 2: Strength mismatch → null
                "Wholesaler Input: CALPOL 650\n"
                "Master Candidates List:\n"
                "1. CALPOL 500 (Pack: 15 TAB, code: 201)\n"
                "2. DOLO 650 (Pack: 15 TAB, code: 202)\n\n"
                "Analysis:\n"
                "- Candidate 1: brand match but strength 500≠650. REJECT.\n"
                "- Candidate 2: brand 'DOLO'≠'CALPOL'. REJECT.\n"
                "- No match found.\n"
                "Result:\n"
                '```json\n{"match_number": null}\n```\n\n'
                # Example 3: Formulation mismatch → null
                "Wholesaler Input: PAN 40 INJ\n"
                "Master Candidates List:\n"
                "1. PAN 40 (Pack: 10 TAB, code: 301)\n"
                "2. PAN D (Pack: 10 CAP, code: 302)\n\n"
                "Analysis:\n"
                "- Candidate 1: brand match, strength match, but formulation INJ≠TAB. REJECT.\n"
                "- Candidate 2: brand mismatch, formulation mismatch. REJECT.\n"
                "- No match found. Query asks for injection but all candidates are oral.\n"
                "Result:\n"
                '```json\n{"match_number": null}\n```\n\n'
                # Example 4: Oral Pack matching (abbreviation compatibility)
                "Wholesaler Input: LIOFEN 10MG 10TAB\n"
                "Master Candidates List:\n"
                "1. LIOFEN 10 (Pack: S10, code: M_3218)\n"
                "2. LIOFEN LIQUID (Pack: BL1, code: M_16166)\n\n"
                "Analysis:\n"
                "- Candidate 1: brand 'LIOFEN'=match, strength '10'=match, oral pack 'S10' is compatible with oral '10TAB'. CORRECT.\n"
                "- Candidate 2: formulation LIQUID is incompatible with TAB. REJECT.\n"
                "Result:\n"
                '```json\n{"match_number": 1}\n```\n\n'
                # Now the real query
                f"Wholesaler Input: {query}\n"
                "Master Candidates List:\n"
            )
            for i, cand in enumerate(candidates, 1):
                prompt += f"{i}. {cand['name']} (Pack: {cand['pack']}, code: {cand['code']})\n"

            prompt += (
                "\nAnalysis:\n"
                f"- Wholesaler input is '{query}'."
            )

        engine.clear_history()
        response = engine.generate(prompt, max_tokens=500, temperature=0.0)
        if self.model_key != "qwen3":
            response = f"- Wholesaler input is '{query}'." + response.strip()
        else:
            response = response.strip()
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
                        if self.validate_match(query, cand, name=name, strict=True):
                            return cand
                        else:
                            print(f"[AI] Guardrail blocked: {cand['name']}")
        except Exception as e:
            print(f"[AI] Parse error: {e}")
        return None

    def ai_verify(self, query: str, candidate: dict, name: str = "") -> bool:
        """AI verification: ask the LLM if a heuristic match is correct (yes/no).
        Used for medium-confidence matches (80-94% brand similarity)."""
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

        q_display = name if name else query

        prompt = (
            "You are a medical data verification assistant. Decide if the Chemist name and the Wholesaler name refer to the exact same medicine (same brand/drug, same strength, compatible formulation).\n"
            "Trailing unit symbols like 'MG' can be omitted in one of them. Oral forms like tablets (TAB) and capsules (CAP) match each other.\n\n"
            "Examples:\n"
            "Chemist: CLOFRANIL 50MG TAB 10TAB\n"
            "Wholesaler: CLOFRANIL 50 (S10)\n"
            "Answer: YES\n\n"
            "Chemist: PAN 40 TAB\n"
            "Wholesaler: PAN D (10 CAP)\n"
            "Answer: NO\n\n"
            "Chemist: CALPOL 650\n"
            "Wholesaler: CALPOL 500 (15 TAB)\n"
            "Answer: NO\n\n"
            "Chemist: ENCORATE SYP 100ML\n"
            "Wholesaler: ENCORATE (S10)\n"
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


    def match(self, query: str, name: str = "", pack: str = "") -> dict:
        results, _ = self.searcher.search(name if name else query, pack=pack, top_k=5)
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
        heur_match, confidence = self.heuristic_match(query, candidates, name=name)
        
        if heur_match:
            if confidence >= 95:
                # HIGH confidence → accept directly
                return heur_match
            else:
                # MEDIUM confidence → ask AI to verify
                print(f"[AI Verify] Heuristic matched '{query}' → '{heur_match['name']}' (confidence={confidence}%). Verifying with AI...")
                if self.ai_verify(query, heur_match, name=name):
                    return heur_match
                else:
                    print(f"[AI Verify] Rejected: '{heur_match['name']}' is NOT the same as '{query}'")

        # Stage 2: AI fallback (slow but accurate for edge cases)
        if candidates:
            print(f"[AI Fallback] Heuristic failed for '{query}'. Asking LLM...")
            ai_match = self.llm_rerank(query, candidates, name=name)
            if ai_match:
                return ai_match

        return None

_matcher = None

@app.route("/match", methods=["POST"])
def match_item():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    pack = data.get("pack", "").strip()
    query = data.get("query", "").strip()
    
    if not query:
        query = f"{name} {pack}".strip()
    if not query:
        return jsonify({"error": "Field 'query' or 'name' is required."}), 400
        
    t0 = time.time()
    result = _matcher.match(query, name=name, pack=pack)
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
    return jsonify({"status": "ok", "mode": "standalone_fast"})

def main():
    global _searcher, _matcher
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone REST API Server")
    parser.add_argument("--csv", type=str, default="", help="Path to catalog CSV")
    args = parser.parse_args()
    
    csv_path = args.csv
    if not csv_path:
        # Check next to the script or inside parent
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
        print(f"Error: Unified CSV database not found.")
        sys.exit(1)
        
    print(f"Loading search database from: {csv_path}...")
    _searcher = FuzzySearcher(csv_path=csv_path)
    _searcher.load()
    
    _matcher = FastMatcher(_searcher)
    _matcher.load_catalog()
    
    # Check if AI model is available
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    model_file = os.path.join(model_dir, "Qwen3-0.6B-Q4_K_M.gguf")
    if os.path.exists(model_file):
        print("✓ AI model found — LLM fallback enabled (lazy load on first miss).")
    else:
        print("⚠ AI model not found — running heuristic-only mode.")

    print("✓ Standalone fast server listening on port 8080...")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)

if __name__ == "__main__":
    main()
