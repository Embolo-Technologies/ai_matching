import csv
import time
import os
import re
import sqlite3
import json
from typing import List, Dict, Optional
from rapidfuzz import process, fuzz
from qwen3_engine.engine import Qwen3Engine
from qwen3_engine.config import N_CTX_MATCHER
from qwen3_engine.searcher import FuzzySearcher
from qwen3_engine.pharma_data import formulations_compatible

def is_abbreviation(abbrev: str, full: str) -> bool:
    abbrev = abbrev.lower().strip()
    full = full.lower().strip()
    if len(abbrev) < 3 or not full:
        return False
    if abbrev[0] != full[0]:
        return False
    it = iter(full)
    return all(char in it for char in abbrev)

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

class HybridMatcher:
    def __init__(self, master_csv_path: str, model_key: str = "qwen3"):
        self.master_csv_path = master_csv_path
        self.model_key = model_key
        self.catalog: List[Dict[str, str]] = []
        self.engine = Qwen3Engine(model_key=self.model_key)
        self.searcher = FuzzySearcher(self.master_csv_path)
        
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache.db")
        self._init_db()
        
        # Override system prompt for medical matching
        if self.model_key == "qwen3":
            self.engine.system_prompt = (
                "You are a strict data-matching engine. Match the wholesaler input to a candidate number (1-5) or return null."
            )
        else:
            self.engine.system_prompt = (
                "You are a strict, deterministic medical data-matching engine.\n"
                "Your job is to match a 'Wholesaler Input' to a single correct candidate in a 'Master Candidates List'.\n\n"
                "CRITICAL MATCHING RULES:\n"
                "1. BRAND/NAME MATCH: The brand name or generic chemical name of the candidate must match the wholesaler input (accounting for minor spelling errors or abbreviations, e.g., 'agument' -> 'AUGMENTIN'). If the brands are completely different (e.g., 'AVAS' vs 'agument'), they DO NOT MATCH.\n"
                "2. FORMULATION MATCH: The formulation (e.g., INJ vs TAB, injection vs tablet, liquid vs solid) must match. If the query specifies 'injection' or 'inj', and the candidate is a tablet ('TAB') or is a brand that is a tablet, they DO NOT MATCH. Mismatched formulations MUST result in null/no match.\n"
                "3. STRENGTH MATCH: The strength (e.g., 40mg vs 400mg vs 40) must match. 40 and 400 are completely different. If the query has '40' or '40mg', it cannot match '400' or '400mg'. If the query doesn't specify a strength, or the product doesn't have one (e.g., medical accessories, bags), strength matches can be ignored.\n"
                "4. DETERMINISTIC/NO GUESSING: If there is no exact matching candidate that satisfies all the rules, you MUST set match_number to null."
            )

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches_cache (
                query_normalized TEXT PRIMARY KEY,
                matched_code TEXT,
                wholesaler_original TEXT,
                matched_name TEXT
            )
        """)
        conn.commit()
        conn.close()

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = text.replace("-", " ").replace("/", " ").replace(",", " ")
        # Split stuck numbers from words: PAN40 -> PAN 40, DOLO650 -> DOLO 650
        text = re.sub(r'([a-z])([0-9])', r'\1 \2', text)
        text = re.sub(r'([0-9])([a-z])', r'\1 \2', text)
        text = re.sub(r'\b(tabs?|tablets?|inj(ection)?s?|caps?(ules?)?|susp(ension)?|syp|syrup|drops?|cream|ointment|gel|liq(uid)?|sol(ution)?)\b', '', text)
        text = re.sub(r'[^a-z0-9\s\.]', '', text)
        tokens = text.split()
        # Strip leading pure-numeric tokens (vendor codes like "001", "123")
        while tokens and re.match(r'^\d+$', tokens[0]):
            tokens.pop(0)
        return ' '.join(tokens)

    def _clean_raw_query(self, text: str) -> str:
        """Clean raw query for strength extraction: strip vendor codes, fix separators."""
        text = text.replace(",", " ").replace("/", " ").replace("-", " ")
        text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)
        text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'(\d)\.(?!\d)', r'\1', text)
        tokens = text.split()
        while tokens and re.match(r'^\d+$', tokens[0]) and len(tokens[0]) <= 3:
            tokens.pop(0)
        return ' '.join(tokens)

    def _has_unit_suffix(self, text: str, num: str) -> bool:
        """Check if a number in the text is followed by a unit suffix (mg, ml, etc.).
        If so, it's a strength, not a pack count."""
        pattern = re.escape(num) + r'\s*(?:mg|mcg|ml|gm|g|iu|units?)\b'
        return bool(re.search(pattern, text, re.IGNORECASE))

    def get_cached_match(self, query: str) -> Optional[Dict[str, str]]:
        normalized = self.normalize_text(query)
        if not normalized:
            return None
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT matched_code, matched_name FROM matches_cache WHERE query_normalized = ?",
            (normalized,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            code, name = row
            if code == "NONE":
                return None
            if not self.catalog:
                self.load_catalog()
            for cand in self.catalog:
                if cand['code'] == code:
                    return cand
        return None

    def save_cached_match(self, query: str, candidate: Optional[Dict[str, str]]):
        normalized = self.normalize_text(query)
        if not normalized:
            return
            
        code = candidate['code'] if candidate else "NONE"
        name = candidate['name'] if candidate else "NONE"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO matches_cache (query_normalized, matched_code, wholesaler_original, matched_name) VALUES (?, ?, ?, ?)",
                (normalized, code, query, name)
            )
            conn.commit()
        except Exception as e:
            print(f"Error saving to cache: {e}")
        finally:
            conn.close()

    def validate_match(self, query: str, candidate: Dict[str, str], name: str = "", compname: str = "") -> bool:
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
            
        # Check brand similarity including spaceless flat matches
        q_brand_flat = q_brand.replace(" ", "")
        c_name_flat = c_name_clean.replace(" ", "")
        c_brand_flat = c_brand_clean.replace(" ", "")
        
        # Check brand match (strict fuzzy similarity check for typos of the same name)
        is_abbrev_match = (
            is_abbreviation(q_brand, c_name_clean) or
            is_abbreviation(q_brand, c_brand_clean)
        )
        brand_match = (
            is_abbrev_match or
            q_brand in c_name_clean or 
            q_brand in c_brand_clean or 
            q_brand_flat in c_name_flat or
            c_name_flat in q_brand_flat or
            (c_brand_flat and (q_brand_flat in c_brand_flat or c_brand_flat in q_brand_flat)) or
            process.extractOne(q_brand, c_name_clean.split(), scorer=fuzz.ratio, score_cutoff=85) is not None or
            (c_brand_clean and process.extractOne(q_brand, c_brand_clean.split(), scorer=fuzz.ratio, score_cutoff=85) is not None)
        )
        # Fallback for vendor prefix codes (e.g. "RS-PAN 40" -> brand "PAN" is in query words)
        if not brand_match:
            c_first_word = c_name_clean.split()[0] if c_name_clean.split() else ""
            c_brand_first = c_brand_clean.split()[0] if c_brand_clean.split() else ""
            if (c_first_word and len(c_first_word) >= 2 and c_first_word.isalpha() and c_first_word in q_words) or (c_brand_first and len(c_brand_first) >= 2 and c_brand_first.isalpha() and c_brand_first in q_words):
                brand_match = True

        if not brand_match:
            return False
            
        # Strength checks — use cleaned raw query to avoid vendor code numbers
        strength_query = self._clean_raw_query(name if name else query)
        q_nums = re.findall(r'\d+\.?\d*', strength_query)
        c_nums = re.findall(r'\d+\.?\d*', candidate['name'] + " " + candidate['strength'])
        
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

        # Smart Company Compatibility check (always checked if input company is provided)
        if compname and compname.strip() and compname.strip() not in ('--', 'none', 'null', 'nan'):
            c_brand = candidate.get('brand', '') or ''
            if c_brand and not companies_compatible(compname, c_brand):
                return False

        return True

    def heuristic_match(self, query: str, candidates: List[Dict[str, str]], name: str = "", compname: str = "") -> Optional[Dict[str, str]]:
        if not candidates:
            return None
            
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
                if self.validate_match(query, cand, name=name, compname=compname):
                    return cand
        return None
 
    def match(self, query: str, top_k: int = 5, name: str = "", pack: str = "", compname: str = "") -> Optional[Dict[str, str]]:
        cached = self.get_cached_match(query)
        if cached:
            print(f"[Cache Hit] '{query}' matched to '{cached['name']}' (Code: {cached['code']})")
            return cached
            
        if not self.catalog:
            self.load_catalog()
            
        candidates = self.fuzzy_search(query, top_k=top_k, name=name, pack=pack)
        if not candidates:
            self.save_cached_match(query, None)
            return None
            
        heur_match = self.heuristic_match(query, candidates, name=name, compname=compname)
        if heur_match:
            print(f"[Heuristic Match] '{query}' matched to '{heur_match['name']}' (Code: {heur_match['code']})")
            self.save_cached_match(query, heur_match)
            return heur_match
            
        print(f"[AI Rerank] Query '{query}' requires LLM logic. Reranking...")
        llm_match = self.llm_rerank(query, candidates, name=name, compname=compname)
        if llm_match:
            print(f"[LLM Match] '{query}' matched to '{llm_match['name']}' (Code: {llm_match['code']})")
            self.save_cached_match(query, llm_match)
            return llm_match
            
        print(f"[No Match] '{query}' could not be matched.")
        self.save_cached_match(query, None)
        return None

    def load_catalog(self):
        if not self.searcher.is_loaded():
            self.searcher.load()
            
        self.catalog = []
        for cand in self.searcher.catalog:
            self.catalog.append({
                "code": cand["code"],
                "name": cand["name"],
                "brand": cand["compname"],
                "pack": cand["pack"],
                "strength": cand["strength"],
            })
        print(f"Loaded {len(self.catalog)} items from FuzzySearcher catalog.")
 
    def fuzzy_search(self, query: str, top_k: int = 5, name: str = "", pack: str = "") -> List[Dict[str, str]]:
        if not self.catalog:
            self.load_catalog()
        if not name:
            name = query
            
        results, _ = self.searcher.search(name, pack=pack, top_k=top_k)
        candidates = []
        for r in results:
            candidates.append({
                "code": r["code"],
                "name": r["name"],
                "brand": r["compname"],
                "pack": r["pack"],
                "strength": r["strength"],
            })
        return candidates

    def llm_rerank(self, query: str, candidates: List[Dict[str, str]], name: str = "", compname: str = "") -> Optional[Dict[str, str]]:
        if not self.engine.is_loaded():
            print(f"Loading LLM {self.model_key} into GPU...")
            self.engine.load(n_ctx=N_CTX_MATCHER)
            
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
                "Wholesaler Input: PAN 40 TAB (Company: SUN PHARMA)\n"
                "Master Candidates List:\n"
                "1. PAN 40 (Pack: 10 TAB, Company: SUN PHAR, code: 101)\n"
                "2. PENTAB 40 (Pack: 1 PC, Company: TORQUE, code: 102)\n"
                "3. PAN D (Pack: 10 CAP, Company: SUN PHAR, code: 103)\n\n"
                "Analysis:\n"
                "- Wholesaler input is 'PAN 40 TAB'. Brand is 'PAN', strength is '40', formulation is 'TAB', Company is 'SUN PHARMA'.\n"
                "- Candidate 1 matches brand 'PAN' (from 'PAN 40'), strength '40', formulation 'TAB', and Company 'SUN PHAR'. Exact match.\n"
                "Result:\n"
                "```json\n"
                "{\n"
                "  \"match_number\": 1\n"
                "}\n"
                "```\n\n"
                "Wholesaler Input: CALPOL 650 (Company: GSK)\n"
                "Master Candidates List:\n"
                "1. CALPOL 500 (Pack: 15 TAB, Company: GSK, code: 201)\n"
                "2. DOLO 650 (Pack: 15 TAB, Company: MICRO, code: 202)\n"
                "3. PENTAB 40 (Pack: 1 PC, Company: TORQUE, code: 203)\n\n"
                "Analysis:\n"
                "- Wholesaler input is 'CALPOL 650'. Brand is 'CALPOL', strength is '650', Company is 'GSK'.\n"
                "- Candidate 1 has mismatched strength (500 vs 650).\n"
                "- Candidate 2 has mismatched brand ('DOLO' vs 'CALPOL').\n"
                "- Candidate 3 has mismatched brand. No candidate matches.\n"
                "Result:\n"
                "```json\n"
                "{\n"
                "  \"match_number\": null\n"
                "}\n"
                "```\n\n"
                "Wholesaler Input: BRIV SYRUP 100ML (Company: DR REDDY)\n"
                "Master Candidates List:\n"
                "1. BRIVATAB 100ML ORAL SOLUTION (Pack: 100ML, Company: HETERO, code: 301)\n\n"
                "Analysis:\n"
                "- Wholesaler input is 'BRIV SYRUP 100ML'. Brand is 'BRIV', strength is '100ML', Company is 'DR REDDY'.\n"
                "- Candidate 1 has mismatched Company ('HETERO' vs 'DR REDDY'). Companies are different, so it cannot match.\n"
                "Result:\n"
                "```json\n"
                "{\n"
                "  \"match_number\": null\n"
                "}\n"
                "```\n\n"
                f"Wholesaler Input: {query}"
            )
            if compname and compname.strip() and compname.strip() != '--':
                prompt += f" (Company: {compname.strip()})"
            prompt += "\nMaster Candidates List:\n"
            for i, cand in enumerate(candidates, 1):
                cand_company = cand.get('brand', '') or ''
                prompt += f"{i}. {cand['name']} (Pack: {cand['pack']}, Company: {cand_company}, code: {cand['code']})\n"
                
            prompt += (
                "\nWrite exactly two sections: 'Analysis:' followed by your step-by-step comparisons, and 'Result:' followed by the final match JSON block.\n\n"
                f"Analysis:\n"
                f"- Wholesaler input is '{query}'."
            )
        
        self.engine.clear_history()
        response = self.engine.generate(prompt, max_tokens=500, temperature=0.0)
        if self.model_key != "qwen3":
            response = f"- Wholesaler input is '{query}'." + response.strip()
        else:
            response = response.strip()
        print(f"LLM Output:\n{response}")
        
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
                        if self.validate_match(query, cand, name=name, compname=compname):
                            return cand
                        else:
                            print(f"[Guardrail] Match code {cand['code']} failed validation checks.")
        except Exception as e:
            print(f"Failed parsing: {e}")
        return None
