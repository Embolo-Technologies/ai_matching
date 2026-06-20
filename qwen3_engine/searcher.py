import csv
import os
import re
import time
from typing import List, Dict, Tuple
from rapidfuzz import fuzz, process

class FuzzySearcher:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.catalog: List[Dict[str, str]] = []
        self.name_corpus: List[str] = []
        self.name_corpus_normalized: List[str] = []
        self.name_corpus_tokens: List[List[str]] = []
        self._prefix_index: Dict[str, List[int]] = {}
        self._company_prefix_index: Dict[str, List[int]] = {}
        
    def is_loaded(self) -> bool:
        return len(self.catalog) > 0
        
    def load(self):
        t0 = time.time()
        self.catalog = []
        self.name_corpus = []
        self.name_corpus_normalized = []
        self.name_corpus_tokens = []
        self._prefix_index = {}
        self._company_prefix_index = {}
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Database CSV not found at: {self.csv_path}")
            
        with open(self.csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames if reader.fieldnames else []
            key_map = {h.lower(): h for h in headers}
            
            def get_val(row, k):
                actual_key = key_map.get(k.lower())
                return row.get(actual_key, "") if actual_key else ""

            for r in reader:
                code = get_val(r, "code").strip()
                name = get_val(r, "name").strip()
                compname = get_val(r, "compname").strip()
                pack = get_val(r, "pack").strip()
                strength = get_val(r, "strength").strip()
                
                if code and name:
                    self.catalog.append({
                        "code": code,
                        "name": name,
                        "compname": compname,
                        "pack": pack,
                        "strength": strength
                    })
                    self.name_corpus.append(name.lower())

        # Pre-compute normalized names and tokens for every catalog item
        for item in self.catalog:
            norm_name = self._normalize_text(item["name"])
            self.name_corpus_normalized.append(norm_name)
            self.name_corpus_tokens.append(self._meili_tokenize(item["name"]))

        # Build first-word prefix index (name)
        for i, norm_name in enumerate(self.name_corpus_normalized):
            words = norm_name.split()
            if words:
                key = words[0][:3]
                if key not in self._prefix_index:
                    self._prefix_index[key] = []
                self._prefix_index[key].append(i)

        # Build company name prefix index
        for i, item in enumerate(self.catalog):
            comp = item.get("compname", "")
            if comp:
                comp_norm = self._normalize_text(comp)
                words = comp_norm.split()
                if words:
                    key = words[0][:3]
                    if key not in self._company_prefix_index:
                        self._company_prefix_index[key] = []
                    self._company_prefix_index[key].append(i)

        print(f"[Searcher] Loaded {len(self.catalog):,} items in {time.time() - t0:.2f}s")

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = text.replace("-", " ").replace("/", " ").replace(",", " ")
        # Split stuck numbers from words: PAN40 -> PAN 40, DOLO650 -> DOLO 650
        text = re.sub(r'([a-z])([0-9])', r'\1 \2', text)
        text = re.sub(r'([0-9])([a-z])', r'\1 \2', text)
        # Remove common formulations
        text = re.sub(r'\b(tabs?|tablets?|inj(ection)?s?|caps?(ules?)?|susp(ension)?|syp|syrup|drops?|cream|ointment|gel|liq(uid)?|sol(ution)?)\b', '', text)
        text = re.sub(r'[^a-z0-9\s\.]', '', text) # Keep decimals
        return ' '.join(text.split())

    def _parse_pack(self, pack_str: str) -> tuple:
        """
        Extract (qty_str, canonical_unit) from any pack format for cross-format comparison.

        Master verbose formats:
            "strip of 10 tablets" → ("10", "tab")
            "bottle of 100 ml Syrup" → ("100", "ml")
            "tube of 20 gm Cream" → ("20", "gm")

        Vendor compact formats:
            "10S" / "10 S" / "S10" → ("10", "tab")
            "100ML" / "60ML"       → ("100", "ml")
            "1*15" / "3*10 TAB"    → ("15"/"10", "tab")   ← per-strip count
            "1PS"                  → ("1", "pc")
            "4 TAB" / "6S"        → ("4"/"6", "tab")
            "1*15G"               → ("15", "gm")
        """
        if not pack_str:
            return ("", "")
        text = pack_str.lower().strip().replace(",", " ")
        # Split stuck number+letter: "100ml" → "100 ml", "10s" → "10 s", "1ps" → "1 ps"
        text = re.sub(r'([0-9])([a-z])', r'\1 \2', text)
        text = re.sub(r'([a-z])([0-9])', r'\1 \2', text)

        # ── Unit detection (most-specific first) ─────────────────────────────
        unit = ""
        if re.search(r'\bml\b', text):                          unit = "ml"
        elif re.search(r'\bgm\b|\bgrams?\b', text):             unit = "gm"
        elif re.search(r'\b(caps?|capsules?)\b', text):         unit = "cap"
        elif re.search(r'\b(tabs?|tablets?)\b', text):          unit = "tab"
        elif re.search(r'\bvials?\b', text):                    unit = "vial"
        elif re.search(r'\b(amps?|ampoules?)\b', text):         unit = "amp"
        elif re.search(r'\bsachets?\b', text):                  unit = "sachet"
        elif re.search(r'\b(pc|pcs|pieces?)\b|\bps\b', text):  unit = "pc"
        elif re.search(r'\bstrips?\b', text):                   unit = "tab"

        # ── Quantity extraction ───────────────────────────────────────────────
        # A*B / AxB: take B (per-strip count), check for trailing unit suffix
        mult = re.search(r'(\d+)\s*[x*×]\s*(\d+(?:\.\d+)?)\s*(ml|gm|g\b)?', text)
        if mult:
            qty = mult.group(2)
            if not unit:
                sfx = (mult.group(3) or "").strip()
                unit = "ml" if sfx == "ml" else ("gm" if sfx in ("gm", "g") else "tab")
        else:
            # "10 s" / "s 10" (strip-S notation)
            s_trail = re.search(r'(\d+)\s*s\b', text)
            s_lead  = re.search(r'\bs\s*(\d+)', text)
            if (s_trail or s_lead) and not unit:
                qty  = s_trail.group(1) if s_trail else s_lead.group(1)
                unit = "tab"
            else:
                # General: strip container/unit words then grab first number
                clean = re.sub(
                    r'\b(ml|gm|g|mg|tab|tabs|tablet|tablets|cap|caps|capsule|capsules|'
                    r'strip|strips|vial|vials|amp|amps|ampoule|ampoules|sachet|sachets|'
                    r'pc|pcs|piece|pieces|ps|bottle|tube|box|of|the|per|each)\b',
                    ' ', text
                )
                num = re.search(r'(\d+(?:\.\d+)?)', clean)
                qty = num.group(1) if num else ""

        return (qty, unit)

    def _split_easysol_blob(self, token: str) -> List[str]:
        # Split digits from letters (e.g. "pantop40" -> ["pantop", "40"])
        parts = re.findall(r'([a-z]+|[0-9\.]+)', token)
        return parts if parts else [token]

    def _meili_tokenize(self, text: str) -> List[str]:
        raw_words = self._normalize_text(text).split()
        tokens = []
        for word in raw_words:
            subtokens = self._split_easysol_blob(word)
            for sub in subtokens:
                # Normalize units
                if sub == "cc" or sub == "grm":
                    sub = "ml"
                elif sub == "g":
                    sub = "gm"
                tokens.append(sub)
        return tokens

    def _meili_score(self, query_tokens: List[str], item_tokens: List[str]) -> Tuple:
        matched_count = 0
        exact_count = 0
        prefix_count = 0
        
        for q_tok in query_tokens:
            found = False
            for i_tok in item_tokens:
                if q_tok == i_tok:
                    exact_count += 1
                    matched_count += 1
                    found = True
                    break
                elif i_tok.startswith(q_tok):
                    prefix_count += 1
                    matched_count += 1
                    found = True
                    break
            # Fuzzy fallback
            if not found and len(q_tok) >= 3:
                for i_tok in item_tokens:
                    if len(i_tok) >= 3 and fuzz.ratio(q_tok, i_tok) >= 70:
                        matched_count += 0.8
                        break
                        
        # Front matches (consecutive matched tokens from index 0)
        front_matches = 0
        limit = min(len(query_tokens), len(item_tokens))
        for i in range(limit):
            if query_tokens[i] == item_tokens[i]:
                front_matches += 1
            else:
                break
                
        # Proximity score (gap size between matching index elements)
        proximity = 0
        matched_indices = []
        for q_tok in query_tokens:
            for idx, i_tok in enumerate(item_tokens):
                if q_tok == i_tok or i_tok.startswith(q_tok):
                    matched_indices.append(idx)
                    break
        if len(matched_indices) >= 2:
            proximity = sum(abs(matched_indices[i] - matched_indices[i-1]) for i in range(1, len(matched_indices)))
            
        exactness = exact_count * 2 + prefix_count
        len_diff = abs(len(query_tokens) - len(item_tokens))
        
        return (front_matches, matched_count, -proximity, exactness, -len_diff)

    def search(self, name: str, pack: str = "", compname: str = "", top_k: int = 5) -> Tuple[List[Dict], float]:
        t0 = time.time()
        if not self.is_loaded():
            self.load()
            
        # Stage 1: Wide Retrieve using prefix index + rapidfuzz batch
        query_norm = self._normalize_text(name)
        if not query_norm:
            return [], 0.0

        query_words = query_norm.split()
        candidate_indices = set()
        
        # Extract candidates for each word in the query to handle code prefixes (e.g. "RS-PNTOP")
        for word in query_words:
            # Skip pure numbers to avoid pulling too many unrelated items
            if word.isdigit() or re.match(r'^\d+\.?\d*$', word):
                continue
                
            prefix_key = word[:3]
            
            # 1. Exact match for prefix_key if it exists (instant O(1) retrieval)
            if prefix_key in self._prefix_index:
                candidate_indices.update(self._prefix_index[prefix_key])
                
            # 2. Fuzzy match prefix_key against prefix keys (handles typos)
            similar_keys = process.extract(prefix_key, self._prefix_index.keys(), scorer=fuzz.ratio, limit=40, score_cutoff=60)
            for key, score, _ in similar_keys:
                candidate_indices.update(self._prefix_index[key])

            # 3. Same for company prefix index
            if self._company_prefix_index:
                if prefix_key in self._company_prefix_index:
                    candidate_indices.update(self._company_prefix_index[prefix_key])
                    
                similar_comp_keys = process.extract(prefix_key, self._company_prefix_index.keys(), scorer=fuzz.ratio, limit=40, score_cutoff=60)
                for key, score, _ in similar_comp_keys:
                    candidate_indices.update(self._company_prefix_index[key])

            # Fallback: if prefix_key is short (less than 3 chars), do startswith matching
            if len(prefix_key) < 3:
                for key in self._prefix_index:
                    if key.startswith(prefix_key):
                        candidate_indices.update(self._prefix_index[key])
                for key in self._company_prefix_index:
                    if key.startswith(prefix_key):
                        candidate_indices.update(self._company_prefix_index[key])

        # Score only the candidate subset using pre-computed normalized names
        candidates = []
        for idx in candidate_indices:
            norm_name = self.name_corpus_normalized[idx]
            score = fuzz.token_set_ratio(query_norm, norm_name)
            if score >= 50:
                candidates.append((self.catalog[idx], score, idx))

        # Sort and take top 120 candidates for re-ranking
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:120]

        # Stage 2: Lexicographical Re-ranking (uses pre-computed tokens)
        query_tokens = self._meili_tokenize(name)
        scored_candidates = []

        for item, stage1_score, idx in candidates:
            item_tokens = self.name_corpus_tokens[idx]
            meili_score = self._meili_score(query_tokens, item_tokens)
            
            # Pack score calculations
            pack_score = 50.0
            if pack and item["pack"]:
                q_pack_norm = self._normalize_text(pack)
                i_pack_norm = self._normalize_text(item["pack"])
                if q_pack_norm == i_pack_norm:
                    pack_score = 100.0
                elif q_pack_norm in i_pack_norm or i_pack_norm in q_pack_norm:
                    pack_score = 80.0
                else:
                    # Cross-format fallback: "10S" vs "strip of 10 tablets" both → ("10","tab")
                    q_qty, q_unit = self._parse_pack(pack)
                    i_qty, i_unit = self._parse_pack(item["pack"])
                    if q_qty and i_qty and q_qty == i_qty:
                        if not q_unit or not i_unit or q_unit == i_unit:
                            pack_score = 90.0  # qty + unit both agree across formats
                    
            # Brand alignment checks (apply 35% penalty to stage 1 score on brand mismatch)
            brand_penalty = 1.0
            q_brand = query_tokens[0] if query_tokens else ""
            if q_brand and len(q_brand) <= 2 and len(query_tokens) > 1:
                q_brand = query_tokens[1]
                
            if q_brand and len(q_brand) > 2:
                c_brand_clean = self._normalize_text(item["compname"]) if item["compname"] else ""
                c_name_clean = self._normalize_text(item["name"])
                # Use fuzzy match for typos in brand
                brand_in_name = (
                    q_brand in c_name_clean or 
                    q_brand in c_brand_clean or
                    any(fuzz.ratio(q_brand, w) >= 75 for w in c_name_clean.split()) or
                    (c_brand_clean and any(fuzz.ratio(q_brand, w) >= 75 for w in c_brand_clean.split()))
                )
                if not brand_in_name:
                    brand_penalty = 0.65

            # Company name matching — boost same company, penalize clear mismatches
            company_bonus = 0.0
            if compname and compname.strip():
                q_comp_norm = self._normalize_text(compname)
                c_comp_norm = self._normalize_text(item["compname"]) if item["compname"] else ""
                if q_comp_norm and c_comp_norm:
                    comp_score = fuzz.partial_ratio(q_comp_norm, c_comp_norm)
                    if comp_score >= 60:
                        company_bonus = 10.0  # Same company boost
                    elif comp_score < 30:
                        company_bonus = -15.0  # Clear mismatch penalty
                    
            name_score = stage1_score * brand_penalty
            final_composite_score = (name_score * 0.75) + (pack_score * 0.25) + company_bonus
            
            scored_candidates.append({
                "item": item,
                "meili_score_tuple": meili_score,
                "final_score": final_composite_score,
                "name_score": name_score,
                "pack_score": pack_score
            })
            
        # Final Sort: prioritize final_score (fuzzy token ratio + pack) over raw meili_score_tuple
        scored_candidates.sort(
            key=lambda x: (x["final_score"], x["meili_score_tuple"]),
            reverse=True
        )
        
        # Format output
        results = []
        for i, sc in enumerate(scored_candidates[:top_k], 1):
            item = sc["item"]
            results.append({
                "rank": i,
                "code": item["code"],
                "name": item["name"],
                "compname": item["compname"],
                "pack": item["pack"],
                "strength": item["strength"],
                "name_score": round(sc["name_score"], 1),
                "pack_score": round(sc["pack_score"], 1),
                "final_score": round(sc["final_score"], 1)
            })
            
        elapsed_ms = round((time.time() - t0) * 1000, 2)
        return results, elapsed_ms
