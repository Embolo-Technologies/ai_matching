"""
Pharmaceutical Reference Data
=============================
Centralized formulation categories and generic ↔ brand name mappings
used by the matching engine for accuracy improvements.
"""

import re

# ──────────────────────────────────────────────────────────────────────────────
# Formulation Categories
# ──────────────────────────────────────────────────────────────────────────────
# Each category is a set of keywords. Items within the same category are
# compatible; items across incompatible categories should NOT match.

FORMULATION_CATEGORIES = {
    "injectable": {
        "inj", "injection", "injections", "vial", "vials", "ampoule",
        "ampule", "amp", "amps", "iv", "im", "sc", "prefilled",
        "syringe", "infusion",
    },
    "solid_oral": {
        "tab", "tabs", "tablet", "tablets", "caplet", "caplets",
        "cap", "caps", "capsule", "capsules", "softgel", "softgels",
        "chewable", "dispersible", "dt", "sr", "xr", "er", "cr", "mr",
        "xl", "la", "retard",
    },
    "liquid_oral": {
        "syp", "syrup", "syrups", "susp", "suspension", "suspensions",
        "sol", "soln", "solution", "solutions", "oral", "elixir",
        "emulsion", "drops", "drop", "oral drops", "paediatric drops",
        "ped drops", "liquid", "liq",
    },
    "topical": {
        "cream", "creams", "ointment", "ointments", "gel", "gels",
        "lotion", "lotions", "paste", "spray", "sprays", "foam",
        "powder", "dusting powder", "topical", "crm", "oint",
    },
    "eye_ear_nasal": {
        "eye drop", "eye drops", "ear drop", "ear drops",
        "nasal drop", "nasal drops", "nasal spray", "nasal",
        "ophthalmic", "otic", "eardrop", "eyedrop",
    },
    "inhaler_respiratory": {
        "inhaler", "inhalers", "rotacap", "rotacaps", "respule",
        "respules", "nebuliser", "nebulizer", "mdi", "dpi",
        "transcap", "autohaler", "inh", "inhalation", "inhalations",
    },
    "suppository_rectal": {
        "suppository", "suppositories", "supp", "rectal", "enema",
    },
}

# Which categories are INCOMPATIBLE with each other (cross-match = error)
INCOMPATIBLE_FORMULATIONS = {
    ("solid_oral", "liquid_oral"),
    ("solid_oral", "injectable"),
    ("solid_oral", "topical"),
    ("solid_oral", "eye_ear_nasal"),
    ("solid_oral", "inhaler_respiratory"),
    ("solid_oral", "suppository_rectal"),
    ("liquid_oral", "injectable"),
    ("liquid_oral", "topical"),
    ("liquid_oral", "inhaler_respiratory"),
    ("injectable", "topical"),
    ("injectable", "eye_ear_nasal"),
    ("injectable", "inhaler_respiratory"),
    ("topical", "eye_ear_nasal"),
    ("topical", "inhaler_respiratory"),
    ("eye_ear_nasal", "inhaler_respiratory"),
}


def detect_formulation(text: str) -> str:
    """
    Detect the formulation category from text.
    Returns the category key or '' if no formulation detected.
    """
    text_lower = text.lower()
    # Replace separators/punctuation with spaces to clean tokens
    text_lower = text_lower.replace("-", " ").replace("/", " ").replace(".", " ").replace(",", " ").replace("(", " ").replace(")", " ")
    # Insert a space between letters and numbers to tokenize units (e.g. 10tab -> 10 tab, 500ml -> 500 ml)
    text_lower = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', text_lower)
    text_lower = re.sub(r'([a-zA-Z]+)(\d+)', r'\1 \2', text_lower)
    
    # Check multi-word keys first (e.g., "eye drop")
    for category, keywords in FORMULATION_CATEGORIES.items():
        for kw in sorted(keywords, key=len, reverse=True):  # longest first
            if " " in kw:
                if kw in text_lower:
                    return category
    # Then check single-word keys
    words = set(text_lower.split())
    for category, keywords in FORMULATION_CATEGORIES.items():
        if words & keywords:
            return category
    return ""


def formulations_compatible(text1: str, text2: str) -> bool:
    """
    Check if two product texts have compatible formulations.
    text1 is the query (vendor input), text2 is the candidate name/pack.
    """
    cat1 = detect_formulation(text1)
    cat2 = detect_formulation(text2)

    # If query does not specify a formulation, it is compatible with anything
    if not cat1:
        return True

    # If query specifies a formulation, but candidate does not, default candidate to solid_oral
    if not cat2:
        cat2 = "solid_oral"

    if cat1 == cat2:
        return True

    return (cat1, cat2) not in INCOMPATIBLE_FORMULATIONS and (cat2, cat1) not in INCOMPATIBLE_FORMULATIONS


# ──────────────────────────────────────────────────────────────────────────────
# Generic ↔ Brand Name Mappings (Indian Pharmacy)
# ──────────────────────────────────────────────────────────────────────────────
# Each entry maps known brand names to their generic/chemical name(s).
# This helps match "AUGMENTIN 625" to "AMOXYCLAV 625" etc.

BRAND_GENERIC_MAP = {
    # Antibiotics
    "augmentin": ["amoxyclav", "amoxicillin clavulanate", "amox clav"],
    "amoxyclav": ["augmentin"],
    "moxclav": ["amoxyclav", "augmentin"],
    "clavam": ["amoxyclav", "augmentin"],
    "azithral": ["azithromycin", "azee", "zithromax"],
    "azee": ["azithromycin", "azithral"],
    "zithromax": ["azithromycin", "azithral", "azee"],
    "cifran": ["ciprofloxacin", "ciplox"],
    "ciplox": ["ciprofloxacin", "cifran"],
    "monocef": ["ceftriaxone", "cefaxone", "oframax"],
    "cefaxone": ["ceftriaxone", "monocef"],
    "taxim": ["cefotaxime"],
    "ceftas": ["ceftazidime"],
    "meronem": ["meropenem"],
    "levoflox": ["levofloxacin", "levoday", "levomac"],
    "levoday": ["levofloxacin", "levoflox"],

    # Pain / Anti-inflammatory
    "crocin": ["paracetamol", "dolo", "calpol", "pacimol", "tylenol"],
    "dolo": ["paracetamol", "crocin", "calpol"],
    "calpol": ["paracetamol", "crocin", "dolo"],
    "pacimol": ["paracetamol", "crocin", "dolo"],
    "combiflam": ["ibuprofen paracetamol", "brufen plus", "imol plus"],
    "brufen": ["ibuprofen", "ibugesic"],
    "ibugesic": ["ibuprofen", "brufen"],
    "voveran": ["diclofenac", "dynapar", "reactin"],
    "dynapar": ["diclofenac", "voveran"],
    "ultracet": ["tramadol paracetamol"],
    "flexon": ["ibuprofen paracetamol"],

    # Gastro
    "pan": ["pantoprazole", "pantop", "pantocid"],
    "pantop": ["pantoprazole", "pan", "pantocid"],
    "pantocid": ["pantoprazole", "pan", "pantop"],
    "omez": ["omeprazole", "ocid"],
    "ocid": ["omeprazole", "omez"],
    "rablet": ["rabeprazole", "razo", "rabeloc"],
    "razo": ["rabeprazole", "rablet", "rabeloc"],
    "rabeloc": ["rabeprazole", "rablet", "razo"],
    "domperidone": ["domstal", "vomistop"],
    "domstal": ["domperidone", "vomistop"],
    "ondansetron": ["emeset", "ondem", "vomikind"],
    "emeset": ["ondansetron", "ondem"],
    "ondem": ["ondansetron", "emeset"],
    "pan d": ["pantoprazole domperidone", "pantocid d"],
    "pantocid d": ["pantoprazole domperidone", "pan d"],
    "rantac": ["ranitidine", "zinetac", "aciloc"],
    "zinetac": ["ranitidine", "rantac", "aciloc"],

    # Cardiac / BP
    "ecosprin": ["aspirin", "disprin"],
    "disprin": ["aspirin", "ecosprin"],
    "atorva": ["atorvastatin", "lipitor", "tonact"],
    "tonact": ["atorvastatin", "atorva"],
    "telma": ["telmisartan", "telmikind", "telvas"],
    "telvas": ["telmisartan", "telma"],
    "amlong": ["amlodipine", "amlovas", "amlip"],
    "amlovas": ["amlodipine", "amlong"],
    "stamlo": ["amlodipine", "amlong"],
    "cardivas": ["carvedilol"],
    "metolar": ["metoprolol", "betaloc"],
    "betaloc": ["metoprolol", "metolar"],
    "concor": ["bisoprolol"],
    "clopitab": ["clopidogrel", "plavix", "clopilet"],
    "clopilet": ["clopidogrel", "clopitab", "plavix"],
    "plavix": ["clopidogrel", "clopitab", "clopilet"],
    "enoxarin": ["enoxaparin", "clexane"],
    "clexane": ["enoxaparin", "enoxarin"],

    # Diabetes
    "glycomet": ["metformin", "glucophage"],
    "glucophage": ["metformin", "glycomet"],
    "amaryl": ["glimepiride", "glimisave"],
    "glimisave": ["glimepiride", "amaryl"],
    "januvia": ["sitagliptin"],
    "galvus": ["vildagliptin"],
    "jardiance": ["empagliflozin"],
    "forxiga": ["dapagliflozin"],
    "mixtard": ["insulin human"],
    "lantus": ["insulin glargine"],
    "novorapid": ["insulin aspart"],
    "humalog": ["insulin lispro"],

    # Steroids / Anti-allergic
    "wysolone": ["prednisolone", "omnacortil"],
    "omnacortil": ["prednisolone", "wysolone"],
    "dexona": ["dexamethasone", "decadron"],
    "medrol": ["methylprednisolone"],
    "defcort": ["deflazacort", "calcort"],
    "allegra": ["fexofenadine"],
    "cetrizine": ["cetirizine", "cetzine", "okacet"],
    "cetzine": ["cetirizine", "okacet"],
    "okacet": ["cetirizine", "cetzine"],
    "montair": ["montelukast", "romilast"],
    "romilast": ["montelukast", "montair"],
    "avil": ["pheniramine"],
    "atarax": ["hydroxyzine"],

    # Respiratory
    "asthalin": ["salbutamol", "ventolin", "derihaler"],
    "ventolin": ["salbutamol", "asthalin"],
    "budecort": ["budesonide", "pulmicort"],
    "pulmicort": ["budesonide", "budecort"],
    "seroflo": ["salmeterol fluticasone", "aerocort"],
    "foracort": ["formoterol budesonide"],
    "duolin": ["ipratropium salbutamol", "combivent"],
    "tiova": ["tiotropium", "spiriva"],
    "spiriva": ["tiotropium", "tiova"],

    # Neuro / Psych
    "calmpose": ["diazepam", "valium"],
    "valium": ["diazepam", "calmpose"],
    "alzolam": ["alprazolam", "restyl", "alprax"],
    "restyl": ["alprazolam", "alzolam"],
    "alprax": ["alprazolam", "alzolam", "restyl"],
    "nexito": ["escitalopram", "cipralex", "stalopam"],
    "stalopam": ["escitalopram", "nexito"],
    "frisium": ["clobazam"],
    "valparin": ["valproate", "epilex"],
    "eptoin": ["phenytoin", "dilantin"],
    "tegrital": ["carbamazepine"],
    "gabapin": ["gabapentin"],
    "pregabalin": ["lyrica", "pregalin"],
    "pregalin": ["pregabalin", "lyrica"],
    "lyrica": ["pregabalin", "pregalin"],

    # Vitamins / Supplements
    "shelcal": ["calcium vitamin d3"],
    "calcimax": ["calcium", "shelcal"],
    "becosules": ["b complex", "beplex"],
    "beplex": ["b complex", "becosules"],
    "neurobion": ["vitamin b12 b6 b1", "methylcobalamin"],
    "methylcobalamin": ["mecobalamin", "methycobal", "neurobion forte"],
    "methycobal": ["methylcobalamin", "mecobalamin"],
    "limcee": ["vitamin c", "ascorbic acid"],
    "zincovit": ["multivitamin zinc"],
    "revital": ["multivitamin"],
    "supradyn": ["multivitamin"],
    "iron sucrose": ["venofer", "orofer"],
    "orofer": ["iron sucrose"],

    # Common combos
    "norflox tz": ["norfloxacin tinidazole", "nortini"],
    "metrogyl": ["metronidazole", "flagyl"],
    "flagyl": ["metronidazole", "metrogyl"],
    "oflox": ["ofloxacin", "zanocin"],
    "zanocin": ["ofloxacin", "oflox"],
    "o2": ["ofloxacin ornidazole"],
    "taxim o": ["cefixime ofloxacin"],
    "zifi": ["cefixime"],
    "sumo": ["nimesulide paracetamol"],
    "zerodol": ["aceclofenac"],
    "zerodol sp": ["aceclofenac serratiopeptidase paracetamol"],
    "hifenac": ["aceclofenac"],
}


def find_brand_aliases(brand_name: str) -> list:
    """
    Given a brand name, return all known aliases (generic + other brands).
    Returns empty list if no mapping found.
    """
    key = brand_name.lower().strip()
    return BRAND_GENERIC_MAP.get(key, [])


def are_brands_equivalent(brand1: str, brand2: str) -> bool:
    """
    Check if two brand names refer to the same drug.
    Returns True if they're known equivalents.
    """
    b1 = brand1.lower().strip()
    b2 = brand2.lower().strip()

    if b1 == b2:
        return True

    aliases1 = set(BRAND_GENERIC_MAP.get(b1, []))
    aliases2 = set(BRAND_GENERIC_MAP.get(b2, []))

    # Direct: b2 is in b1's aliases or vice versa
    if b2 in aliases1 or b1 in aliases2:
        return True

    # Transitive: they share a common alias
    if aliases1 & aliases2:
        return True

    return False
