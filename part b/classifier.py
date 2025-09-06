# classifier.py

import os
import re
import pandas as pd
import phonenumbers
from dateutil.parser import parse

# --- Setup: Robustly locate and load helper data ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    with open(os.path.join(DATA_DIR, "countries.txt"), "r", encoding="utf-8") as f:
        COUNTRY_SET = {line.strip().lower() for line in f if line.strip()}

    with open(os.path.join(DATA_DIR, "legal.txt"), "r", encoding="utf-8") as f:
        # **THIS IS THE FIX**: Build flexible regex patterns for each suffix
        
        # 1. Read all unique, stripped lines from legal.txt
        suffixes = {line.strip() for line in f if line.strip()}
        
        # 2. Create a flexible pattern for each suffix
        flexible_patterns = []
        for s in suffixes:
            # For a suffix like "s.r.o.", the core is "sro"
            core = "".join(filter(str.isalnum, s))
            if not core:
                continue
            
            # Create a pattern that allows optional spaces/periods between letters
            # e.g., "sro" -> "s\s*\.?\s*r\s*\.?\s*o"
            pattern = r"\s*\.?\s*".join(list(core))
            flexible_patterns.append(pattern)

        # 3. Sort patterns by length of their original suffix to match longer ones first
        flexible_patterns = sorted(flexible_patterns, key=len, reverse=True)
        
        # 4. Combine into the final master regex
        LEGAL_SUFFIX_REGEX = re.compile(
            # The pattern is now much more robust to spacing and periods
            r"(?<!\w)(" + "|".join(flexible_patterns) + r")(?!\w)",
            re.IGNORECASE
        )

except FileNotFoundError as e:
    print(f"[ERROR] Classifier setup failed. Could not find data file: {e.filename}")
    exit(1)

# (The rest of the file remains the same)
CONFIDENCE_THRESHOLD = 0.3

def _calculate_match_percentage(column: pd.Series, match_function) -> float:
    samples = column.dropna().head(100)
    if samples.empty: return 0.0
    matches = samples.apply(match_function).sum()
    return matches / len(samples)

def get_phone_score(column: pd.Series) -> float:
    phone_regex = re.compile(r'\d{7,}')
    def is_phone(val):
        s_val = str(val)
        try:
            p = phonenumbers.parse(s_val, "US")
            if phonenumbers.is_possible_number(p): return True
        except Exception: pass
        return phone_regex.search(s_val) is not None
    return _calculate_match_percentage(column, is_phone)

def get_date_score(column: pd.Series) -> float:
    def is_date(val):
        s_val = str(val)
        if len(s_val) > 35: return False
        try:
            parse(s_val, fuzzy=False)
            return True
        except (ValueError, TypeError, OverflowError): return False
    return _calculate_match_percentage(column, is_date)

def get_country_score(column: pd.Series) -> float:
    return _calculate_match_percentage(column, lambda val: str(val).lower() in COUNTRY_SET)

def get_company_score(column: pd.Series) -> float:
    """Uses the new, more robust regex to find company names."""
    def is_company(val):
        return LEGAL_SUFFIX_REGEX.search(str(val)) is not None
    return _calculate_match_percentage(column, is_company)

def get_all_scores(column: pd.Series) -> dict:
    col_as_str = column.astype(str)
    return {
        "Phone Number": get_phone_score(col_as_str),
        "Company Name": get_company_score(col_as_str),
        "Country": get_country_score(col_as_str),
        "Date": get_date_score(col_as_str),
    }

def classify_column(column: pd.Series) -> str:
    scores = get_all_scores(column)
    best_category = max(scores, key=scores.get)
    max_score = scores[best_category]
    if max_score >= CONFIDENCE_THRESHOLD:
        return best_category
    else:
        return "Other"