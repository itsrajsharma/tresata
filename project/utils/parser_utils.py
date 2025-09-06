import phonenumbers
import re
import os
from phonenumbers import geocoder
from typing import List, Tuple
from project.utils.data_loader import GLOBAL_LEGAL_SUFFIXES_SET, GLOBAL_COUNTRIES_SET

class ParserUtils:
    def __init__(self):
        self.countries_set = GLOBAL_COUNTRIES_SET
        
        # --- Legal Suffix Regex Setup (from part b/classifier.py) ---
        suffixes = {s.strip() for s in GLOBAL_LEGAL_SUFFIXES_SET if s.strip()}
        flexible_patterns = []
        for s in suffixes:
            core = "".join(filter(str.isalnum, s))
            if not core:
                continue
            pattern = r"\s*\.?\s*".join(list(core)) # Escape backslashes for string literal
            flexible_patterns.append(pattern)
        
        flexible_patterns = sorted(flexible_patterns, key=len, reverse=True)
        self.legal_suffix_regex = re.compile(
            r"(?<!\w)(" + "|".join(flexible_patterns) + r")((?!\\w)|[.,!?;:\s]*)", # Adjusted regex for non-word boundary after suffix, allowing trailing punctuation/whitespace
            re.IGNORECASE
        )

    _WORD_RE = re.compile(r'\S+')  # tokens: contiguous non-whitespace sequences

    def _norm_token(self, tok: str) -> str:
        """Normalize a single token for comparison (lower, strip dots, normalize '&'/'and', remove surrounding punctuation)."""
        if not tok:
            return ""
        s = tok.lower()
        s = re.sub(r'\band\b', '&', s)       # unify "and" -> "&"
        s = s.replace('.', '')                # remove dots
        s = s.replace("'", "")                # remove apostrophes
        s = re.sub(r'[^a-z0-9& ]+', ' ', s)   # remove stray punctuation (keep & and alnum)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _tokenize_with_spans(self, s: str) -> List[Tuple[str, int, int]]:
        """
        Return list of (token_text, start_index, end_index) for original string s.
        Use regex to get token spans so we can map back to original string indices.
        """
        tokens = []
        for m in self._WORD_RE.finditer(s):
            tokens.append((m.group(0), m.start(), m.end()))
        return tokens

    def _prepare_suffix_lists(self, legal_suffixes: List[str]) -> List[Tuple[List[str], str]]:
        """
        Return a deduped list of normalized suffix token lists, sorted longest-first.
        Each suffix list is a list of normalized tokens (no punctuation).
        Also returns the original raw suffix string for precise slicing.
        """
        normalized_with_raw = []
        for raw_suf in legal_suffixes:
            if not raw_suf:
                continue
            toks = [t for t in re.split(r'\s+', raw_suf.strip()) if t]
            norm_toks = tuple(t for t in (self._norm_token(tok) for tok in toks) if t)
            if len(norm_toks) == 0:
                continue
            normalized_with_raw.append((list(norm_toks), raw_suf)) # Store as list for later mutability if needed, and raw string
        
        # Deduplicate and sort by number of tokens then joined-length (both descending)
        uniq_map = {}
        for norm_seq, raw_suf in normalized_with_raw:
            key = " ".join(norm_seq)
            if key not in uniq_map or len(raw_suf) > len(uniq_map[key][1]):
                uniq_map[key] = (norm_seq, raw_suf)
                
        uniq = list(uniq_map.values())
        uniq.sort(key=lambda item: (-len(item[0]), -len(" ".join(item[0]))))
        return uniq

    def _build_normalized_suffix_token_lists(self, legal_suffixes: List[str]) -> List[Tuple[List[str], str]]:
        """
        From raw suffix strings, build normalized token lists sorted by token count desc (longest first).
        Returns list of normalized token lists along with their original raw string.
        """
        normalized_with_raw = []
        for raw_suf in legal_suffixes:
            if not raw_suf:
                continue
            toks = [t for t in re.split(r'\s+', raw_suf.strip()) if t]
            norm_toks = [t for t in (self._norm_token(tok) for tok in toks) if t]
            if norm_toks:
                normalized_with_raw.append((norm_toks, raw_suf))
        
        # Deduplicate and sort by number of tokens then joined-length (both descending)
        uniq_map = {}
        for norm_seq, raw_suf in normalized_with_raw:
            key = " ".join(norm_seq)
            # Keep the raw suffix from the first occurrence for uniqueness, or the one that's longer if order matters
            if key not in uniq_map or len(raw_suf) > len(uniq_map[key][1]):
                uniq_map[key] = (norm_seq, raw_suf)
                
        uniq = list(uniq_map.values())
        uniq.sort(key=lambda item: (-len(item[0]), -len(" ".join(item[0]))))
        return uniq

    def _token_to_regex(self, tok: str) -> str:
        """
        Build a permissive regex for a single token that:
        - Matches the token case-insensitively
        - Allows optional dots after token letters (e.g., "pvt." vs "pvt")
        - Allows small punctuation around token
        """
        if tok == '&':
            # allow '&' or 'and' (case-insensitive), possibly with surrounding punctuation/spaces
            return r'(?:&|\band\b)'
        # escape token but allow optional dot(s) inside/after (e.g., "co." "gmbh.")
        esc = re.escape(tok)
        # allow optional dots immediately after token characters (common in abbreviations)
        # we will accept 0+ dots or spaces/punctuation right after token
        return esc + r'\.?'

    def parse_phone_number(self, number_str: str) -> Tuple[str, str]:
        try:
            p = phonenumbers.parse(str(number_str), "US")
            if phonenumbers.is_valid_number(p):
                country = geocoder.country_name_for_number(p, "en")
                number = str(p.national_number)
                return country, number
        except Exception:
            pass
        return "", str(number_str).strip()

    def parse_company_name(self, name_str: str) -> Tuple[str, str]:
        original_name = str(name_str).strip()
        matches = list(self.legal_suffix_regex.finditer(original_name))
        if matches:
            last_match = matches[-1]
            split_point = last_match.start()
            name = original_name[:split_point].strip().rstrip('.,').strip()
            legal = last_match.group(1).strip()
            # Also capture the trailing punctuation/whitespace that was matched by the second group in the regex
            trailing_punct_ws = last_match.group(2).strip()
            legal = legal + trailing_punct_ws if trailing_punct_ws else legal # Re-attach if present
            return name, legal
        return original_name, ""
