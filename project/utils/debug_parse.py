from project.utils.parser_utils import ParserUtils
import re
import pathlib
from typing import List, Tuple

def debug_case(company, legal_suffixes, top_n=20):
    print("ORIG:", repr(company))
    print("--------------")
    
    # Instantiate ParserUtils for debugging
    # This allows us to use its internal helper methods and parse_company_name as intended
    parser_utils_instance = ParserUtils()
    
    # Temporarily override legal_suffixes_raw for this debug case if needed
    # In your solution, legal_suffixes are passed directly to the standalone parse_company_name function.
    # Since it is now a method, we need to explicitly rebuild its internal suffix lists for the debug case.
    parser_utils_instance.legal_suffixes_raw = legal_suffixes
    parser_utils_instance.suffix_token_lists = parser_utils_instance._prepare_suffix_lists(legal_suffixes)

    # show tokenization
    tokens = [m.group(0) for m in re.finditer(r'\S+', company)]
    print("TOKENS:", tokens)
    norm_tokens = [parser_utils_instance._norm_token(t) for t in tokens]
    print("NORM TOKENS:", norm_tokens)
    print("--------------")
    suf_lists = parser_utils_instance.suffix_token_lists
    print("TOP SUFFIX CANDIDATES (normalized):")
    for i, s in enumerate(suf_lists[:top_n]):
        print(f" {i+1}.", " ".join(s))
    print("--------------")
    
    print("parse_company_name result =>", parser_utils_instance.parse_company_name(company))
    print("--------------")
    # show attempted regex patterns for first few suffixes (how they look)
    for s in suf_lists[:10]:
        parts = [parser_utils_instance._token_to_regex(t) for t in s]
        sep = r'(?:[\s\.\,\-\/\&\(\)\\\u2013\u2014]*)'
        pat = sep.join(parts) + r'\s*$'
        print("SUFFIX PATTERN:", " ".join(s))
        print(" PATTERN:", pat)
        try:
            m = re.search(pat, company, flags=re.IGNORECASE)
            print("  matched?", bool(m))
            if m:
                print("  match.span():", m.span(), "matched_text:", repr(company[m.start():m.end()]))
        except re.error as e:
            print("  pattern error:", e)
    print("==============")

# Usage:
# legal = pathlib.Path('project/data/legal.txt').read_text().splitlines()
# debug_case("Tresata pvt ltd.", legal)
# debug_case("Enno Roggemann GmbH & Co. KG", legal) 