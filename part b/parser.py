# parser.py

import pandas as pd
import argparse
import phonenumbers
from classifier import get_all_scores, LEGAL_SUFFIX_REGEX, CONFIDENCE_THRESHOLD

def parse_phone_number(number_str: str):
    try:
        p = phonenumbers.parse(str(number_str), "US")
        if phonenumbers.is_valid_number(p):
            country = phonenumbers.geocoder.country_name_for_number(p, "en")
            number = str(p.national_number)
            return country, number
    except Exception:
        pass
    return None, str(number_str).strip()

def parse_company_name(name_str: str):
    original_name = str(name_str).strip()
    matches = list(LEGAL_SUFFIX_REGEX.finditer(original_name))
    if matches:
        last_match = matches[-1]
        split_point = last_match.start()
        name = original_name[:split_point].strip().rstrip('.,').strip()
        legal = last_match.group(1)
        return name, legal
    return original_name, None

def main():
    parser = argparse.ArgumentParser(description="Parse Phone Number and Company Name columns from a CSV.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input CSV file.")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        print(f"[INFO] Successfully loaded '{args.input}'. Analyzing {len(df.columns)} columns...")
    except FileNotFoundError:
        print(f"[ERROR] Input file not found at '{args.input}'")
        return
    
    best_columns = {
        "Phone Number": {"col_name": None, "score": 0.0},
        "Company Name": {"col_name": None, "score": 0.0},
    }

    for col_name in df.columns:
        scores = get_all_scores(df[col_name])
        for sem_type in best_columns:
            if scores[sem_type] > best_columns[sem_type]["score"]:
                best_columns[sem_type]["score"] = scores[sem_type]
                best_columns[sem_type]["col_name"] = col_name
    
    phone_info = best_columns["Phone Number"]
    company_info = best_columns["Company Name"]

    print(f"[INFO] Best candidate for Phone Number: '{phone_info['col_name']}' (Score: {phone_info['score']:.2f})")
    print(f"[INFO] Best candidate for Company Name: '{company_info['col_name']}' (Score: {company_info['score']:.2f})")
    
    # **THIS IS THE FIX**: This section is restored to the previous, more detailed output format
    results = {}
    
    if phone_info['col_name'] and phone_info['score'] >= CONFIDENCE_THRESHOLD:
        col = phone_info['col_name']
        results['original_phone_number'] = df[col]
        parsed_phones = df[col].apply(parse_phone_number)
        results['parsed_country'] = parsed_phones.str[0]
        results['parsed_phone_number'] = parsed_phones.str[1]

    if company_info['col_name'] and company_info['score'] >= CONFIDENCE_THRESHOLD:
        col = company_info['col_name']
        results['original_company_name'] = df[col]
        parsed_companies = df[col].apply(parse_company_name)
        results['parsed_company_name'] = parsed_companies.str[0]
        results['parsed_legal_suffix'] = parsed_companies.str[1]
    
    if results:
        # Restore the previous, more detailed column order
        final_column_order = [
            'original_phone_number', 'parsed_country', 'parsed_phone_number',
            'original_company_name', 'parsed_company_name', 'parsed_legal_suffix'
        ]
        
        output_df = pd.DataFrame(results)
        
        existing_columns_in_order = [col for col in final_column_order if col in output_df.columns]
        output_df = output_df[existing_columns_in_order]

        output_df.to_csv('output.csv', index=False)
        print("\n[SUCCESS] Processing complete. Detailed 'output.csv' has been generated.")
    else:
        print(f"\n[WARN] No columns met the {CONFIDENCE_THRESHOLD:.0%} confidence threshold. No output generated.")

if __name__ == "__main__":
    main()