import pandas as pd
import argparse
import joblib
from pathlib import Path

# Import Classifier from Part A and ParserUtils from updated utils
from project.utils.classifier import Classifier
from project.utils.parser_utils import ParserUtils

MODEL_PATH = Path("models/classifier.pkl")
CONFIDENCE_THRESHOLD = 0.6 # Using a default threshold, can be adjusted

def main():
    parser = argparse.ArgumentParser(description="Parse Phone Number and Company Name columns from a CSV.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", "-o", default="output.csv", help="Path to the output CSV file.")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        print(f"[INFO] Successfully loaded '{args.input}'. Analyzing {len(df.columns)} columns...")
    except FileNotFoundError:
        print(f"[ERROR] Input file not found at '{args.input}'")
        return
    
    # Load the Part A Classifier
    if not MODEL_PATH.exists():
        print(f"[ERROR] Classifier model not found at {MODEL_PATH}. Run `python scripts/save_classifier.py` first.")
        return
    try:
        clf = joblib.load(MODEL_PATH)
        print(f"[INFO] Loaded classifier from {MODEL_PATH}.")
    except Exception as e:
        print(f"[ERROR] Failed to load classifier model: {e}")
        return

    # Instantiate ParserUtils
    parser_utils = ParserUtils()
    print("[INFO] Initialized ParserUtils.")

    best_columns = {
        "PhoneNumber": {"col_name": None, "score": 0.0},
        "CompanyName": {"col_name": None, "score": 0.0},
    }

    for col_name in df.columns:
        try:
            label, conf = clf.classify_column(df[col_name])
            print(f"[INFO] Column '{col_name}' classified as {label} with confidence {conf:.2f}")
            if label == "PhoneNumber" and conf > best_columns["PhoneNumber"]["score"]:
                best_columns["PhoneNumber"]["score"] = conf
                best_columns["PhoneNumber"]["col_name"] = col_name
            elif label == "CompanyName" and conf > best_columns["CompanyName"]["score"]:
                best_columns["CompanyName"]["score"] = conf
                best_columns["CompanyName"]["col_name"] = col_name
        except Exception as e:
            print(f"[WARN] Error classifying column '{col_name}': {e}")
    
    phone_info = best_columns["PhoneNumber"]
    company_info = best_columns["CompanyName"]

    print(f"[INFO] Best candidate for Phone Number: '{phone_info['col_name']}' (Score: {phone_info['score']:.2f})")
    print(f"[INFO] Best candidate for Company Name: '{company_info['col_name']}' (Score: {company_info['score']:.2f})")
    
    results = {}
    
    if phone_info['col_name'] and phone_info['score'] >= CONFIDENCE_THRESHOLD:
        col = phone_info['col_name']
        results['original_phone_number'] = df[col]
        parsed_phones = df[col].apply(parser_utils.parse_phone_number)
        results['parsed_country'] = parsed_phones.str[0]
        results['parsed_phone_number'] = parsed_phones.str[1]

    if company_info['col_name'] and company_info['score'] >= CONFIDENCE_THRESHOLD:
        col = company_info['col_name']
        results['original_company_name'] = df[col]
        parsed_companies = df[col].apply(parser_utils.parse_company_name)
        results['parsed_company_name'] = parsed_companies.str[0]
        results['parsed_legal_suffix'] = parsed_companies.str[1]
    
    if results:
        final_column_order = [
            'original_phone_number', 'parsed_country', 'parsed_phone_number',
            'original_company_name', 'parsed_company_name', 'parsed_legal_suffix'
        ]
        
        output_df = pd.DataFrame(results)
        
        existing_columns_in_order = [col for col in final_column_order if col in output_df.columns]
        output_df = output_df[existing_columns_in_order]

        output_df.to_csv(args.output, index=False)
        print(f"\n[SUCCESS] Processing complete. Detailed '{args.output}' has been generated.")
    else:
        print(f"\n[WARN] No columns met the {CONFIDENCE_THRESHOLD:.0%} confidence threshold. No output generated.")

if __name__ == "__main__":
    main() 