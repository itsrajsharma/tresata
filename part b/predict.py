# predict.py

import pandas as pd
import argparse
from classifier import classify_column

# Added "phone number" with a space for more flexibility
COLUMN_ALIASES = {
    "Phone Number": ["ph_nb", "number", "phone", "phone_number", "phone number", "mobile", "contact"],
    "Company Name": ["company", "company_name", "org", "organization", "firm"],
    "Country": ["country", "nation", "location"],
    "Date": ["date", "dob", "created_at", "time"]
}

def find_best_column(df, requested_column):
    """Finds the best matching column in the dataframe using aliases."""
    if requested_column in df.columns:
        return requested_column

    # Try alias matching
    req_col_lower = requested_column.lower()
    for aliases in COLUMN_ALIASES.values():
        if req_col_lower in aliases:
            for alias in aliases:
                if alias in df.columns:
                    print(f"[INFO] Using '{alias}' instead of '{requested_column}'")
                    return alias
    
    # Try case-insensitive match as a fallback
    for col in df.columns:
        if col.lower() == req_col_lower:
            return col
            
    return None


def main():
    """Main function to parse arguments and predict the semantic type of a column."""
    parser = argparse.ArgumentParser(description="Classify the semantic type of a CSV column.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input CSV file")
    parser.add_argument("--column", "-c", required=True, help="Name of the column to classify")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.input}")
        return

    target_col = find_best_column(df, args.column)
    if not target_col:
        print(f"[ERROR] Column '{args.column}' not found. Available columns: {list(df.columns)}")
        return

    semantic_type = classify_column(df[target_col])
    print(f"[RESULT] Column '{target_col}' classified as: {semantic_type}")


if __name__ == "__main__":
    main()