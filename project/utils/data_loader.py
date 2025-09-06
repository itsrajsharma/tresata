import pandas as pd
import os

# Define the base data directory
DATA_DIR = "project/data"

def load_countries(filepath=os.path.join(DATA_DIR, "Countries.txt")) -> set:
    """
    Loads country names from Countries.txt into a set.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Countries file not found at {filepath}")
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        return {line.strip().lower() for line in f if line.strip()}

def load_legal_suffixes(filepath=os.path.join(DATA_DIR, "legal.txt")) -> set:
    """
    Loads legal suffixes from legal.txt into a set.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Legal suffixes file not found at {filepath}")
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        return {line.strip().lower() for line in f if line.strip()}

def load_company_data(filepath=os.path.join(DATA_DIR, "Company.csv")) -> pd.DataFrame:
    """
    Loads company examples from Company.csv into a Pandas DataFrame.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Company data file not found at {filepath}")
        return pd.DataFrame()
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading Company.csv: {e}")
        return pd.DataFrame()

def load_date_data(filepath=os.path.join(DATA_DIR, "Dates.csv")) -> pd.DataFrame:
    """
    Loads date examples from Dates.csv into a Pandas DataFrame.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Dates data file not found at {filepath}")
        return pd.DataFrame()
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading Dates.csv: {e}")
        return pd.DataFrame()

def load_phone_number_data(filepath=os.path.join(DATA_DIR, "phoneNumber.csv")) -> pd.DataFrame:
    """
    Loads phone number examples from phoneNumber.csv into a Pandas DataFrame.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Phone number data file not found at {filepath}")
        return pd.DataFrame()
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading phoneNumber.csv: {e}")
        return pd.DataFrame()

# Global resources
GLOBAL_COUNTRIES_SET = load_countries()
GLOBAL_LEGAL_SUFFIXES_SET = load_legal_suffixes()
GLOBAL_COMPANY_DF = load_company_data()
GLOBAL_DATES_DF = load_date_data()
GLOBAL_PHONE_NUMBERS_DF = load_phone_number_data() 