import re
import os
import pandas as pd
from utils.data_loader import GLOBAL_COUNTRIES_SET, GLOBAL_LEGAL_SUFFIXES_SET

class FeatureExtractor:
    def __init__(self):
        self.countries_set = GLOBAL_COUNTRIES_SET
        self.legal_suffixes = GLOBAL_LEGAL_SUFFIXES_SET
        
        # Regex patterns
        self.phone_regexes = [
            re.compile(r"^\+\d{1,3}\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$"), # +CC (XXX) XXX-XXXX or +CC XXX-XXX-XXXX
            re.compile(r"^\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$"),           # (XXX) XXX-XXXX or XXX-XXX-XXXX (US-like)
            re.compile(r"^\d{4}\s?\d{6,7}$"),                                # XXXX XXXXXX/XXXXXXX (Indian-like)
            re.compile(r"^\+\d{1,3}\s?\d{6,10}$"),                           # +CC XXXXXXXXXX (International simple)
            re.compile(r"^\(?\d+\)?[\s.-]?\d+[\s.-]?\d+$") # More general phone pattern
        ]
        self.date_regexes = [
            re.compile(r"^\d{4}-\d{2}-\d{2}$"),                               # YYYY-MM-DD
            re.compile(r"^\d{2}/\d{2}/\d{4}$"),                               # DD/MM/YYYY
            re.compile(r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},\s\d{4}$", re.IGNORECASE), # Month DD, YYYY
            re.compile(r"^\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}$", re.IGNORECASE), # DD Mon YYYY
            re.compile(r"^\d{1,2}(?:st|nd|rd|th)?\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}$", re.IGNORECASE), # DDth Mon YYYY
            re.compile(r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2}(?:st|nd|rd|th)?,\s\d{4}$", re.IGNORECASE), # Mon DDth, YYYY
            re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$"),                               # DD-MM-YYYY or MM-DD-YYYY
            re.compile(r"^\d{4}\.\d{2}\.\d{2}$"),                               # YYYY.MM.DD
            re.compile(r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s\d{1,2}(?:st|nd|rd|th)?\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}$", re.IGNORECASE), # Day DD Mon YYYY
            re.compile(r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2}(?:st|nd|rd|th)?\s\d{4}$", re.IGNORECASE), # Day Mon DD YYYY
            re.compile(r"^\d{1,2}/\d{1,2}/\d{2}$"), # MM/DD/YY or DD/MM/YY
            re.compile(r"\b(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?[\s,]*\d{4}\b", re.IGNORECASE) # More flexible date with text
        ]
        self.month_names = set([
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ])
        
        # Pre-process legal suffixes for efficient token-based lookup
        self.normalized_legal_suffix_token_lists = []
        for suffix in self.legal_suffixes:
            # Normalize suffix: remove common punctuation and split into lowercase tokens
            normalized_tokens = tuple(re.findall(r'\b\w+\b', suffix.lower()))
            if normalized_tokens:
                self.normalized_legal_suffix_token_lists.append(normalized_tokens)
        # Sort by length descending for greedy matching of longer suffixes first
        self.normalized_legal_suffix_token_lists.sort(key=len, reverse=True)

    def _load_countries(self, filepath):
        # This method is no longer needed as countries are loaded globally
        pass

    def _load_legal_suffixes(self, filepath):
        # This method is no longer needed as legal suffixes are loaded globally
        pass

    def _calculate_features_for_value(self, value):
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        
        features = {}
        
        # Deterministic features
        features['pct_numeric_chars'] = sum(c.isdigit() for c in value) / len(value) if value else 0
        features['has_plus_sign'] = '+' in value
        features['contains_parentheses'] = '(' in value or ')' in value
        features['contains_dash'] = '-' in value
        features['contains_slash'] = '/' in value
        
        tokens = re.findall(r'\b\w+\b', value.lower()) # Re-extract tokens after cleaning
        features['avg_token_len'] = sum(len(token) for token in tokens) / len(tokens) if tokens else 0
        features['num_unique_tokens'] = len(set(tokens))
        features['max_token_len'] = max(len(token) for token in tokens) if tokens else 0
        
        # Dictionary checks
        features['is_country'] = 1 if any(token in self.countries_set for token in tokens) else 0 # More flexible country check
        
        # Improved legal suffix check: token-based matching
        features['has_legal_suffix'] = 0
        if tokens: # Only proceed if there are tokens to check
            for suffix_tokens in self.normalized_legal_suffix_token_lists:
                # Check if the sequence of suffix_tokens appears at the end of the value's tokens
                if len(tokens) >= len(suffix_tokens) and tokens[len(tokens) - len(suffix_tokens):] == list(suffix_tokens):
                    features['has_legal_suffix'] = 1
                    break

        # Regex checks
        features['regex_phone_matches_count'] = sum(1 for regex in self.phone_regexes if regex.search(value))
        features['regex_date_matches_count'] = sum(1 for regex in self.date_regexes if regex.search(value))
        features['contains_month_names'] = 1 if any(month in value.lower() for month in self.month_names) else 0
        
        return features

    def extract_features(self, column_values: pd.Series) -> pd.DataFrame:
        """
        Extracts features from a pandas Series of column values.

        Args:
            column_values (pd.Series): A series containing the values of a column.

        Returns:
            pd.DataFrame: A DataFrame where each row is a feature vector for a value.
        """
        feature_list = [self._calculate_features_for_value(val) for val in column_values]
        return pd.DataFrame(feature_list)

class Classifier:
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
        # Placeholder for ML model (will be trained later)
        self.ml_model = None 

    def classify_column(self, column_values: pd.Series) -> tuple[str, float]:
        """
        Classifies a column based on its values and returns a semantic label and confidence.
        
        Args:
            column_values (pd.Series): The values of the column to classify.

        Returns:
            tuple[str, float]: A tuple containing the predicted label and a confidence score (0-1).
        """
        features_df = self.feature_extractor.extract_features(column_values)
        
        # Heuristic-based classification
        phone_score = features_df['regex_phone_matches_count'].mean()
        # Modified date_score calculation to give more weight to pure regex matches
        date_score = features_df['regex_date_matches_count'].mean() * 0.8 + features_df['contains_month_names'].mean() * 0.2
        country_score = features_df['is_country'].mean()
        company_score = features_df['has_legal_suffix'].mean()
        
        scores = {
            "PhoneNumber": phone_score,
            "Date": date_score,
            "Country": country_score,
            "CompanyName": company_score,
        }
        
        # Find the label with the maximum score
        best_label = "Other"
        max_score = 0.0
        
        for label, score in scores.items():
            if score > max_score:
                max_score = score
                best_label = label
        
        # If all scores are 0, default to "Other" with 0.0 confidence (or initial 0.5)
        if max_score == 0.0: # If no specific feature was detected at all
            return "Other", 0.5 # Default confidence when no specific feature matches
            
        return best_label, max_score 