import pytest
import pandas as pd
from project.utils.classifier import FeatureExtractor, Classifier

@pytest.fixture
def feature_extractor():
    # FeatureExtractor now loads global resources, so data_dir is not needed
    return FeatureExtractor()

@pytest.fixture
def classifier(feature_extractor):
    return Classifier(feature_extractor)

def test_feature_extraction_phone_number(feature_extractor):
    data = pd.Series(["+1 475-216-2114", "(080) 1234 5678", "4853859590", "Not a phone"]) # Added a non-phone entry
    features_df = feature_extractor.extract_features(data)
    
    # For phone numbers, we expect high regex_phone_matches_count and has_plus_sign for some
    assert features_df.loc[0, 'regex_phone_matches_count'] == 1
    assert features_df.loc[0, 'has_plus_sign'] == True
    assert features_df.loc[1, 'regex_phone_matches_count'] == 1
    assert features_df.loc[1, 'contains_parentheses'] == True
    assert features_df.loc[2, 'regex_phone_matches_count'] >= 1 # Could match the 10-digit regex variant
    assert features_df.loc[3, 'regex_phone_matches_count'] == 0

def test_feature_extraction_company_name(feature_extractor):
    data = pd.Series(["Tresata pvt ltd.", "Enno Roggemann GmbH & Co. KG", "First National Bank", "Just a name"])
    features_df = feature_extractor.extract_features(data)
    
    # For company names, we expect has_legal_suffix for some
    assert features_df.loc[0, 'has_legal_suffix'] == 1
    assert features_df.loc[1, 'has_legal_suffix'] == 1
    assert features_df.loc[2, 'has_legal_suffix'] == 0 # 'Bank' is not in legal.txt
    assert features_df.loc[3, 'has_legal_suffix'] == 0

def test_feature_extraction_date(feature_extractor):
    data = pd.Series(["2023-01-15", "12/25/2024", "February 14, 2023", "Not a date"])
    features_df = feature_extractor.extract_features(data)
    
    # For dates, we expect high regex_date_matches_count and contains_month_names for some
    assert features_df.loc[0, 'regex_date_matches_count'] == 1
    assert features_df.loc[1, 'regex_date_matches_count'] == 1
    assert features_df.loc[2, 'regex_date_matches_count'] == 1
    assert features_df.loc[2, 'contains_month_names'] == 1
    assert features_df.loc[3, 'regex_date_matches_count'] == 0
    assert features_df.loc[3, 'contains_month_names'] == 0

def test_classifier_phone_number(classifier):
    data = pd.Series(["+1 475-216-2114", "(080) 1234 5678", "9876543210"])
    label, confidence = classifier.classify_column(data)
    assert label == "PhoneNumber"
    assert confidence > 0.8

def test_classifier_company_name(classifier):
    data = pd.Series(["Tresata pvt ltd.", "Enno Roggemann GmbH & Co. KG"])
    label, confidence = classifier.classify_column(data)
    assert label == "CompanyName"
    assert confidence > 0.7

def test_classifier_date(classifier):
    data = pd.Series(["2023-01-15", "12/25/2024", "February 14, 2023"])
    label, confidence = classifier.classify_column(data)
    assert label == "Date"
    assert confidence > 0.7

def test_classifier_other(classifier):
    data = pd.Series(["Random text", "Another random string", "Just some words"])
    label, confidence = classifier.classify_column(data)
    assert label == "Other" 