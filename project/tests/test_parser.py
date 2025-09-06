import pytest
import pandas as pd
from project.utils.parser_utils import ParserUtils
import os

@pytest.fixture
def parser_utils():
    # ParserUtils now loads global resources, so data_dir is not needed
    return ParserUtils()

def test_parse_phone_number_e164(parser_utils):
    # E.164 format with country code
    phone_str = "+14752162114"
    parsed = parser_utils.parse_phone_number(phone_str)
    assert "US" in parsed["Country"] or "United States" in parsed["Country"] or "Connecticut" in parsed["Country"]
    assert parsed["Number"] == "+14752162114"

def test_parse_phone_number_us_format(parser_utils):
    # US format with parentheses and dashes
    phone_str = "(475) 216-2114"
    parsed = parser_utils.parse_phone_number(phone_str, default_region="US")
    assert "US" in parsed["Country"] or "United States" in parsed["Country"] or "Connecticut" in parsed["Country"]
    assert parsed["Number"] == "+14752162114" # Should normalize to E.164

def test_parse_phone_number_international_simple(parser_utils):
    # Simple international without + but with inferred region
    phone_str = "080 1234 5678"
    parsed = parser_utils.parse_phone_number(phone_str, default_region="IN") # Assuming default_region for India
    assert "IN" in parsed["Country"] or "India" in parsed["Country"]
    assert parsed["Number"] == "+918012345678" # Example E.164 for India

def test_parse_phone_number_no_match(parser_utils):
    # Random string that is not a phone number
    phone_str = "Not a phone number string"
    parsed = parser_utils.parse_phone_number(phone_str)
    assert parsed["Country"] == ""
    assert parsed["Number"] == ""

def test_parse_company_name_with_suffix(parser_utils):
    company_str = "Tresata pvt ltd."
    name, legal = parser_utils.parse_company_name(company_str)
    assert name.lower() == "tresata"
    assert legal.lower() == "pvt ltd."

def test_parse_company_name_multi_word_suffix(parser_utils):
    company_str = "Enno Roggemann GmbH & Co. KG"
    name, legal = parser_utils.parse_company_name(company_str)
    assert name.lower() == "enno roggemann"
    assert legal.lower() == "gmbh & co. kg"

def test_parse_company_name_no_suffix(parser_utils):
    company_str = "First National Bank"
    name, legal = parser_utils.parse_company_name(company_str)
    assert name.lower() == "first national bank"
    assert legal == ""

def test_parse_company_name_empty_string(parser_utils):
    company_str = ""
    name, legal = parser_utils.parse_company_name(company_str)
    assert name == ""
    assert legal == "" 