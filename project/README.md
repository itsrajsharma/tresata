# Semantic Column Classifier and Parser

## Project Overview

This project provides an autonomous pipeline to classify CSV columns by semantic type (Phone Number, Company Name, Country, Date, Other) based on values only. It then parses identified Phone Number and Company Name columns into normalized fields and writes the output to a new CSV file.

## Structure

```
project/
├── predict.py          # CLI tool: classifies a given column
├── parser.py           # CLI tool: orchestrates classification, parsing, and output generation
├── utils/
│   ├── classifier.py   # Feature extraction and ML/rule-based classification logic
│   ├── parser_utils.py # Phone and company parsing/normalization utilities
│   └── __init__.py
├── data/               # Training and dictionary files
│   ├── Company.csv
│   ├── Countries.txt
│   ├── Dates.csv
│   ├── phoneNumber.csv
│   └── legal.txt
├── tests/              # Unit tests
│   ├── test_predict.py
│   ├── test_parser.py
│   └── __init__.py
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <repository_url>
    cd tresata/project
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### `predict.py`

Classify a single column in a CSV file:

```bash
python3 predict.py --input data/Company.csv --column company
```

Output: `CompanyName 0.93` (example)

### `parser.py`

Detect column types for all columns, parse Phone Number and Company Name columns (if detected with high confidence), and produce `output.csv`:

```bash
python3 parser.py --input data/test.csv
```

Output: `output.csv` with original and parsed fields (e.g., `PhoneNumber`, `PhoneNumber_Country`, `PhoneNumber_Number`, `CompanyName`, `CompanyName_Name`, `CompanyName_Legal`).

### Model Serialization

To save the trained classifier model:

```bash
python3 scripts/save_classifier.py
```

This will create `models/classifier.pkl`.

### Classify all columns (diagnostic)

To classify all columns in a CSV and view results:

```bash
python3 scripts/show_all_columns.py --input data/test.csv
```

## Testing

To run the unit tests:

```