#!/usr/bin/env python3
"""
Show predicted semantic label + confidence for every column in a CSV using the saved classifier
(or by instantiating the in-memory Classifier if model pickle is absent).
Saves results to an optional CSV file.
"""

import argparse
import joblib
from pathlib import Path
import pandas as pd
import sys

MODEL_PATHS = [Path("models/classifier.pkl"), Path("models/classifier_rf.pkl")]
DEFAULT_INPUT = Path("data/test.csv")
DEFAULT_OUTPUT = Path("output/part_a_results.csv")

def load_model_or_fallback():
    # try to load a saved model first
    for p in MODEL_PATHS:
        if p.exists():
            try:
                return joblib.load(p)
            except Exception:
                continue
    # fallback: try to instantiate the in-memory Classifier
    try:
        from project.utils.data_loader import load_global_resources
    except Exception:
        try:
            from project.utils.data_loader import load_resources as load_global_resources
        except Exception:
            load_global_resources = None

    try:
        from project.utils.classifier import Classifier
    except Exception:
        Classifier = None

    if Classifier is None:
        raise RuntimeError("No classifier found: neither serialized model nor Classifier class available.")

    resources = load_global_resources() if load_global_resources is not None else None
    # First, try to instantiate FeatureExtractor
    fe = None
    try:
        fe = FeatureExtractor()
    except Exception as e:
        raise RuntimeError(f"Could not instantiate FeatureExtractor: {e}")
    
    try:
        clf = Classifier(feature_extractor=fe)
    except Exception as e:
        try:
            clf = Classifier(resources=resources) # Fallback to original attempts if feature_extractor fails
        except Exception:
            try:
                clf = Classifier(resources)
            except Exception:
                clf = Classifier()
    return clf

def call_classifier_obj(clf, values):
    # Prefer clf.classify_column(values) if available
    if hasattr(clf, "classify_column"):
        try:
            out = clf.classify_column(pd.Series(values))
            if isinstance(out, tuple) and len(out) >= 2:
                return out[0], float(out[1])
            if isinstance(out, dict) and "label" in out and "confidence" in out:
                return out["label"], float(out["confidence"])
        except Exception as e:
            pass

    # try feature-extractor + sklearn predict_proba
    fe = None
    if hasattr(clf, "feature_extractor"):
        fe = clf.feature_extractor
    try:
        from project.utils.classifier import FeatureExtractor
        if fe is None:
            fe = FeatureExtractor()
    except Exception as e:
        fe = fe

    if fe is not None and hasattr(clf, "predict_proba"):
        try:
            raise NotImplementedError("predict_proba path for rule-based classifier is not applicable.")
        except Exception as e:
            pass

    # last resort: clf.predict on raw values
    if hasattr(clf, "predict"):
        try:
            out = clf.predict([values])
            if isinstance(out, (list, tuple)) and len(out) > 0:
                return out[0], 1.0
        except Exception as e:
            pass

    return "Unknown", 0.0

def pretty_print_table(results):
    # results: list of (column, label, conf)
    max_col_len = max(len(r[0]) for r in results) if results else 6
    print(f"{str('COLUMN').ljust(max_col_len)} | LABEL         | CONF")
    print("-" * (max_col_len + 3 + 15 + 3 + 6))
    for col, label, conf in results:
        print(f"{str(col).ljust(max_col_len)} | {str(label).ljust(13)} | {conf:.2f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="Path to input CSV file")
    p.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Optional output CSV for results")
    p.add_argument("--no-save", action="store_true", help="Do not save results to CSV")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input CSV not found: {input_path}")
        sys.exit(1)

    try:
        clf = load_model_or_fallback()
    except Exception as e:
        print("Failed to load or construct classifier:", e)
        sys.exit(2)

    df = pd.read_csv(input_path, header=0)
    results = []
    for col in df.columns:
        vals = df[col].astype(str).fillna("").tolist()
        try:
            label, conf = call_classifier_obj(clf, vals)
        except Exception as e:
            label, conf = "Error", 0.0
            print(f"Error classifying column {col}: {e}")
        results.append((col, label, conf))

    pretty_print_table(results)

    if not args.no_save:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame(results, columns=["column", "label", "confidence"])
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved results to {out_path.resolve()}")

if __name__ == "__main__":
    main() 