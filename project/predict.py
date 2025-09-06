# predict.py
import argparse
import joblib
from pathlib import Path
import pandas as pd
import sys

MODEL_PATH = Path("models/classifier.pkl")

def load_column_values(path: Path, column_name: str): # Renamed `path` param from `Path` to `path` 
    df = pd.read_csv(path)
    if column_name not in df.columns:
        raise SystemExit(f"Column '{column_name}' not found in {path}")
    vals = df[column_name].astype(str).fillna("").tolist()
    return vals

def call_classifier_obj(clf, values):
    """
    Try to call the classifier object in preferred order:
    1) clf.classify(values) -> (label, confidence)
    2) clf.predict_proba / clf.predict with a wrapper (best-effort)
    """
    # 1) direct wrapper
    if hasattr(clf, "classify_column"):
        try:
            out = clf.classify_column(pd.Series(values))
            # normalize to (label, confidence)
            if isinstance(out, tuple) and len(out) >= 2:
                return out[0], float(out[1])
            # if it returns dict-like
            if isinstance(out, dict) and "label" in out and "confidence" in out:
                return out["label"], float(out["confidence"])
        except Exception:
            pass

    # 2) sklearn-like model that expects features -> try to find feature extractor
    # many wrappers store feature_extractor or expose a transform function
    fe = None
    if hasattr(clf, "feature_extractor"):
        fe = clf.feature_extractor
    try:
        from project.utils.classifier import FeatureExtractor
        if fe is None:
            fe = FeatureExtractor()
    except Exception:
        fe = fe  # may be None

    if fe is not None and hasattr(clf, "predict_proba"):
        try:
            # fe.transform should accept a list of column values and produce 1-d feature vector
            # Assuming FeatureExtractor.extract_features takes a pd.Series
            X = fe.extract_features(pd.Series(values))
            # ensure shape: (1, n_features) if model expects single sample
            import numpy as np
            # If X is already a DataFrame from extract_features, ensure it's ready for predict_proba
            if isinstance(X, pd.DataFrame):
                X = X.mean(axis=0).to_frame().T # Aggregate features if needed for single prediction

            probs = clf.predict_proba(X)[0]
            label = clf.classes_[probs.argmax()]
            conf = float(probs.max())
            return label, conf
        except Exception:
            pass

    # 3) last resort: if clf has predict on raw values
    if hasattr(clf, "predict"):
        try:
            # This path might need more specific handling based on how 'predict' is implemented for raw values
            out = clf.predict(pd.Series(values))
            if isinstance(out, (list, tuple)) and len(out) > 0:
                return out[0], 1.0
        except Exception:
            pass

    raise RuntimeError("Could not call classifier object: unsupported interface.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to CSV input file")
    p.add_argument("--column", required=True, help="Column name to classify")
    p.add_argument("--output-file", help="Optional path to write classification output to")
    args = p.parse_args()

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Run `python3 scripts/save_classifier.py` to create it.")
        sys.exit(1)

    clf = joblib.load(MODEL_PATH)

    vals = load_column_values(Path(args.input), args.column)
    try:
        label, conf = call_classifier_obj(clf, vals)
    except Exception as e:
        print("Error while calling classifier:", e)
        sys.exit(2)

    output_string = f"{label} {conf:.2f}"

    if args.output_file:
        try:
            with open(args.output_file, "w") as f:
                f.write(output_string + "\n")
            print(f"Classification output written to {args.output_file}")
        except IOError as e:
            print(f"Error writing to output file {args.output_file}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output_string)

if __name__ == "__main__":
    main() 