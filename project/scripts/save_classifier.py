# scripts/save_classifier.py
import pathlib
import sys
import joblib

# Import directly from project.utils
try:
    from project.utils.data_loader import (GLOBAL_COUNTRIES_SET, GLOBAL_LEGAL_SUFFIXES_SET, 
                                         GLOBAL_COMPANY_DF, GLOBAL_DATES_DF, GLOBAL_PHONE_NUMBERS_DF)
    from project.utils.classifier import FeatureExtractor, Classifier
except Exception as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure your PYTHONPATH is correctly set or run this script from the project root.")
    sys.exit(1)

def main():
    models_dir = pathlib.Path("models")
    models_dir.mkdir(exist_ok=True)

    # Instantiate FeatureExtractor first
    feature_extractor = FeatureExtractor()

    # Instantiate Classifier with the FeatureExtractor
    clf = Classifier(feature_extractor=feature_extractor) # Use the correct constructor for your Classifier

    # If classifier indicates it needs training, do not auto-train. Instead instruct user.
    # (Our current Classifier is rule-based and doesn't need explicit training, but keeping this check)
    if hasattr(clf, "needs_training") and getattr(clf, "needs_training"):
        print("Classifier indicates it needs training. Please run the training script (if provided) before saving.")
        sys.exit(2)

    out_path = models_dir / "classifier.pkl"
    joblib.dump(clf, out_path)
    print(f"Saved classifier to: {out_path.resolve()}")

if __name__ == "__main__":
    main() 