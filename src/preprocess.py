from data_loader import load_csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(file_path: str) -> pd.DataFrame:
    
    """
    Load data, vectorize descriptions, encode targets, and split into train/test.

    Args:
        file_path: Path to the CSV dataset.

    Returns:
        Original DataFrame. (Note: in practice you would return/persist the splits and fitted objects.)
    """
    
    df = load_csv(file_path)
    
     # TF-IDF features from description text
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(df["description"])
    
    # Encode labels with separate encoders (one per column)
    lang_enc = LabelEncoder()
    y_lang = lang_enc.fit_transform(df["language"])
    
    fw_enc = LabelEncoder()
    y_fram = fw_enc.fit_transform(df["framework"])
    
    # Two-task target matrix: (n_samples, 2)
    y = np.column_stack((y_lang, y_fram))
    
    # Optional auxiliary text kept for API responses / explainability
    reasons = df.get("reason", pd.Series([""] * len(df))).astype(str).values
    
    # Preserve joint distribution of (language|framework) during split
    stratify_key = df["language"].astype(str) + "|" + df["framework"].astype(str)
    
     # Split features, targets, and reasons in one aligned call
    X_train, X_test, y_train, y_test, reasons_train, reasons_test = train_test_split(
        x, y, reasons,
        test_size=0.2,
        random_state=42,
        stratify=stratify_key
    )
    
    return df


#if file is executed directly
if __name__ == "__main__":
        process = preprocess_data("./data/dataset.csv")
    

    