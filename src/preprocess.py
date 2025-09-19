from data_loader import load_csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorize

def preprocess_data(file_path: str) -> pd.DataFrame:
    df = load_csv(file_path)
    
    
    
    return df


#if file is executed directly
if __name__ == "__main__":
    process = preprocess_data("./data/dataset.csv")
    print(process.head())