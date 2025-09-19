import pandas as pd

"""
Loads CSV file into pandas DataFrame

Args: 
file_path (str): path to a CSV file

Returns: 
pd.DataFrame: the loaded data
"""
def load_csv(file_path : str) -> pd.DataFrame:
    
    df = pd.read_csv(file_path)
    
    return df


#if this file is executed directly
if __name__ == "__main__": 
    df = load_csv("./data/dataset.csv")
    print(df.head())
