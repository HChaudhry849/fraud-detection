import pandas as pd
from pathlib import Path 
import os
import random

class DataPipeline:

    # .venv\Scripts\activate
    # Created virtual environment which acts as a container 
    
    def __init__(self):
        # --- SMART PATH LOGIC START ---
        BASE_DIR = Path(__file__).resolve().parent
        
        # If /data is in the same folder (Docker), use it. 
        # Otherwise, look one level up (Windows Local).
        if (BASE_DIR / "data").exists():
            self.data_root = BASE_DIR / "data"
        else:
            self.data_root = BASE_DIR.parent / "data"
            
        self.csv_path = self.data_root / "raw" / "raw_transactions.csv"
        # --- SMART PATH LOGIC END ---
    
    def validate_file(self):
        file_exists = os.path.exists(self.csv_path)

        # check if the file exists
        if file_exists:
            # get file size
            file_size = os.path.getsize(self.csv_path)
            # check file size
            if file_size > 0:
                # load file
                self.load_data()
            else:
                raise RuntimeError(f"File at {self.csv_path} is empty")
        else:
            raise RuntimeError(f"File does NOT exist at: {self.csv_path}")

            
    def load_data(self):
        try:
            self.data = pd.read_csv(self.csv_path)
            print(self.data.to_string())
        except Exception as e:
            raise RuntimeError(f"Unable to load file: {e}")
        
        return self.data
    
    def validate_schemas(self):
        # Define required columns
        REQUIRED_COLUMNS = {"timestamp", "amount", "location", "type", "is_fraud"}
        # Count the total number of columns required 
        TOTAL_REQUIRED_COLUMNS = len(REQUIRED_COLUMNS)
        # Standardise csv column names by lowercasing, removing whitespace, removing special characters, save into new list

        ACTUAL_COLUMNS = [i.lower().strip() for i in self.data.columns]     
        # Are all the  required columns in the dataset, compares to the standardised list
        if not REQUIRED_COLUMNS.issubset(ACTUAL_COLUMNS):
            # find which column is missing 
            missing = REQUIRED_COLUMNS - set(ACTUAL_COLUMNS)
            raise ValueError(f"Missing columns: {missing}")
        
        elif len(self.data.columns) > TOTAL_REQUIRED_COLUMNS:
            extra = set(self.data.columns) - REQUIRED_COLUMNS
            print(f"Extra Columns:  {extra}")
        
        # now assign the list to actual column names 
        self.data.columns = ACTUAL_COLUMNS
        # remove blank rows 
        self.data = self.data.dropna(axis=0)
        CITY_LIST = ["London", "Paris", "Dubai"]
        # regex pattern to remove
        regExPattern = ",,$"
        # remove the pattern 
        self.data["timestamp"] = self.data["timestamp"].str.replace(regExPattern, ",", regex=True)

        # loop over rows 
        # self.data.loc[x] — Represents an entire row.
        # self.data.loc[:, "amount"] — Represents an entire column.
        # self.data.loc[x, "amount"] — Represents a single cell.

        for x in self.data.index:
            # checks for negative 'amount' and replaces it 
            if self.data.loc[x, "amount"] < 0:
                self.data.loc[x, "amount"] = random.randrange(1, 100)
            # checks for unknown 'location' and replaces it
            if self.data.loc[x, "location"] == "Unknown":
                    self.data.loc[x, "location"] = random.choice(CITY_LIST)

        return self.data
        
    def saveToParquet(self):
        # create folder if missing using the smart data_root
        processed_dir = self.data_root / "processed"
        os.makedirs(processed_dir, exist_ok=True)
        # save as parquet file
        self.data.to_parquet(processed_dir / 'processed_data.parquet')

        ###############################################################
        # 0. what is index leel in parquet??? why its not printed out
        # 1. schedule set this script to run daily Windows Task Scheduler 

# Wrap the execution so it doesn't crash main.py during import
if __name__ == "__main__":
    dp = DataPipeline()
    dp.validate_file()
    dp.validate_schemas()
    dp.saveToParquet()