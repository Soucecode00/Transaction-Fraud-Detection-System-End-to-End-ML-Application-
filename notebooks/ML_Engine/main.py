# main.py
# 1. Reach into the 'src' folder and grab the 'load_data' tool
from src.data_loader import load_data

def start_pipeline():
    print("--- Starting Fraud Detection Pipeline ---")
    
    # 2. Execute the data loading
    df = load_data()
    
    if df is not None:
        print("Success! Here is a preview of the data:")
        print(df.head())
    else:
        print("Failed to load data. Check the errors above.")

# This line ensures the code only runs if you run main.py directly
if __name__ == "__main__":
    start_pipeline()
    