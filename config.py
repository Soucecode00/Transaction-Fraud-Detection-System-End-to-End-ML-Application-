# config.py 
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Get data path from environment variable
DATA_PATH = os.getenv("Data_path", r"C:\Users\acer\Desktop\AI x Fintech projects\Transaction Fraud Detection pipeline\Data\Transaction.csv")
