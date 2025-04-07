import os
os.environ["HF_HOME"] = "C:\\Users\\Hi\\huggingface_cache"
from transformers import GPT2Tokenizer
import logging

logging.basicConfig(level=logging.DEBUG)
try:
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', local_files_only=False)
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error: {str(e)}")