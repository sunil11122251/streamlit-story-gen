# download_model.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel

try:
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    tokenizer.save_pretrained('distilgpt2')
    model.save_pretrained('distilgpt2')
    print("Model downloaded and saved successfully!")
except Exception as e:
    print(f"Error: {str(e)}")