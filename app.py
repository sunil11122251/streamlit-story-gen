import streamlit as st
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch

try:
    tokenizer = GPT2Tokenizer.from_pretrained('./distilgpt2')
    model = GPT2LMHeadModel.from_pretrained('./distilgpt2')
    device = torch.device('cpu')
    if model.device.type == 'meta':
        model.to_empty(device=device)
    else:
        model.to(device)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', device=device)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

st.markdown("""
    <style>
    .main { background: radial-gradient(circle, #ff6b6b, #4ecdc4); padding: 40px; border-radius: 20px; max-width: 1000px; margin: 30px auto; }
    h1 { color: #fff; text-align: center; font-family: 'Montserrat'; text-shadow: 0 0 10px #ffeb3b; }
    </style>
""", unsafe_allow_html=True)

st.title("Story Generator")
genre = st.selectbox("Choose Genre", ["Fantasy", "Sci-Fi", "Horror"])
prompt = st.text_input("Starting Sentence")
if st.button("Generate Story"):
    try:
        story = generator(
            f"{genre} story: {prompt}",
            max_length=100,
            truncation=True,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            num_return_sequences=1
        )[0]['generated_text']
        st.write(story)
        st.download_button("Download Story", story, "story.txt")
    except Exception as e:
        st.error(f"Error generating story: {str(e)}")