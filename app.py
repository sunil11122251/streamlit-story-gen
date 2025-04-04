# app.py
import streamlit as st
from transformers import pipeline

generator = pipeline('text-generation', model='distilgpt2')

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
    story = generator(f"{genre} story: {prompt}", max_length=1000)[0]['generated_text']
    st.write(story)
    st.download_button("Download Story", story, "story.txt")