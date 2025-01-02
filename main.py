from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os
import textwrap
from app.chat import get_gemini_response
import google.generativeai as genai



def to_markdown(text):
  text = text.replace('•', '  *')
  return (textwrap.indent(text, '> ', predicate=lambda _: True))

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



##initialize our streamlit app

st.set_page_config(page_title="GEMINI GEN AI")

st.header("Gemini Application")

input=st.text_input("Input: ",key="input")


submit=st.button("Ask the question")


if submit:
    
    response=get_gemini_response(input)
    st.subheader("The Response is")
    st.write(response)