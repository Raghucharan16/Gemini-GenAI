import os
from dotenv import load_dotenv
from google.generativeai import genai
import streamlit as st 
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Gemini Image Demo")
model=genai.GenerativeModel('gemini-pro-vision')

def get_gemini_response_vision(input,image,prompt):
    if input!="":
        response = model.generate_content([input,image[0],prompt])
    else:
        response = model.generate_content(image)
    return response.text

#validate image
def validate_image(image):
    if image is not None:
        return Image.open(image)
    return None

st.header("Gemini Application")
input=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image=validate_image(uploaded_file)
submit=st.button("Tell me about the image")
instructions = "You are a powerful image content extractor and will analyze the image to provide a detailed description of the image based on the input prompt."
if submit:
    response=get_gemini_response_vision(instructions,image,input)
    st.subheader("The Response is")
    st.write(response)