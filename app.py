import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="AI Summarization Tool")



@st.cache_resource
def load_captioning_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

st.title("Image Captioning Tool")
captioning_model = load_captioning_model()

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_image = st.file_uploader("upload an image", type=["png", "jpg", "jpeg"])
    caption_generator = st.button("Generate Captions", type="primary")

with col2:
    st.markdown("Powered by Pinitha Savidya")

if uploaded_image and caption_generator:
    with st.spinner("Generating captions..."):
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, use_container_width=True)
        result = captioning_model(image)
        gen_text = result[0]['generated_text']
        st.markdown(gen_text)

elif uploaded_image:
    st.warning("Please Upload an image to Captioning")
