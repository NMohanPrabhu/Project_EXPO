import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Title and description
st.set_page_config(page_title="Smarteye", layout="centered")
st.title("📸 Smarteye: AI that Sees and Describes")
st.write("Upload one or more images and choose a captioning style.")

# Load model and processor once
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

processor, model = load_model()

# Define caption modes
CAPTION_MODES = {
    "Descriptive": "a photo of",
    "Creative": "an artistic image of",
    "Detailed": "a detailed photo of"
}

# Sidebar options
caption_mode = st.sidebar.selectbox("Select Caption Mode", list(CAPTION_MODES.keys()))
st.sidebar.write("💡 *Try 'Creative' for fun captions!*")

# Upload images
uploaded_files = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Function to generate caption
def generate_caption(image: Image.Image, mode: str) -> str:
    if mode == "Unconditional":
        inputs = processor(image, return_tensors="pt")
    else:
        prompt = CAPTION_MODES[mode]
        inputs = processor(image, prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Display images and captions
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.spinner("Generating caption..."):
                caption = generate_caption(image, caption_mode)

            st.success("📝 Caption:")
            st.markdown(f"**{caption}**")
            st.markdown("---")

        except Exception as e:
            st.error(f"Error processing image: {e}")

