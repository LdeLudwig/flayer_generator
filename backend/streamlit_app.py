import os
import requests
from streamlit import st
from PIL import Image
from backend.models.prompt import Prompt

# Backend api url
fastapi_url = "http://localhost:8000"

# Input form
st.title("🎴 Flyer Generator")
text_prompt = st.text_area("Enter text prompt")
uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])

"""
TODO - Implement the following features:
- Add the choose model logic
- Ensure the right API usage
"""


with st.sidebar:
    st.sidebar.header("Choose your model", divider="red")
    model_name = st.selectbox(
        "Select a model",
        ("dall-e-3", "stable_diffusion", "flux"),
        index=0,
        help="Choose a model for generating text.",
    )

    st.sidebar.header("View generated flyers", divider="red")
    col1, col2, col3 = st.sidebar.columns(3)

    with col1:
        st.image("./flyers_data/train/ENCARTE-1.jpg")
        st.caption("ENCARTE-1")
    
    with col2:
        st.image("./flyers_data/train/10042854.jpg")
        st.caption("ENCARTE-2")
    with col3:
        st.image("./flyers_data/train/10557831.jpg")
        st.caption("ENCARTE-3")


if st.button("Generate Flyer"):

    prompt_data = Prompt(text='', images=[])

    if not text_prompt and not uploaded_file:
        st.error("Please enter text or upload an image.")
    else:
        if text_prompt:
            prompt_data.text = text_prompt
        if uploaded_file:

            os.makedirs("./temp_uploads", exist_ok=True)
            temp_path = f"./temp_uploads/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            prompt_data.images.append(Image.open(temp_path)) 
        
        # Call backend endpoints
        try:
            response = requests.post(f"{fastapi_url}/api/dalle", prompt_data.model_dump_json())
            if response.status_code == 200:
                result = response.json()
                st.success(result['message'])

                # Display generated flyer
                flyer_path = result['path']
                st.image(flyer_path, caption="Generated Flyer", use_column_width=True)
            else:
                st.error(f"Error Streamlit dalle request: {response.json().get('detail')}")
        
        except Exception as e:
                st.error(f"Error Streamlit: {str(e)}")