import os
from io import BytesIO
from datetime import date
from urllib.request import urlopen
from PIL import Image
from fastapi import APIRouter, HTTPException, status
# Langchain imports
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.prompts import PromptTemplate
# Local imports
from backend.models.prompt import Prompt
from backend.services.RAG import RAG


router = APIRouter(prefix="/api", tags=["generate_flyer"])

""" 
TODO - Implement the following features:
- Add multimodal models
- Ensure different handling of different models (text-to-image and multimodal)
"""

# Initialize RAG instance
rag = RAG()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Prompt templates
multimodal_prompt_template = PromptTemplate(
    input_variables=["user_prompt", "rag_embedding_text", "rag_embedding_image"],
    template=(
        "You are a Senior Designer of a giant Brazilian supermarket and will generate a high-quality flyer in Brazilian Portuguese "
        "based on the following prompt by user: {user_prompt}. You will consider these images: {rag_embedding_image} and texts: "
        "{rag_embedding_text} as examples to create the flyer. Ensure vibrant colors, clear text, and a professional supermarket flyer layout."
    )
)

textonly_prompt_template = PromptTemplate(
    input_variables=["user_prompt", "rag_embedding_text"],
    template=(
        "You are a Senior Designer of a giant Brazilian supermarket and will generate a high-quality flyer in Brazilian Portuguese "
        "based on the following prompt by user: {user_prompt}. You will consider these texts: {rag_embedding_text} as examples of "
        "descriptions of flyer images to create what the user wants. Ensure vibrant colors, clear text, and a professional supermarket flyer layout."
    )
)


def save_image(image: Image, model_name: str) -> tuple[str,str]:
    """Save the generated image and return filename and path."""
    os.makedirs("./uploads", exist_ok=True)
    filename = f"generated_flyer_{date.today().isoformat()}_{model_name}.jpg"
    save_path = os.path.join("./uploads", filename)
    image.save(save_path, format="JPEG")
    return filename, save_path

async def get_rag_data(prompt: Prompt, multimodal: bool = False) -> tuple[list[str], list[str]]:
    """Fetch RAG data and extract text and image embeddings."""
    rag_responses = await rag.get_similar_flyers(query=prompt)
    if not rag_responses:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No similar flyers found in RAG."
        )
    text_rag_embedding = [rag_response.get('text', '') for rag_response in rag_responses]
    
    if multimodal:
        image_rag_embedding = [rag_response.get('image', '') for rag_response in rag_responses]
        return text_rag_embedding, image_rag_embedding
    
    return text_rag_embedding


@router.post("/dalle")
async def generate_dalle(prompt: Prompt):
    """
    Generate a flyer using DALL-E (text-to-image model)
    """

    test = Prompt(text='Flyer', images=[])
    try:
        # Get similar flyer and combined prompt from RAG
        text_rag_embedding = await get_rag_data(test)

        input_vars = {
            "user_prompt": prompt.text,
            "rag_embedding_text": text_rag_embedding
        }        
        
        # Create prompt for DALL-E
        formatted_prompt = textonly_prompt_template.format(**input_vars)

        print(formatted_prompt)

        # Generate image using DALL-E
        dalle = DallEAPIWrapper(model="dall-e-3", n=1, size="1024x1024")
        image_url = dalle.run(formatted_prompt)

        if not image_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to generate image with DALL-E."
            )
        
        # Download and process image
        image_data = urlopen(image_url).read()
        image = Image.open(BytesIO(image_data))
        
        # Save image
        filename, save_path = save_image(image, "DALL-E")

        return {
            "message": "Flyer generated successfully",
            "model": "DALL-E",
            "filename": filename,
            "path": save_path,
        }, status.HTTP_200_OK
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in generate_dalle: {str(e)}"
        ) 