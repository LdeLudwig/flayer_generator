import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from io import BytesIO
from PIL import Image


def caption_image(
    image: Image.Image,
) -> str:
    def pil_image_to_base64(image:Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode()

        return encoded_image
    
    llm = ChatOpenAI(temperature=0.5, model_name="gpt-4o")

    base64_img = pil_image_to_base64(image)
    message = HumanMessage(content=[
            {"type": "text", "text":"Generate a descriptive caption for this image. Focused on the structure with at least 10 words."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_img}"}}
        ]
    )
    response = llm.invoke([message])
    caption = response.content

    return caption
            