# from kaggle import KaggleApi
import torch
import base64
import io
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os


class Embedding:
    def __init__(self):
        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the image embedding model
        # Use use_fast=False to avoid torchvision dependency issues while still being explicit
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", use_fast=False
        )
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # Initialize OpenAI client for caption generation
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY")
        )

    def generate_caption(self, images):
        """
        Generate captions for images using GPT-4o vision capabilities
        Args:
            images: List of PIL Image objects or image paths
        Returns:
            List of tuples (image, caption)
        """
        try:
            pair_list = []

            for image in images:
                # Convert PIL Image to base64 for GPT-4o
                if isinstance(image, str):
                    # If image is a path, load it
                    image = Image.open(image)

                # Convert image to base64
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                # Create message for GPT-4o with image
                message = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Generate a detailed, descriptive caption for this flyer image. Focus on the visual elements, text content, colors, layout, and overall design. Keep it concise but informative for image embedding purposes.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        },
                    ]
                )

                # Get caption from GPT-4o
                response = self.llm.invoke([message])
                caption = response.content.strip()

                # Add to pair list
                pair_list.append((image, caption))

            return pair_list

        except Exception as e:
            raise Exception("Error in generate_caption: ", e)

    def get_train_image_embeddings(self, pair_list):
        try:
            embed_list = []
            for image, caption in pair_list:
                # Load process to text
                input_caption = self.clip_processor(
                    text=[caption],
                    return_tensors="pt",
                    padding=True,
                    max_length=100,
                    truncation=True,
                )
                input_image = self.clip_processor(
                    images=image,
                    return_tensors="pt",
                    padding=True,
                    max_length=100,
                    truncation=True,
                )

                with torch.no_grad():
                    caption_embedding = self.clip_model.get_text_features(
                        **input_caption
                    )
                    image_embedding = self.clip_model.get_image_features(**input_image)

                caption_array = caption_embedding.squeeze().cpu().numpy()
                image_array = image_embedding.squeeze().cpu().numpy()

                embed_list.append(
                    {
                        "image": image_array,
                        "text": caption_array,
                    }
                )

            return embed_list
        except Exception as e:
            raise Exception("Error in get_image_embeddings: ", e)

    def input_image_embedding(self, image: Image):
        try:
            input_image = self.clip_processor(
                images=image,
                return_tensors="pt",
                padding=True,
                max_length=100,
                truncation=True,
            )

            with torch.no_grad():
                image_embedding = self.clip_model.get_image_features(**input_image)

            image_array = image_embedding.squeeze().cpu().numpy()
            return image_array
        except Exception as e:
            raise Exception("Error in input_image_embedding: ", e)

    def input_text_embedding(self, text: str):
        try:
            input_text = self.clip_processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                max_length=200,
                truncation=True,
            )

            with torch.no_grad():
                text_embedding = self.clip_model.get_text_features(**input_text)

            text_array = text_embedding.squeeze().cpu().numpy()
            return text_array
        except Exception as e:
            raise Exception("Error in input_text_embedding: ", e)
