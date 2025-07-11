#from kaggle import KaggleApi
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from backend.services.caption_generator import caption_image


class Embedding:
    def __init__(self):
        # Define the image embedding model
        model_name = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
        force_offline = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"

        try:
            if force_offline:
                # Force offline mode
                print("Loading models in offline mode...")
                self.clip_processor = CLIPProcessor.from_pretrained(
                    model_name,
                    use_fast=True,
                    local_files_only=True
                )
                self.clip_model = CLIPModel.from_pretrained(
                    model_name,
                    local_files_only=True
                )
                print("Successfully loaded models from local cache.")
            else:
                # Try to load with fast processor first (online mode)
                self.clip_processor = CLIPProcessor.from_pretrained(
                    model_name,
                    use_fast=True
                )
                self.clip_model = CLIPModel.from_pretrained(model_name)
                print("Successfully loaded models (online mode).")
        except (OSError, ConnectionError) as e:
            # If network is unreachable, try to load from local cache
            print(f"Network error occurred: {e}")
            print("Attempting to load models from local cache...")
            try:
                self.clip_processor = CLIPProcessor.from_pretrained(
                    model_name,
                    use_fast=True,
                    local_files_only=True
                )
                self.clip_model = CLIPModel.from_pretrained(
                    model_name,
                    local_files_only=True
                )
                print("Successfully loaded models from local cache.")
            except Exception as cache_error:
                print(f"Failed to load from cache: {cache_error}")
                print("Please ensure you have internet connectivity or the models are cached locally.")
                print("You can download models using: python backend/scripts/download_models.py")
                raise RuntimeError(
                    "Unable to load CLIP models. Please check your internet connection "
                    "or ensure the models are available in the local cache."
                ) from cache_error


    def generate_caption(self, images):
        try:
            pair_object = []
            for image in images:
                if not isinstance(image, Image.Image):
                    raise ValueError("Invalid image format. Expected PIL.Image.Image.")
                caption = caption_image(image)
        
                pair_object.append((image, caption))
                
            return pair_object
        
        except Exception as e:
            raise Exception("Error in generate_caption: ", e)


    def get_train_image_embeddings(self, pair_list):
        
        try:
            embed_list = []
            for image, caption in pair_list:
                # Load process to text
                input_caption = self.clip_processor(text=[caption], return_tensors="pt", padding=True, max_length=100, truncation=True)
                input_image = self.clip_processor(images=image, return_tensors="pt", padding=True, max_length=100, truncation=True)

                
                caption_embedding = self.clip_model.get_text_features(**input_caption)
                image_embedding = self.clip_model.get_image_features(**input_image)

                caption_array = caption_embedding.squeeze().cpu().numpy()
                image_array = image_embedding.squeeze().cpu().numpy()
                                
                embed_list.append({
                    'image':image_array,
                    'text': caption_array,
                })  
        
            return embed_list
        except Exception as e:
            raise Exception("Error in get_image_embeddings: ", e)
        
    def input_image_embedding(self, image: Image):
        try:
            input_image = self.clip_processor(images=image, return_tensors="pt", padding=True, max_length=100, truncation=True)

            
            image_embedding = self.clip_model.get_image_features(**input_image)

            image_array = image_embedding.squeeze().cpu().numpy()
            return image_array
        except Exception as e:
            raise Exception("Error in input_image_embedding: ", e)
        
    def input_text_embedding(self, text: str):
        try:
            input_text = self.clip_processor(text=[text], return_tensors="pt", padding=True, max_length=200, truncation=True)

            
            text_embedding = self.clip_model.get_text_features(**input_text)
            
            text_array = text_embedding.squeeze().cpu().numpy()
            return text_array
        except Exception as e:
            raise Exception("Error in input_text_embedding: ", e)