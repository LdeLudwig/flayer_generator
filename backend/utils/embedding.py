#from kaggle import KaggleApi
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipForConditionalGeneration, AutoProcessor

class Embedding:
    def __init__(self):
        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the image captioning model
        self.captioning_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
        self.captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # Define the image embedding model
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


    def generate_caption(self, images):
        try:
            pair_object = []
            for image in images:
                if not isinstance(image, Image.Image):
                    raise ValueError("Invalid image format. Expected PIL.Image.Image.")
                input = self.captioning_processor(image, return_tensors="pt", max_length=100, truncation=True)
                
                with torch.no_grad():
                    output = self.captioning_model.generate(**input)

                caption = self.captioning_processor.decode(output[0], skip_special_tokens=True)
        
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

                with torch.no_grad():
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

            with torch.no_grad():
                image_embedding = self.clip_model.get_image_features(**input_image)

            image_array = image_embedding.squeeze().cpu().numpy()
            return image_array
        except Exception as e:
            raise Exception("Error in input_image_embedding: ", e)
        
    def input_text_embedding(self, text: str):
        try:
            input_text = self.clip_processor(text=[text], return_tensors="pt", padding=True, max_length=200, truncation=True)

            with torch.no_grad():
                text_embedding = self.clip_model.get_text_features(**input_text)
            
            text_array = text_embedding.squeeze().cpu().numpy()
            return text_array
        except Exception as e:
            raise Exception("Error in input_text_embedding: ", e)