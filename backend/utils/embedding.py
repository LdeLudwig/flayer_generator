#from kaggle import KaggleApi
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipForConditionalGeneration, AutoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image captioning model
captioning_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Define the image embedding model
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

def generate_caption(images):
    try:
        pair_object = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise ValueError("Invalid image format. Expected PIL.Image.Image.")
            input = captioning_processor(image, return_tensors="pt", max_length=100, truncation=True).to(device)
            
            with torch.no_grad():
                output = captioning_model.generate(**input)

            caption = captioning_processor.decode(output[0], skip_special_tokens=True)
    
            pair_object.append((image, caption))
            
        return pair_object
    
    except Exception as e:
        raise Exception("Error in generate_caption: ", e)


def get_image_embeddings(pair_list):
    
    try:
        embed_list = []
        for image, caption in pair_list:
            # Load process to text
            input_caption = clip_processor(text=[caption], return_tensors="pt", padding=True, max_length=100, truncation=True).to(device)
            input_image = clip_processor(images=image, return_tensors="pt", padding=True, max_length=100, truncation=True).to(device)

            with torch.no_grad():
                caption_embedding = clip_model.get_text_features(**input_caption)
                image_embedding = clip_model.get_image_features(**input_image)

            caption_array = caption_embedding.squeeze().cpu().numpy()
            image_array = image_embedding.squeeze().cpu().numpy()
                            
            embed_list.append({
                'image':image_array,
                'text': caption_array,
            })  
    
        return embed_list
    except Exception as e:
        raise Exception("Error in get_image_embeddings: ", e)
    
