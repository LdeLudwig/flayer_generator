#from kaggle import KaggleApi
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration, AutoProcessor
from image_tratative import sample_image_urls


# Define the image captioning model
captioning_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# Define the image embedding model
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
captioning_model.to(device)
clip_model.to(device)

def generate_caption(images):
    try:
        pair_object = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise ValueError("Invalid image format. Expected PIL.Image.Image.")
            input = captioning_processor(image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = captioning_model.generate(**input)

            caption = captioning_processor.decode(output[0], skip_special_tokens=True)
    
            pair_object.append((image, caption))
            
        return pair_object, caption
    
    except Exception as e:
        raise Exception("Error in generate_caption: ", e)


def get_image_embeddings(pair_list):
    
    try:
        embed_list = []
        for image, caption in pair_list:

            
            # Load process to text
            input_caption = clip_processor(text=[caption], return_tensors="pt", padding=True, truncate=True).to(device)
            input_image = clip_processor(images=image, return_tensors="pt", padding=True, truncate=True).to(device)

            with torch.no_grad():
                caption_embedding = clip_model.get_text_features(**input_caption)
                image_embedding = clip_model.get_image_features(**input_image)

            caption_array = caption_embedding.squeeze().cpu().numpy()
            image_array = image_embedding.squeeze().cpu().numpy()
            
            # calculating similarity
            combined_inputs = clip_processor(images=image, text=[caption], return_tensors="pt", padding=True, truncate=True).to(device)
            with torch.no_grad():
                clip_outputs = clip_model(**combined_inputs)
                clip_similarity = float(clip_outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0][0])
                
            embed_list.append({
                "image":image_array,
                "caption": caption_array,
                "similarity": clip_similarity
            })  
        
        print(embed_list[0])
        return embed_list
    except Exception as e:
        raise Exception("Error in get_image_embeddings: ", e)

def main():
    # get the images from the database
    images = list(map(lambda el: Image.open(el), sample_image_urls))
    #print(images)
    # get the embeddings from the images
    pair_list, caption = generate_caption(images)

    # send pairs of images and captions to the embedding model
    get_image_embeddings(pair_list)

if __name__ == "__main__":
    main()
    
