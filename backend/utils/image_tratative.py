import os
from PIL import Image


""" 
    TODO(get_user_uploaded_images) - in case of a complex application

 - receive the uploaded product images, logos or templates of their supermarkets uploaded by the user
 - convert the images to jpg
 - save the images in the relational database
"""

# this part may be removed later to be replaced by the function to get the images uploaded by the user
# get all images from ./flayers/train
base_dir = "../../flyers_data/train"
all_images = os.listdir(base_dir)

# get the full path of the images
sample_image_urls = list(map(lambda item: f"{base_dir}/{item}", all_images))


""" Used locally to convert all the test images in jpg """
# convert the images to jpg
def convert_image_to_jpg(sample_images_urls):
    for image_url in sample_images_urls:
        image = Image.open(image_url)
        rgb_image = image.convert("RGB")
        rgb_image.save(f"./flayers/train/{image_url.split('/')[-1].split('.')[0]}.jpg")
        os.remove(image_url)
    return sample_images_urls