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

""" Used locally to convert all the test images in jpg """
def formatting_train_images_to_jpg():
    base_dir = "flyers_data/train"
    all_images = os.listdir(base_dir)
    # get the full path of the images
    train_sample_flyers = list(map(lambda item: f"{base_dir}/{item}", all_images))
    
    flyers_images = []
    for image_url in train_sample_flyers:
        image = Image.open(image_url)
        if image_url.split('.')[-1] != 'jpg':
            image.save(f"./flayers/train/{image_url.split('/')[-1].split('.')[0]}.jpg")
            os.remove(image_url)

        flyers_images.append(image)
         
    return flyers_images