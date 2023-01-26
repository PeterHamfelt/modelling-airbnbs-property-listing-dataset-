from PIL import Image
import os
import glob
import shutil

def download_images():
    pass

def resize_images():
    """Resize property images

    This function resizes all the images in the images folder by changing the heights of
    the images but at the same time maintaining the aspect ratio of the images. The resized
    images is then saved into a new folder named processed_images. This function will also 
    remove any images which is not RGB.
    """
    
    image_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data/images")
    dir_list = os.listdir(image_dir)
    
    image_folder_list = []
    min_image_height = None
    
    for folder in dir_list:
        image_folder = os.path.join(image_dir,folder)
        
        image_folder_list.append(glob.glob(image_folder +"/*.png"))
        
    for img_num, image_folder in enumerate(image_folder_list):
        
        for image in image_folder:
            
            img = Image.open(image)
            _, height = img.size
            
            if img.mode != "RGB":
                image_folder_list.pop(img_num)
                break
            
            if min_image_height == None:
                min_image_height = int(height)
            elif height < min_image_height:
                min_image_height = int(height)
        
    alter_image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data/processed_images")     
        
    if os.path.exists(alter_image_path) == False:
        os.makedirs(alter_image_path)
    else:
        shutil.rmtree(alter_image_path)
        os.makedirs(alter_image_path)  
        
    for image_folder in image_folder_list:
        
        for image in image_folder:
            
            image_name = image.split("\\")[-1]
            save_image_dir = os.path.join(alter_image_path,image_name)
            
            img = Image.open(image)
            
            img_current_width , img_current_height = img.size
            
            img_new_width = int(img_current_width*(min_image_height/img_current_height))
            
            new_image = img.resize((img_new_width,min_image_height))
            new_image.save(save_image_dir)                     
    
    
alter_images = resize_images()
    
