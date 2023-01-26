from PIL import Image
import os
import glob

def download_images():
    pass

def resize_images():
    image_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data/images")
    dir_list = os.listdir(image_dir)
    
    image_folder_list = []
    min_image_height = None
    
    for folder in dir_list:
        image_folder = os.path.join(image_dir,folder)
        
        image_folder_list.append(glob.glob(image_folder +"/*.png"))
        
    for image_folder in image_folder_list:
        
        for image in image_folder:
        
            print(image)
            
            img = Image.open(image)
            _, height = img.size
            
            if img.mode != "RGB":
                image_folder_list.remove(image)
                break
            
            if min_image_height == None:
                min_image_height = height
            elif height < min_image_height:
                min_image_height = height
            
            print(height)
            
            
        
            
            

            
    
    
alter_images = resize_images()
    
