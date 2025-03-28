
from modules import ImageParameters
from modules import Display_Image 
from modules import Image_Genetic_Algorithm
from  modules import Filter
def Main(image_path,FolderPath,Apprunning=False,display=False):
    if Apprunning==True:
        parameters_list  = ImageParameters.Main(image_path)
        Image_Genetic_Algorithm.Main(parameters_list,FolderPath)
        if display==True:
            img=FolderPath+"/1.jpg"
            Display_Image.display_image(img)
if __name__ == "__main__":
    Main(Apprunning=True,display=True,image_path="FlaskBackend/UPLOAD_FOLDER/UploadedImage.jpg",FolderPath="FlaskBackend/UPLOAD_FOLDER")