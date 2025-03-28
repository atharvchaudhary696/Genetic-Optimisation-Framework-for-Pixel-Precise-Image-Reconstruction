from Modules import ImageParameters
from Modules import Display_Image 
from Modules import Image_Genetic_Algorithm
# from  Modules import Filter
def Main(Apprunning=False,image_path="fruit.jpg",FolderPath="Modules",display=False):
    if Apprunning==True:
        parameters_list  = ImageParameters.Main(image_path)
        Image_Genetic_Algorithm.Main1(parameters_list,FolderPath)
        if display==True:
            img=FolderPath+"/Finalimg.png"
            Display_Image.display_image(img)
if __name__ == "__main__":
    Main(Apprunning=True,display=True,image_path="Output\InputImages\\fruit.jpg",FolderPath="Output\OutputImages")