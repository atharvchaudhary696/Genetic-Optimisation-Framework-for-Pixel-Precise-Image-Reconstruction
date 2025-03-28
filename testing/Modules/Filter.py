import cv2
import numpy as np
# from skimage.util import random_noise
def display(Final_image="Output\FinalImg1.png"):
    image = cv2.imread(Final_image)
    cv2.imshow('Painterly Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def ImageReshaping(FinalImagePath="Output\OutputImages\Finalimg.png",OutputFolder='Output/'):
    image = cv2.imread(FinalImagePath)
    resized_image = cv2.resize(image, (800,600))
    output_image_path=OutputFolder+"FinalImg1.png"
    cv2.imwrite(output_image_path, resized_image)
def filter1(ImagePath="Output\FinalImg1.png",OutputFolder='Output/'):
    image = cv2.imread(ImagePath)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 15
    dynamic_range = 1
    image = cv2.stylization(image, sigma_s=radius, sigma_r=dynamic_range)
    cv2.imwrite(ImagePath, image)
def filter3(ImagePath="Output\FinalImg1.png",OutputFolder='Output/'):
        # Load an image
    im_arr = cv2.imread("D:/downloads/opencv_logo.PNG")
    
    # Add salt and pepper noise to the image
    noise_img = random_noise(im_arr, mode="s&p",amount=0.3)
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    
    # Apply median filter
    median = cv2.medianBlur(noise_img,5)
def filter2(ImagePath="Output\FinalImg1.png",OutputFolder='Output/'):
    image = cv2.imread(ImagePath)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    upsampled_image = cv2.resize(blurred_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    sharpened_image = cv2.filter2D(upsampled_image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype="int"))
    cv2.imwrite(ImagePath, sharpened_image)

if __name__ == "__main__":
    ImageReshaping()
    filter3()
    display()