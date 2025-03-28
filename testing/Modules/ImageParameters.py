import numpy as np
from PIL import Image
def Main(image_path):
    target_img = Image.open(image_path)
    target_arr = np.array(target_img)
    img_shape = target_arr.shape
    target_chromosome = target_arr.flatten()
    parameters_list = [target_img, target_arr, img_shape, target_chromosome]
    return parameters_list
if __name__ == "__main__":
    image_path="Output\InputImages\\fruit.jpg"
    parameters_list = Main(image_path)
    target_img, target_arr, img_shape, target_chromosome = parameters_list
    print("target_img:", target_img)
    print("target_arr:", target_arr)
    print("img_shape:", img_shape)
    print("target_chromosome:", target_chromosome)
