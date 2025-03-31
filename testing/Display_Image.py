import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

def display_images_in_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return
    file_list = os.listdir(folder_path)
    image_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if "solution.png" in image_files:
         image_files.remove("solution.png")
    num_images = len(image_files)
    num_cols = 3  
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))
    # Ensure axes is a 2D array
    axes = np.array(axes).reshape((num_rows, num_cols))
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)

       
        image = cv2.imread(image_path)

        
        if image is not None:
            
            ax = axes[i // num_cols, i % num_cols]
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(image_file)
            ax.axis('off')  

    
    plt.tight_layout()
    plt.show()
def display_image(image_path="Output\OutputImages\solution.png"):
    """
    Display a single image.

    Args:
        image_path (str): The path to the image file.
    """
    if not os.path.isfile(image_path):
        print(f"Error: {image_path} is not a valid file.")
        return

    image = cv2.imread(image_path)

    if image is not None:
       
        plt.figure(figsize=(6, 4))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(os.path.basename(image_path))
        plt.axis('off')
        plt.show()
def display_2images(image1_name, image2_name,folder_path="Output\OutputImages"):
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return

    # Check if the specified images exist in the folder
    image1_path = os.path.join(folder_path, image1_name)
    image2_path = os.path.join(folder_path, image2_name)

    if not os.path.isfile(image1_path):
        print(f"Error: {image1_name} not found in {folder_path}.")
        return

    if not os.path.isfile(image2_path):
        print(f"Error: {image2_name} not found in {folder_path}.")
        return

    # Create a 1x2 grid for the two images
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

  
    image1 = cv2.imread(image1_path)
    if image1 is not None:
        axes[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        axes[0].set_title(image1_name)
        axes[0].axis('off')

    image2 = cv2.imread(image2_path)
    if image2 is not None:
        axes[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        axes[1].set_title(image2_name)
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    folder_path = "Output\OutputImages"
    display_image()
