import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
def display_image(image_path="Output\OutputImages\solution.png"):
    """
    Display a single image.

    Args:
        image_path (str): The path to the image file.
    """
    if not os.path.isfile(image_path):
        print(f"Error: {image_path} is not a valid file.")
        return

    # Read the image
    image = cv2.imread(image_path)

    if image is not None:
        # Display the image
        plt.figure(figsize=(6, 4))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(os.path.basename(image_path))
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
   display_image()
