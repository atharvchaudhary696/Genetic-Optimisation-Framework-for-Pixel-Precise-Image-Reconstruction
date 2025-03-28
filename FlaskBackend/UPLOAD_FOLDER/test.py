import cv2
import os
import time
# Load the image
print(os.getcwd())
img = cv2.imread("1.jpg")

# Check if the image is empty
if img is None:
    print("The image is empty.")
else:
    while True:
        # Make changes to the image
        # For example, to blur the image:
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Save the changed image
        cv2.imwrite("1.jpg", img)
        time.sleep(0.5)