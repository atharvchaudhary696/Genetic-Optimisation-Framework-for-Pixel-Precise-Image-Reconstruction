import cv2
import numpy as np
def main(img,FolderPath):
    image = cv2.imread(img)
    for i in range(1,4):
        img=FolderPath+f"/{i}.jpg"
        cv2.imwrite(img,image)
if __name__ == "__main__":
    pass