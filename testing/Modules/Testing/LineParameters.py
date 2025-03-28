import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

class Main:
    def GenerateParameters(self):
        Image = cv2.imread(self.ImagePath)
        Image = cv2.resize(Image, (800, 800))
        GrayImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(GrayImage, threshold1=1, threshold2=150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=15)
        blank_image = np.zeros_like(Image)
        LineNumber = 1
        line_parameters = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            thickness = 1
            color = Image[y1, x1]
            line_params = {
                "Starting Point": (x1, y1),
                "Ending Point": (x2, y2),
                "Angle": angle,
                "Line Thickness": thickness,
                "Line Color (BGR)": color.tolist(),  # Convert color to a list
            }
            line_parameters.append(line_params)

            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if self.debug:
                print(LineNumber)
                print(line_params)
            LineNumber += 1

        csv_file_path = "Output/Csv/Input_Image_Line_parameters.csv"
        with open(csv_file_path, mode="w", newline="") as csv_file:
            fieldnames = [
                "Starting Point",
                "Ending Point",
                "Angle",
                "Line Thickness",
                "Line Color (BGR)",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for params in line_parameters:
                writer.writerow(params)

        if self.debug:
            plt.figure(figsize=(12, 6))

            # Display the original image
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(Image, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")

            # Display the blank image with detected lines
            plt.subplot(122)
            plt.imshow(blank_image)
            plt.title("Detected Lines on Blank Image")

            plt.show()

    def __init__(self, ImagePath, debug=False):
        self.ImagePath = ImagePath
        self.debug = debug
        self.GenerateParameters()

if __name__ == "__main__":
    App = Main("Output\InputImages\slanting_lines.png", True)
