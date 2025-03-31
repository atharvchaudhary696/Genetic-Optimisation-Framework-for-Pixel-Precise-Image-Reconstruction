import csv
import cv2
import numpy as np
import random
import math
from Generative_module import DrawLines
image_gen_count= 1
class Main:
    def Generator(self, num):
        line_parameters = []
        with open(self.csv_file_path, mode="r") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                line_parameters.append(
                    {
                        "Starting Point": tuple(map(int, row["Starting Point"].strip("()").split(", "))),
                        "Ending Point": tuple(map(int, row["Ending Point"].strip("()").split(", "))),
                        "Angle": float(row["Angle"]),
                        "Line Thickness": int(row["Line Thickness"]),
                        "Line Color (BGR)": list(map(int, row["Line Color (BGR)"].strip("[]").split(", "))),
                    }
                )

        image_height = 800
        image_width = 800

        # Initialize the image to all white pixels
        blank_image = np.ones(
            (image_height, image_width, 3), dtype=np.uint8) * 255

        DrawLines(blank_image,line_parameters,"Output/Csv/modified_line_parameters.csv",debug=self.debug)

      
        cv2.imwrite(f"Output\OutputImages/{num}.png", blank_image)

        

    def __init__(self, csv_file_path="Output\Csv\Input_Image_Line_parameters.csv",debug=False):
        self.csv_file_path = csv_file_path
        self.debug=debug
        self.Generator(image_gen_count)
        self.Generator(image_gen_count+1)
        self.Generator(image_gen_count+2)

if __name__ == "__main__":
    App = Main()
               