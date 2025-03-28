import csv
import cv2
import numpy as np


class Main:
    def Generator(self):
        line_parameters = []
        with open(self.csv_file_path, mode="r") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                line_parameters.append(
                    {
                        "Starting Point": tuple(
                            map(int, row["Starting Point"].strip("()").split(", "))
                        ),
                        "Ending Point": tuple(
                            map(int, row["Ending Point"].strip("()").split(", "))
                        ),
                        "Angle": float(row["Angle"]),
                        "Line Thickness": int(row["Line Thickness"]),
                        "Line Color (BGR)": list(
                            map(int, row["Line Color (BGR)"].strip("[]").split(", "))
                        ),
                    }
                )
        image_height = 800
        image_width = 800
        blank_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        for params in line_parameters:
            x1, y1 = params["Starting Point"]
            x2, y2 = params["Ending Point"]
            angle = params["Angle"]
            thickness = params["Line Thickness"]
            color = tuple(params["Line Color (BGR)"])
            cv2.line(blank_image, (x1, y1), (x2, y2), color, thickness)
        cv2.imshow("Generated Image", blank_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __init__(self, csv_file_path="Output\Csv\modified_line_parameters.csv"):
        self.csv_file_path = csv_file_path
        self.Generator()


if __name__ == "__main__":
    App = Main()
