import random
import cv2
import math
import numpy as np
import csv

def DrawLines(blank_image, line_parameters, output_csv_file, debug=False):
    modified_line_parameters = []

    for params in line_parameters:
        x1, y1 = params["Starting Point"]
        x2, y2 = params["Ending Point"]
        angle = params["Angle"]
        thickness = params["Line Thickness"]

        # Randomize parameters
        x1 += random.randint(-50, 50)
        y1 += random.randint(-50, 50)
        x2 += random.randint(-50, 50)
        y2 += random.randint(-50, 50)
        angle += random.uniform(-30, 30)
        thickness += random.randint(-2, 2)

        # Ensure thickness is within a valid range
        thickness = max(1, min(thickness, 10))

        # Generate a random BGR color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Print out parameters for debugging
        if debug:
            print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, angle: {angle}, thickness: {thickness}, color: {color}")

        # Draw the line on the image
        cv2.line(blank_image, (x1, y1), (x2, y2), color, thickness)

        # Append modified parameters to the list
        modified_line_parameters.append({
            "Starting Point": (x1, y1),
            "Ending Point": (x2, y2),
            "Angle": angle,
            "Line Thickness": thickness,
            "Line Color (BGR)": color
        })

    # Write modified parameters to a new CSV file
    with open(output_csv_file, mode="w", newline='') as csv_file:
        fieldnames = ["Starting Point", "Ending Point", "Angle", "Line Thickness", "Line Color (BGR)"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write the modified parameters
        for params in modified_line_parameters:
            writer.writerow({
                "Starting Point": params["Starting Point"],
                "Ending Point": params["Ending Point"],
                "Angle": params["Angle"],
                "Line Thickness": params["Line Thickness"],
                "Line Color (BGR)": params["Line Color (BGR)"]
            })

# Example usage
if __name__ == "__main__":
    image_height = 800
    image_width = 800

    # Initialize the image to all white pixels
    blank_image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

    line_parameters = [...]  # Load your line parameters from the CSV file

    DrawLines(blank_image, line_parameters, "Output/Csv/modified_line_parameters.csv", debug=True)
