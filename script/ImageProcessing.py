import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class Main:
    def create_color_histograms(self):
        blue_channel, green_channel, red_channel = cv.split(self.image)
        hist_size = 256
        hist_range = (0, 256)
        blue_hist = cv.calcHist([blue_channel], [0], None, [hist_size], hist_range)
        green_hist = cv.calcHist([green_channel], [0], None, [hist_size], hist_range)
        red_hist = cv.calcHist([red_channel], [0], None, [hist_size], hist_range)
        total_pixels = self.image.shape[0] * self.image.shape[1]
        blue_hist /= total_pixels
        green_hist /= total_pixels
        red_hist /= total_pixels
        parameter_vector = np.concatenate((blue_hist, green_hist, red_hist))
        plt.figure(figsize=(10, 5))
        plt.subplot(131)
        plt.plot(parameter_vector[:256], color="blue")
        plt.title("Blue Histogram")
        plt.xlim([0, 256])

        plt.subplot(132)
        plt.plot(parameter_vector[256:512], color="green")
        plt.title("Green Histogram")
        plt.xlim([0, 256])

        plt.subplot(133)
        plt.plot(parameter_vector[512:], color="red")
        plt.title("Red Histogram")
        plt.xlim([0, 256])

        plt.tight_layout()
        plt.show()

   

        return parameter_vector

    def extract_edge_features(self):
        self.edges = cv.Canny(self.gray_image, threshold1=100, threshold2=200)
        plt.figure(figsize=(8, 6))
        plt.subplot(121)
        plt.imshow(cv.cvtColor(self.image, cv.COLOR_BGR2RGB))  # Display the original image
        plt.title("Original Image")
        plt.subplot(122)
        plt.imshow(self.edges, cmap="gray")
        plt.title("Edges")
        plt.show()

     

        total_pixels = self.gray_image.size
        edge_pixels = cv.countNonZero(self.edges)
        edge_density = edge_pixels / total_pixels
        orientations = 8
        hist, bin_edges = np.histogram(
            np.arctan2(*np.gradient(self.gray_image)), bins=orientations
        )
        parameter_vector = [edge_density] + list(hist)
        return parameter_vector

    def __init__(self, ImagePath):
        self.IMAGE_PATH = ImagePath
        self.Width = 800
        self.Height = 600
        self.THRESHOLD_VALUE = 128
        self.BLUR_KERNEL_SIZE = (5, 5)
        self.BRIGHTNESS_FACTOR = 1.5
        try:
            self.input_image = cv.imread(self.IMAGE_PATH)
            self.image = cv.resize(self.input_image, (self.Width, self.Height))
            self.gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"Error in Image processing: {str(e)}")
            return None
        self.create_color_histograms()
        self.extract_edge_features()

if __name__ == "__main__":
    App = Main("IMAGES/1.png")

