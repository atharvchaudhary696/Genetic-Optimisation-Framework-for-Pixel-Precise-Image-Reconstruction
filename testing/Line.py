import sys
import LineParameters
import Image_reconstruction
sys.path.append('Modules')
import Display_Image 

if __name__ == "__main__":
    LineParameters.Main("Output\InputImages\slanting_lines.png",debug=False)
    Image_reconstruction.Main(csv_file_path="Output\Csv\Input_Image_Line_parameters.csv",debug=False)
    Display_Image.display_images_in_folder("Output\OutputImages")
    