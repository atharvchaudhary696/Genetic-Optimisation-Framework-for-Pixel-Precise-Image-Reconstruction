from flask import Flask, jsonify, send_from_directory , request
from flask_cors import CORS
import os
import json
import sys
sys.path.append("FlaskBackend/Scripts")
import Main

app = Flask("FlaskBackend")
CORS(app)
UPLOAD_FOLDER = "FlaskBackend/UPLOAD_FOLDER"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/images/<filename>', methods=['GET'])
def serve(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
@app.route('/upload',methods=['POST'])
def save_file():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"})
    
    file=request.files['image']
    boolean_value = request.form.get('booleanValue')# This is the value to start your python Main Function,
   

    Main.Main(Apprunning=bool(boolean_value),image_path="FlaskBackend\\UPLOAD_FOLDER\\UploadedImage.jpg",FolderPath="FlaskBackend\\UPLOAD_FOLDER")

    if file.filename == '':
        return jsonify({"error": "No selected file"})
  

    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return jsonify({"message": "File successfully uploaded"})
    else:
        return jsonify({"error": "File format not supported"})

if __name__ == '__main__':
    app.run(debug=True)
