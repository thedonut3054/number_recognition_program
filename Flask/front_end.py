from flask import Flask, request, redirect, url_for, render_template_string, session
from markupsafe import escape
from werkzeug.utils import secure_filename
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import torchvision
import torchvision.transforms.v2 as v2
import torchvision.io as io
import matplotlib.pyplot as plt
from torchvision.transforms.v2 import InterpolationMode
import Network.Network_definition

def transform_image():
    image_path = "/workspaces/number_recognition_program/feed_image/image_to_scan/scan.jpg"
    if not os.path.exists(image_path):
        return "No image uploaded. Please upload a JPG file first."
    
    image = io.read_image(image_path)
    original_size = (216, 216)
    pixel_size = 28

    gray = v2.Grayscale(1)
    try:
        image = gray(image)
    except:
        pass
    transform = v2.Compose([
        v2.Resize(pixel_size, interpolation=InterpolationMode.NEAREST),
        v2.Lambda(lambda x: torch.where(x > 0, 1.0, 0.0)),
        v2.RandomInvert(1),
    ])
    image = transform(image)
    plt.imshow(image.permute(1, 2, 0).detach().cpu(), cmap='gray')
    plt.savefig('compressed_image.png')
    return read_image(image)

def read_image(image):
    model = torch.load('/workspaces/number_recognition_program/Saved_Model', weights_only=False)
    model.eval()
    image = image.float()
    input = image.unsqueeze(0)
    with torch.no_grad():
        prediction = model(input)
    predicted_class = prediction.argmax(dim=1).item()
    return predicted_class

# from feed_image.transform_image import transform_image

base_style = """
<style>
body {
    font-family: Arial, sans-serif;
    background: #f4f6f8;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    width: 300px;
    text-align: center;
}
input {
    width: 90%;
    padding: 8px;
    margin: 8px 0;
    border: 1px solid #ccc;
    border-radius: 5px;
}
button {
    padding: 10px;
    width: 60%;
    background: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}
button:hover {
    background: #45a049;
}
a {
    display: block;
    margin-top: 10px;
    color: #333;
    text-decoration: none;
}
.error {
    color: red;
    margin-top: 10px;
}
</style> """

main_page = base_style + """
<div class="card">
<h2>Upload a photo</h2>
<form method="POST" enctype="multipart/form-data" action="{{ url_for('main') }}">
      <input type=file name=file>
      <input type=submit value=Upload>
</form>
<form action="{{ url_for('run_my_code') }}" method="post">
    <button type="submit">Execute Code</button>
</form>
<p>{{output}}<p>
</div> """

UPLOAD_FOLDER = '/workspaces/number_recognition_program/feed_image/image_to_scan'
ALLOWED_EXTENSIONS = {'jpg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def main():
    output = "Upload a JPG file to read"
    if request.method == "POST":
        if 'file' not in request.files:
            output = "No file part"
        else:
            f = request.files['file']
            if f.filename == '':
                output = "No selected file"
            elif not allowed_file(f.filename):
                output = "Invalid file type. Please upload a JPG file."
            else:
                save_path = '/workspaces/number_recognition_program/feed_image/image_to_scan/scan.jpg'
                if os.path.exists(save_path):
                    os.remove(save_path)
                f.save(save_path)
                output = "File Uploaded"
    return render_template_string(main_page, output=output)

@app.route('/run-my-code', methods=['POST'])
def run_my_code():
    temp = str(transform_image())
    print(temp)
    return render_template_string(main_page, output=temp)