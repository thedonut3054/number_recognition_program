from flask import Flask, request, redirect, url_for, render_template_string, session
from markupsafe import escape
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
base_style = """
<style>
body {
    font-family: Arial, sans-serif;
    background: #f4f6f8;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    enctype="multipart/form-data"
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
<form method="POST" enctype="multipart/form-data">
      <input type=file name=file>
      <input type=submit value=Upload>
</form>
<p class="error">{{ error }}</p>
</div> """

UPLOAD_FOLDER = '/workspaces/number_recognition_program/feed_image/image_to_scan'
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        f = request.files['file']
        f.save('/workspaces/number_recognition_program/feed_image/image_to_scan/scan.jpg')

    return render_template_string(main_page)