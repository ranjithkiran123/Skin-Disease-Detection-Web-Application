import numpy as np
from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = None

def load_skin_model():
    global model
    model_path = "C:/skin_disease_dataset/skin_proj.h5"
    model = load_model(model_path)

@app.route('/', methods=["GET"])
def index():
    return render_template("base.html")

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        f = request.files["image"]
        basepath = os.path.dirname(__file__)
        print("current path:", basepath)
        file_path = os.path.join(basepath, "uploads", secure_filename(f.filename))
        f.save(file_path)
        print("joined path:", file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        p = model.predict(x)
        predicted_class = np.argmax(p)
        print(p)
        index = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        
        if predicted_class < len(index):
            prediction_text = "The predicted disease is: " + index[predicted_class]
        else:
            prediction_text = "Unable to determine the disease."
        
        return prediction_text

if __name__ == "__main__":
    load_skin_model()
    app.run(debug=True)
