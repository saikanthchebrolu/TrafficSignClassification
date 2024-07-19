from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the model outside of the function
model = load_model('./model/TSR1.h5')

# Classes of traffic signs
classes ={
    0: 'Aphids',
    1: 'Army_worm',
    2: 'Bacterial_Blight',
    3: 'Healthy',
    4: 'powdery_Mildew',
    5: 'Target_spot'
}

def image_processing(img):
    #data=[]
    #image = Image.open(img)
    #image = image.resize((30,30))
    #data.append(np.array(image))
    #X_test=np.array(data)
    #Y_pred = model.predict(X_test)
    #return Y_pred
    data=[]
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = model.predict(X_test)
    return Y_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processing(file_path)
        s = [str(i) for i in result]
        probabilities = [float(value) for value in s[0].strip('[]\n').split()]
        predicted_class_index = np.argmax(probabilities)
        print("Predicted traffic sign is: ", classes[predicted_class_index])
        os.remove(file_path)
        return classes[predicted_class_index]
    return None

if __name__ == '__main__':
    app.run(debug=True)