from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the traffic sign classification model
model = load_model('./model/TSR.h5')

# Load the pretrained image classification model (e.g., ResNet50)
# Make sure you have the model file and the appropriate preprocessing function
from keras.applications.resnet50 import ResNet50, preprocess_input

# Instantiate the ResNet50 model
image_classifier = ResNet50(weights='imagenet')

# Classes of traffic signs
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Vehicle > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing vehicle > 3.5 tons' }

# Threshold probability to consider it as a traffic sign
threshold_prob = 0.5

def image_processing(img):
    # Load the image
    image = Image.open(img)
    image = image.resize((224, 224))  # Resize to match the input size of ResNet50
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Use ResNet50 to predict if the image contains a traffic sign
    resnet_prediction = image_classifier.predict(image)

    # If ResNet50 predicts a non-traffic image with high confidence, return "Not a traffic sign"
    if np.max(resnet_prediction) > 0.9:
        return "Not a traffic sign"

    # Use the traffic sign classification model
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
        
        # If the result is "Not a traffic sign", return immediately
        if result == "Not a traffic sign":
            os.remove(file_path)
            return result
        
        s = [str(i) for i in result]
        probabilities = [float(value) for value in s[0].strip('[]\n').split()]
        
        # Check if any class has probability greater than threshold
        if max(probabilities) >= threshold_prob:
            predicted_class_index = np.argmax(probabilities)
            print("Predicted traffic sign is: ", classes[predicted_class_index])
            os.remove(file_path)
            return classes[predicted_class_index]
        else:
            print("Not a traffic sign")
            os.remove(file_path)
            return "Not a traffic sign"
    return None

if __name__ == '__main__':
    app.run(debug=True)

