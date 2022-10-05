from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import cv2
from PIL import Image
import numpy as np

app = Flask(__name__)


def predict_label(img_path):
	model = load_model("model/ct-scan-model.h5")

	img = cv2.imread(img_path)
	img = Image.fromarray(img)
	img = img.resize((224, 224))
	img = np.array(img)
	img = np.expand_dims(img, axis=0)

	pred = model.predict(img)
	return pred[0]

# routes
@app.route("/")
def main():
	return render_template("ctscan.html")


@app.route("/predictLCimg", methods = ['GET', 'POST'])
def get_output():
	dic ={ 0:"Adenocarcinoma", 1:"Carcinoma", 2:"Normal", 3:"Squamous"}

	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename
		img.save(img_path)

		p = np.argmax(predict_label(img_path))
		# print(p)


	return render_template("ctscan.html", prediction = dic[p], img_path = img_path)


if __name__ =='__main__':
	app.run(debug = True)
