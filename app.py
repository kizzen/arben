from flask import Flask, render_template, request #, routes
import os
import numpy as np
from PIL import Image
import tensorflow
import keras
from keras.datasets import mnist
from keras.models import model_from_json
from keras import backend as K
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
from diffimg import diff # to get the image difference
# attack libraries
import art
from art.attacks import FastGradientMethod, CarliniL2Method
from art.classifiers import KerasClassifier
import foolbox
from foolbox.criteria import Misclassification

app = Flask(__name__)

### FLASK ###
@app.route('/',methods=['POST', 'GET'])
def home():  

	###load parameters###
	def hyper_params():
		global img_rows, img_cols, channels, num_classes, params, batch_size
		img_rows, img_cols = 28, 28 # image dimensions
		channels=1 # channel for black and white
		num_classes = 10 # 0 through 9 digits as class
		params = [32, 32, 64, 64, 200, 200] # parameter for the CNN
		batch_size = 128 # batch size
	hyper_params()
	print('parameters created')

	###load MNIST data###
	def load_data():

	    global x_train, x_test, y_train, y_test # set as global variables
	    # load and split data between test and train set
	    (x_train, y_train), (x_test, y_test) = mnist.load_data()

	    # data transformation for model
	    from keras import backend as K
	    if K.image_data_format() == 'channels_first':
	        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
	        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
	        input_shape = (channels, img_rows, img_cols)
	    else:
	        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
	        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
	        input_shape = (img_rows, img_cols, channels)

	    x_train = x_train.astype('float32')
	    x_test = x_test.astype('float32')
	    x_train /= 255
	    x_test /= 255

	    # convert class vectors to binary class matrices
	    y_train = keras.utils.to_categorical(y_train, num_classes)
	    y_test = keras.utils.to_categorical(y_test, num_classes)
	load_data() # execute load daya function
	print('MNIST data loaded')

	###load models###
	def load_models():
		global undistilled_model, distilled_model
		# Model reconstruction from JSON file
		with open('static/undistilled2architecture_CNN.json', 'r') as f:
		    undistilled_model = model_from_json(f.read())

		# Load weights into the new model
		undistilled_model.load_weights('static/undistilled2weights_CNN.h5')

		# Model reconstruction from JSON file # say how Keras load module could not be used directly
		with open('static/distilled2architecture_CNN.json', 'r') as f:
		    distilled_model = model_from_json(f.read())

		# Load weights into the new model
		distilled_model.load_weights('static/distilled2weights_CNN.h5')
	load_models()
	print('Models loaded')

	attack_select = request.form.get("attack")
	print('ATTACK?', type( attack_select))
	if attack_select == 'fgsm':
		attack_type_leg = 'FGSM'
	elif attack_select == 'cw':
		attack_type_leg = 'CW'
	else:
		attack_select = random.choice(['fgsm','cw'])
		if attack_select == 'fgsm':
			attack_type_leg = 'FGSM'
		elif attack_select == 'cw':
			attack_type_leg = 'CW'

	distillation_select = request.form.get("distillation")
	print('DISTILLATION?', distillation_select)
	if distillation_select == 'undistilled':
		model = undistilled_model
		cnn_type_leg = 'Undistilled CNN Prediction'
	elif distillation_select == 'distilled':
		model = distilled_model
		cnn_type_leg = 'Distilled CNN Prediction'
	# else statement for when app first open and no selection made (random selection)
	else:
		distillation_select = random.choice(['undistilled','distilled'])
		if distillation_select == 'undistilled':
			model = undistilled_model
			cnn_type_leg = 'Undistilled CNN Prediction'
		elif distillation_select == 'distilled':
			model = distilled_model
			cnn_type_leg = 'Distilled CNN Prediction'

	# random generator to randomly select image
	randnum = random.randint(0,9999)
	x = x_test[randnum]
	y = y_test[randnum].argmax()

	pred_x = np.reshape(x,[1,28,28,1]) # image reshape for prediction
	prediction = int(model.predict_classes(pred_x)) # prediction 

	# plot and save image to calculate noise
	plt.imshow(x.reshape((28,28)), cmap='Greys')
	plt.tight_layout()
	pylab.savefig('static/original_diff.png')

	# plot and save image to be displayed on the screen
	plt.title('Original Image', fontsize = 20)
	plt.xlabel('True Class: {} \n {}: {}'.format(y,cnn_type_leg,prediction),fontsize=15) 
	plt.imshow(x.reshape((28,28)), cmap='Greys')
	plt.tight_layout()
	pylab.savefig('static/original.png')


	if attack_select == 'fgsm':
		classifier = KerasClassifier(clip_values=(0, 255), model=model)
		epsilon = 0.2
		adv_crafter = FastGradientMethod(classifier)
		x_art = np.reshape(x,[1,28,28,1])
		img_adv = adv_crafter.generate(x=x_art, eps=epsilon)

	elif  attack_select == 'cw':
		classifier = KerasClassifier(clip_values=(0, 255), model=model)
		adv = CarliniL2Method(classifier, targeted=False, max_iter=100, binary_search_steps=2, learning_rate=1e-2, initial_const=1)
		img_adv = adv.generate(x.reshape(1,28,28,1))

	pred_advimg = np.reshape(img_adv,[1,28,28,1]) # reshape of adversarial image
	prediction_adv = int(model.predict_classes(pred_advimg))

	# plot and save adv image to calculate noise
	plt.imshow(img_adv.reshape((28,28)), cmap='Greys')
	plt.tight_layout()
	pylab.savefig('static/adversarial_diff.png')

	# calculate noise level
	im_diff = round(diff('static/original_diff.png', 'static/adversarial_diff.png') * 100,2)

	# plot and save image to be displayed on the screen
	plt.title('{} Adversarial Image'.format(attack_type_leg), fontsize = 20)
	plt.xlabel('Noise level: {}% \n {}: {}'.format(im_diff,cnn_type_leg,prediction_adv),fontsize=15)
	plt.imshow(img_adv.reshape((28,28)), cmap='Greys')
	plt.tight_layout()
	pylab.savefig('static/adversarial.png')

	K.clear_session()

	return render_template('enternum.html')

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

if __name__ == '__main__':
	local_fpath = '/Users/khalilezzine/Desktop/DS/flasky/arben'
	cwd = os.getcwd()
	if cwd == local_fpath:
		app.run(debug=True)
	else:
		app.run(host='seaford.nsqdc.city.ac.uk', debug=True)




