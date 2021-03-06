from flask import Flask, render_template, request # routes
import os
import numpy as np
from PIL import Image
import tensorflow
import keras
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import model_from_json
from keras.models import load_model
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
# from art.utils import load_dataset
import foolbox
from foolbox.criteria import Misclassification
#import smtplib
from flask_mail import Mail, Message 

app = Flask(__name__)

app.config.update(
	DEBUG=True,
	#EMAIL SETTINGS
	MAIL_SERVER='smtp.gmail.com',
	MAIL_PORT= 465,
	MAIL_USE_SSL=True,
	MAIL_USERNAME = 'karben1917@gmail.com',
	MAIL_PASSWORD = 'adversarial!'
	)

mail = Mail(app)

### FLASK ###

@app.route('/MNIST',methods=['POST', 'GET'])
def mnistpage():  
	
	###load parameters###
	img_rows, img_cols = 28, 28 # image dimensions
	channels=1 # channel for black and white
	num_classes = 10 # 0 through 9 digits as class
	params = [32, 32, 64, 64, 200, 200] # parameter for the CNN
	batch_size = 128 # batch size
	
	print('parameters created')

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

	model_select = request.form.get("model")
	print('Model', model_select)
	if model_select == "Undistilled CNN":
		model = undistilled_model
		cnn_type_leg = 'Undistilled CNN Prediction'
	elif model_select == "Distilled CNN":
		model = distilled_model
		cnn_type_leg = 'Distilled CNN Prediction'
	# else statement for when app first open and no selection made (random selection)
	else:
		model_select = random.choice(['undistilled','distilled'])
		if model_select == 'undistilled':
			model = undistilled_model
			cnn_type_leg = 'Undistilled CNN Prediction'
		elif model_select == 'distilled':
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
			
	return render_template('mnist.html')

@app.route('/CIFAR',methods=['POST', 'GET'])
def cifarpage():  

	params = [64, 64, 128, 128, 256, 256]
	num_channels = 3
	img_rows = 32
	img_cols = 32
	num_labels = 10
	batch_size = 128 # batch size
	channels=3
	num_classes = 10

	# # load and split data between test and train set
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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
			
	###load models###
	def load_models():

		global undistilled_model, distilled_model

		undistilled_model = load_model("static/CIFAR_CNN_undistilled2.h5")

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
		model = undistilled_model
		cnn_type_leg = 'Undistilled CNN Prediction'
	# else statement for when app first open and no selection made (random selection)
	else:
		model = undistilled_model
		cnn_type_leg = 'Undistilled CNN Prediction'

	mapping_cifarlabels = ['airplane','automobile','bird','cat','deer',
	'dog','frog','horse','ship','truck'] 

	print('x_test shape: ',x_test.shape)
	randnum = random.randint(0,10000)
	x = x_test[randnum]
	y = y_test[randnum].argmax()
	y = mapping_cifarlabels[int(y)]

	pred_x = np.reshape(x,[1,32,32,3]) # image reshape for prediction
	prediction = int(model.predict_classes(pred_x)) # prediction



	prediction = mapping_cifarlabels[prediction]
	
	# plot and save image to calculate noise
	plt.imshow(x.reshape((32,32,3)), cmap='Greys')
	plt.tight_layout()
	pylab.savefig('static/original_diff.png')

	# plot and save image to be displayed on the screen
	plt.title('Original Image', fontsize = 20)
	plt.xlabel('True Class: {} \n {}: {}'.format(y,cnn_type_leg,prediction),fontsize=15) 
	plt.imshow(x.reshape((32,32,3)), cmap='Greys')
	plt.tight_layout()
	pylab.savefig('static/original.png')
	
	print('attack_select: ', attack_select)
	if attack_select == 'fgsm':
		# classifier = KerasClassifier(clip_values=(min_, max_), model=model)
		classifier = KerasClassifier((0.0, 1.0), model=undistilled_model)
		epsilon = 0.2
		adv_crafter = FastGradientMethod(classifier)
		x_art = np.reshape(x,[1,32,32,3])
		img_adv = adv_crafter.generate(x=x_art, eps=epsilon)
	
	elif  attack_select == 'cw':
		# classifier = KerasClassifier(clip_values=(min_, max_), model=model)
		classifier = KerasClassifier(clip_values=(0.0, 1.0), model=model)
		adv = CarliniL2Method(classifier, targeted=False, max_iter=100, binary_search_steps=2, learning_rate=1e-2, initial_const=1)
		img_adv = adv.generate(x.reshape(1,32,32,3))

	pred_advimg = np.reshape(img_adv,[1,32,32,3]) # reshape of adversarial image
	prediction_adv = int(model.predict_classes(pred_advimg))
	prediction_adv = mapping_cifarlabels[prediction_adv]

	# plot and save adv image to calculate noise
	plt.imshow(img_adv.reshape((32,32,3)), cmap='Greys')
	plt.tight_layout()
	pylab.savefig('static/adversarial_diff.png')

	# calculate noise level
	im_diff = round(diff('static/original_diff.png', 'static/adversarial_diff.png') * 100,2)

	# plot and save image to be displayed on the screen
	plt.title('{} Adversarial Image'.format(attack_type_leg), fontsize = 20)
	plt.xlabel('Noise level: {}% \n {}: {}'.format(im_diff,cnn_type_leg,prediction_adv),fontsize=15)
	plt.imshow(img_adv.reshape((32,32,3)), cmap='Greys')
	plt.tight_layout()
	pylab.savefig('static/adversarial.png')

	K.clear_session()
			
	return render_template('cifar.html')

@app.route('/',methods=['POST', 'GET'])
@app.route('/Home',methods=['POST', 'GET'])
def home():
	return render_template('home.html')

@app.route('/Leaderboard',methods=['POST', 'GET'])
def leaderboard():
	return render_template('leaderboard.html')

@app.route('/Contact',methods=['POST', 'GET'])
def contact():
	print(request.method)
	if request.method == 'GET':
		print("test")
		fname = request.args.get("fname","")
		 
	return render_template('contact.html')

@app.route('/Confirmation',methods=['POST', 'GET'])
def confirm():
	if request.method == 'POST':
		fname = request.form["firstname"]
		email = request.form["emailadd"]
		msg = 'Sender: ' + fname + '\n' + 'Email: ' + email + '\n\n' + request.form["msg"]
		receiver = ['khalil.ezzine@city.ac.uk']

		# try:
		msg = Message(subject='Arben Request',recipients=receiver,body=msg,sender="karben1939@gmail.com") 
              # sender="karben1939@outlook.com",
              # recipients=[receiver])     
		mail.send(msg)
		print ("Successfully sent email")
		# except:
		print ("Error: unable to send email")



	return render_template('Contact_Return.html')

@app.route('/ImageNet',methods=['POST', 'GET'])
def imagenet_page():
	return  render_template('imagenet.html')

@app.route('/Audio_Samples',methods=['POST', 'GET'])
def audio_samples_page():
	return  render_template('audio_samples.html')

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




