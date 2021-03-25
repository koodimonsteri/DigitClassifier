
import data_decoder as dc
import numpy as np
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

import torch

from util import *

def mlp_optimizer(train_images, train_labels):

	train_images2 = train_images / train_images.max()

	mlp = MLPClassifier()
	grid_params = {
				'solver'             : ['adam'],
				'activation'         : ['relu', 'logistic'],
				'learning_rate'      : ['constant'],
				'hidden_layer_sizes' : [100, 150, 200, 500, 1000]}

	gs = GridSearchCV(mlp, grid_params, cv=3, verbose=10, n_jobs=-1)
	gs.fit(train_images, train_labels)
	print("Best score:", gs.best_score_)
	print("With params", gs.best_params_)


def train_mlp_model(train_images, train_labels, test_images, test_labels):
	mlp = MLPClassifier(hidden_layer_sizes=(150,),
						activation="relu",
						solver="adam",
						max_iter=50,
						verbose=True)

	# Train nn
	mlp.fit(train_images, train_labels)
	#print("Image:\n", train_images[0], "\nLabel: ", train_labels[0])
	# Predict with train and test data
	pred_train = mlp.predict(train_images)
	pred_test = mlp.predict(test_images)
	#print("nn outputs:", mlp.n_outputs)

	print("Traindata confusion matrix:")
	print(confusion_matrix(train_labels, pred_train))
	print("Traindata classification report:")
	print(classification_report(train_labels, pred_train))

	print("Testdata confusion matrix:")
	print(confusion_matrix(test_labels, pred_test))
	print("Testdata classification report:")
	print(classification_report(test_labels, pred_test))

	print("Accuracy score:", accuracy_score(test_labels, pred_test))

	return mlp


def main():

    train_images = dc.get_images('''../datasets/train-images-idx3-ubyte.gz''')
    train_labels = dc.get_labels('''../datasets/train-labels-idx1-ubyte.gz''')
    test_images = dc.get_images('''../datasets/t10k-images-idx3-ubyte.gz''')
    test_labels = dc.get_labels('''../datasets/t10k-labels-idx1-ubyte.gz''')
    train_images1 = train_images.reshape((len(train_images), IMG_SIZE))
    test_images1 = test_images.reshape((len(test_images), IMG_SIZE))
    train_images2 = train_images1 / 255.0
    test_images2 = test_images1 / 255.0
    train_images2 = np.where(train_images2 > 0.5, 1.0, 0.0)
    test_images2 = np.where(test_images2 > 0.5, 1.0, 0.0)
    
    mlp = train_mlp_model(train_images2, train_labels, test_images2, test_labels)
    #file_name = "MLPModel2.pkl"
    #pickle.dump(mlp, open(file_name, "wb"))
	

	#nn_optimizer(train_images2, train_labels)
	#print("n_outputs:", mlp.n_outputs_)
	#model = pickle.load(open(file_name, 'rb'))

if __name__ == "__main__":
	main()