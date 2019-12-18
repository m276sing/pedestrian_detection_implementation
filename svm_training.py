import pickle
from sklearn import svm
from sklearn.svm import LinearSVC
import numpy

def train_svm(features, labels, reg_param, kernel_type):
	clf = svm.SVC(C = reg_param, kernel = kernel_type)
	svm_model = clf.fit(features, labels)
	print("fitting model done!")
	return svm_model


features_of_images = pickle.load(open("/Users/15195/Desktop/ece613/ped_features.p", 'rb'))
labels_of_images = pickle.load(open("/Users/15195/Desktop/ece613/peds_feature_to_label.p", 'rb'))

svm_model = train_svm(features_of_images, labels_of_images, 0.01, 'linear')

pickle.dump(svm_model, open("/Users/15195/Desktop/ece613/trained_svm_model.p", 'wb'))
