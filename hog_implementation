import pickle
import random
from os import walk
import cv2
from skimage import color
from skimage.feature import hog
import numpy


def extract_pos_hog_features(got_path, num_samples):
    feature_of_image = []
    count = 0
    for directory_path, directory_names, name_of_files in walk(got_path):
        for got_file in name_of_files:
            print(got_path + got_file)
            if count < num_samples:
                count = count + 1
                im = cv2.imread(got_path + got_file)
                print(im.shape)
                image = color.rgb2gray(im)
                image = image[17:145, 16:80]
                my_feature, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                    visualize=True)
                feature_of_image.append(my_feature)
    return feature_of_image


def extract_neg_hog_features(path, num_samples):
    feature_list = []
    count = 0
    for directory_path, directory_name, filenames in walk(path):
        for my_file in filenames:
            if count < num_samples:
                count = count + 1
                im = cv2.imread(path + my_file)
                image = color.rgb2gray(im)
                image = image[17:145, 16:80]
                my_feature, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                    visualize=True)
                feature_list.append(my_feature)
    return feature_list


def neg_hog_rand(path, num_samples, window_size, num_window_per_image):
    rows = window_size[0]
    cols = window_size[1]
    feature_list = []
    count = 0
    for directory_path, directory_name, name_of_files in walk(path):
        for my_file in name_of_files:
            if count < num_samples:
                print(count, my_file)
                count = count + 1
                im = cv2.imread(path + my_file)
                image = color.rgb2gray(im)
                image_rows = image.shape[0]
                image_cols = image.shape[1]

                for i in range(0, num_window_per_image):
                    x_min = random.randrange(0, image_rows - rows)
                    y_min = random.randrange(0, image_cols - cols)
                    x_max = x_min + rows
                    y_max = y_min + cols
                    image_hog = image[x_min:x_max, y_min:y_max]
                    my_feature, _ = hog(image_hog, orientations=9, pixels_per_cell=(8, 8),
                                                cells_per_block=(2, 2), visualize=True)
                    feature_list.append(my_feature)
    return feature_list


image_path_pos = "/Users/15195/Desktop/ece613/INRIAPerson/Train/pos_n/"
pos_features = extract_pos_hog_features(image_path_pos, 4400)  # 4400
print(len(pos_features))

print("finished positive")

image_path_neg = "/Users/15195/Desktop/ece613/INRIAPerson/Train/neg_n/"
neg_features_rand = neg_hog_rand(image_path_neg, 2100, [128, 64], 10)
print(len(neg_features_rand))

# concatenate positive and negative hog features
features = pos_features + neg_features_rand
pickle.dump(features, open("/Users/15195/Desktop/ece613/ped_features.p", 'wb'))

pos_labels = [1] * len(pos_features)
neg_labels = [-1] * len(neg_features_rand)

labels = pos_labels + neg_labels
pickle.dump(labels, open("/Users/15195/Desktop/ece613/peds_feature_to_label.p", 'wb'))
print("done hog extraction")
