import matplotlib.pyplot as plt
from joblib.numpy_pickle_utils import xrange
from skimage.feature import hog
from skimage import data, color, exposure
import cv2
import numpy
import pickle
import gaussian_downsampling
import nms
import matplotlib.pyplot
import math

scale_factor = 1.2
test_path = "/Users/15195/Desktop/ece613/test.jpg"


def visualize_boxes(final_proposal_boxes, image_path_local):
    raw_image = cv2.imread(image_path_local)
    for point in final_proposal_boxes:
        cv2.rectangle(raw_image, (point[0], point[1]), (point[2], point[3]), (0,255,0), 2)
    cv2.imshow('testing the image', raw_image)
    cv2.waitKey(2)


def scale(scale_factor_local, depth):  # formatting done
    scale_value = math.pow(scale_factor_local, depth - 1)
    return scale_value


def scaled_proposal_boxes(proposal_boxes, scale_factor_local):  # formatting done
    proposals = []
    for current_depth, current_rectangle in proposal_boxes.items():
        if not current_depth == 1:
            for rectangle in current_rectangle:
                scaling_value = scale(scale_factor_local, current_depth)
                l1 = round(rectangle[0] * scaling_value)
                l2 = round(rectangle[1] * scaling_value)
                l3 = round(rectangle[2] * scaling_value)
                l4 = round(rectangle[3] * scaling_value)
                new_proposals = [int(l1), int(l2), int(l3), int(l4)]
                proposals.append(new_proposals)

        elif current_depth == 1:
            for rectangle in current_rectangle:
                proposals.append(rectangle)
    return proposals


def main_detector(test_path_local, scale_factor_local, counter):
    svm_model = pickle.load(open("/Users/15195/Desktop/ece613/trained_svm_model.p", 'rb'))
    size_of_window = [128, 64]
    size_of_block = 2
    size_of_cell = 8
    minimum_h = 128
    minimum_w = 64
    orientation = 9
    threshold = -1

    weight = svm_model.coef_
    bias = svm_model.intercept_
    test_raw_image = cv2.imread(test_path_local)
    test_raw_image = color.rgb2gray(test_raw_image)

    # calculate the total block size
    total_size = size_of_block * size_of_cell
    confidence_scores = []
    proposal_boxes = {}
    depth = 0

    for image in gaussian_downsampling.gaussian_reduction(test_raw_image, scale_factor_local, minimum_h, minimum_w):
        depth += 1
        height = image.shape[0]
        width = image.shape[1]
        dimension_feature = weight.shape[1]
        negative_mining_count = 0
        for h in xrange(0, height, int(total_size / 2)):
            for w in xrange(0, width, int(total_size / 2)):
                if size_of_window[1] + w <= width and size_of_window[0] + h <= height:
                    fd, hog_image = hog(image[h:(size_of_window[0] + h), w: (size_of_window[1] + w)], orientations=orientation, pixels_per_cell=(size_of_cell, size_of_cell), cells_per_block=(size_of_block, size_of_block), block_norm='L1', visualize=True, transform_sqrt=False)
                    calculate_score = numpy.dot(numpy.reshape(fd, (1, dimension_feature)), numpy.transpose(weight)) + bias
                    print(calculate_score[0][0], depth)
                    if calculate_score[0][0] >= threshold:
                        print("score {} and depth {}: ".format(calculate_score[0][0], depth))
                    if not counter == 1:
                        """negative mining to be done here"""
                        negative_image = image[h:(size_of_window[0] + h), w: (size_of_window[1] + w)]
                        matplotlib.pyplot.imshow(negative_image)
                        matplotlib.pyplot.savefig("C/Users/15195/Desktop/ece613/negative_mining")
                        negative_mining_count += 1
                    else:
                        confidence_scores.append(calculate_score[0][0])
                        rectangle = [w, h, w + size_of_window[1], h + size_of_window[0]]
                        if depth in proposal_boxes:
                            proposal_boxes[depth].append(rectangle)
                        else:
                            proposal_boxes[depth] = [rectangle]
    if not counter == 1:
        """do nothing"""
    else:
        scaled_proposal_box = scaled_proposal_boxes(proposal_boxes, scale_factor_local)
        return scaled_proposal_box, confidence_scores


counter_global = 1


def call_detector():
    if counter_global == 1:
        final_proposal_boxes, final_confidence_scores = main_detector(test_path, scale_factor, counter_global)
        print("Final Boxes: {}".format(len(final_proposal_boxes)))
        print("Final Scores : {}".format(len(final_confidence_scores)))

        # to check the number of rectangles before applying the non-max suppression
        visualize_boxes(final_proposal_boxes, test_path)

        # to find the non max suppress boxes
        final_proposal_boxes = nms.nm_s(final_proposal_boxes, final_confidence_scores)
        visualize_boxes(final_proposal_boxes, test_path)


if __name__ == "__main__":
    call_detector()
