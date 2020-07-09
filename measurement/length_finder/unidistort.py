from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import pickle
import settings,os
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())
with open(settings.CALIB_FILE_NAME, 'rb') as f:
    calib_data = pickle.load(f)
    mtx = calib_data["cam_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]
# load the image, convert it to grayscale, and blur it slightly
for image_file in os.listdir(args["image"]):
        if image_file.endswith("jpg"):
            image = cv2.imread(os.path.join(args["image"], image_file))
            cv2.namedWindow('image',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 502, 376)
            print(image_file)
            # for i in range(2):
            #     image=cv2.pyrDown(image)
            image=cv2.undistort(image, mtx, dist_coeffs)
            # image = cv2.resize(image,(500,500))
            cv2.imwrite("ds2.jpg",image)
            cv2.waitKey(0)
