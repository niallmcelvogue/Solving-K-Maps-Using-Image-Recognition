# import the necessary packages
from preprocessing import preprocessed
from SlidingWindow import pyramid
from SlidingWindow import sliding_window
from tensorflow.keras.models import load_model
import time
import cv2
import numpy as np
import imutils
from imutils.contours import sort_contours


# construct the argument parser and parse the arguments
# load the image and define the window width and height
# apply image preprocessing
image = cv2.imread('images/kmapgrid.jpg')
(winW, winH) = (128, 128)
image = preprocessed(image)

# load model
print("[INFO] loading handwriting OCR model...")
model = load_model('C:\\Users\\Niall\\OneDrive\\Documents\\Uni\\ELE3001\\Solving-K-Maps-Using-Image-Recognition\\modelling\\handwriting.model')

for resized in pyramid(image, scale=1.5):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        cropped = resized[y:y + winH, x:x + winW]
        cropped = np.array(cropped, dtype="float32")
        preds = model.predict(cropped)
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)
