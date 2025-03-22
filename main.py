import cv2
import numpy as np
import os
import math
import time
import sys

source_folder = "driving_images"

if len(sys.argv) > 1:
    source_folder = sys.argv[2]
    print(f"Loaded source: {source_folder}")

# TODO: remove in final!
from numba import jit, njit, prange

"""
Noise: both gaussian and salt-and-pepper
Warping: use projective transforms, rotation? - rotation done
Contrast/Brightness: HSV stuff? or histogram equalisation
Colour channel imbalance: some channels brighter/darker than others
Missing region: inpainting techniques
"""

os.makedirs("./Results/", exist_ok=True)

# From lectures/you
# https://github.com/atapour/ip-python-opencv/blob/main/butterworth-low-high-pass-filter.py
def create_butterworth_low_pass_filter(width, height, d, n):
    lp_filter = np.zeros((height, width, 2), np.float32)
    centre = (width / 2, height / 2)

    for i in range(0, lp_filter.shape[1]):  # image width
        for j in range(0, lp_filter.shape[0]):  # image height
            radius = max(1, math.sqrt(math.pow((i - centre[0]), 2.0) + math.pow((j - centre[1]), 2.0)))
            lp_filter[j, i] = 1 / (1 + math.pow((radius / d), (2 * n)))
    return lp_filter

# https://medium.com/@abhishekjainindore24/salt-and-pepper-noise-and-how-to-remove-it-in-machine-learning-032d76b138a5
@njit
def adaptive_median_filter(image : np.ndarray[np.uint8], kernel_size: int, kernel_max: int):
    width, height = image.shape
    output_image = np.copy(image)
    padded_image = np.zeros((height + kernel_max, width + kernel_max), dtype=np.uint8)
    padded_image[:height, :width] = image

    for y in prange(height):
        for x in prange(width):
            kernel_current = kernel_size

            while kernel_current <= kernel_max:
                window = padded_image[y:y + kernel_current, x:x + kernel_current]
                z_min = np.min(window)
                z_max = np.max(window)
                z_median = np.median(window)
                z_xy = image[y, x]

                if z_min < z_median < z_max:
                    if z_min < z_xy < z_max:
                        output_image[y, x] = z_xy
                    else:
                        output_image[y, x] = z_median
                    
                    break
                else:
                    kernel_current += 2
            else:
                output_image[y, x] = z_xy

    return output_image

# https://www.digimizer.com/manual/contraharmonic-mean-filter.php
@njit
def contraharmonic_mean(image, kernel_size, Q):
    padded_image = np.pad(image, kernel_size // 2, mode="constant", constant_values=0).astype(np.float32)
    padded_image /= 255.0
    output_image = np.copy(image)
    rows, cols = image.shape

    for i in prange(rows):
        for j in prange(cols):
            sub_img = padded_image[i:i+kernel_size, j:j+kernel_size]
            
            with np.errstate(divide="ignore"):
                nominator = np.sum(np.power(sub_img, Q + 1.0))
                denominator = np.sum(np.power(sub_img, Q))
                value = np.nan_to_num(nominator / denominator, nan=0.0)

            output_image[i, j] = value * 255.0

    return output_image

def process_image(filename):
    """
    Open the file, process the image appropriately, and output the file to processed images
    """

    img = cv2.imread(f"./{source_folder}/{filename}")

    rows, cols, _ = img.shape

    # Median filter for salt-and-pepper noise
    for i in range(3):
        img[:,:,i] = cv2.morphologyEx(img[:,:,i], cv2.MORPH_OPEN, np.ones((1,1), np.uint8))
        img[:,:,i] = cv2.medianBlur(img[:,:,i], 3)

    # Calculate rotation
    #   Turn image into binary black/white
    #   Find contours (and get largest area)
    #   Get the rotated bounding box of biggest contour, and get angle from it
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Largest contour tends to always be the bounding box
    contourAreas = np.fromiter(map(cv2.contourArea, contours), dtype=np.float64)
    indexSort = np.argsort(contourAreas)
    sortedContours = [contours[i] for i in indexSort]
    maxContour = sortedContours[-1]
    secondMaxContour = sortedContours[-2] # Hopefully the circle

    # Get rotated bounding box
    box = cv2.boxPoints(cv2.minAreaRect(maxContour))
    # Note all images rotated the same way, so get the left most point and top most point
    left_most = min(box, key=lambda p: p[0])
    top_most = max(box, key=lambda p: p[1])
    # Now calculate theta
    angle = math.atan2(top_most[1] - left_most[1], top_most[0] - left_most[0])

    mat = cv2.getRotationMatrix2D(((cols - 1.0) / 2.0, (rows - 1.0) / 2.0), math.degrees(angle) - 90, 1.1)
    img = cv2.warpAffine(img, mat, (cols, rows))
    # Blur edges of image slightly to remove some artefacts after in-painting the edges
    bw = cv2.drawContours(bw, [maxContour], 0, 0, 3)
    bw = cv2.drawContours(bw, [secondMaxContour], 0, 0, 3)
    # Perform same warp on grayscale image
    grayscale_rotated = cv2.warpAffine(bw, mat, (cols, rows))

    # In-paint
    #   Detect region to inpaint
    mask = 255 - grayscale_rotated 
    img = cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA)

    # NLM denoising
    img = cv2.fastNlMeansDenoisingColored(img, hColor=4.0, templateWindowSize=7, searchWindowSize=21)

    # CLAHE for contrast adjustment
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
    img[:,:,0] = clahe.apply(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)

    # Sharpen edges
    laplace = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img, -1, laplace)

    cv2.imwrite(f"./Results/{filename}", img)

for (dirpath, dirnames, filenames) in os.walk(f"./{source_folder}/"):
    for filename in filenames:
        process_image(filename)