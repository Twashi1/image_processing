import cv2
import numpy as np
import os
import math

"""
Noise: both gaussian and salt-and-pepper
Warping: use projective transforms, rotation? - rotation done
Contrast/Brightness: HSV stuff? or histogram equalisation
Colour channel imbalance: some channels brighter/darker than others
Missing region: inpainting techniques
"""

os.makedirs("./Results/", exist_ok=True)
os.makedirs("./contoured/", exist_ok=True)

def process_image(filename):
    """
    Open the file, process the image appropriately, and output the file to processed images
    """
    img = cv2.imread(f"./driving_images_original/{filename}")

    print(filename)

    # Median filter for salt-and-pepper noise
    img = cv2.medianBlur(img, 3)

    # Attempt to get rotation
    #   Turn image into binary black/white
    #   Find contours (and get largest area)
    #   Get the rotated bounding box of biggest contour, and get angle from it
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Largest contour tends to always be the bounding box
    maxContour = max(contours, key=lambda c: cv2.contourArea(c))
    # Get rotated bounding box
    box = cv2.boxPoints(cv2.minAreaRect(maxContour))
    # Note all images rotated the same way, so get the left most point and top most point
    left_most = min(box, key=lambda p: p[0])
    top_most = max(box, key=lambda p: p[1])
    # Now calculate theta
    angle = math.atan2(top_most[1] - left_most[1], top_most[0] - left_most[0])

    # TODO: experiment with moment
    rows, cols, _ = img.shape
    mat = cv2.getRotationMatrix2D(((cols - 1.0) / 2.0, (rows - 1.0) / 2.0), math.degrees(angle) - 90, 1.1)
    img = cv2.warpAffine(img, mat, (cols, rows))
    # Blur edges of image slightly to remove some artefacts after in-painting the edges
    bw = cv2.drawContours(bw, [maxContour], 0, 0, 5)
    # Perform same warp on grayscale image
    grayscale_rotated = cv2.warpAffine(bw, mat, (cols, rows))

    # In-paint
    #   Detect region to inpaint
    mask = 255 - grayscale_rotated 

    cv2.imwrite(f"./contoured/{filename}", mask)

    # Actually perform inpainting
    img = cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA)

    # NLM denoising
    img = cv2.fastNlMeansDenoisingColored(img, hColor=10.0, templateWindowSize=7, searchWindowSize=21)

    # CLAHE (fancy histogram equalisation)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img[0,:,:] = clahe.apply(img[0,:,:])
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite(f"./Results/{filename}", img)

for (dirpath, dirnames, filenames) in os.walk("./driving_images_original/"):
    for filename in filenames:
        process_image(filename)
