from PIL import Image
import pytesseract as tes
import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import platform

#Assuming we are only running on either Windows or Linux OS
OSSystem=platform.system()
if OSSystem == 'Windows':
    tes.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
else:
    tes.pytesseract.tesseract_cmd = r'tesseract'


import cv2
import numpy as np

image = Image.open('ColdADC_test_images/Full_test_2.jpg')

image_array = np.array(image)
target_color = np.array([47, 33, 14])
tolerance = 20

lower_color = np.maximum(target_color - tolerance, [0, 0, 0])
upper_color = np.minimum(target_color + tolerance, [255, 255, 255])
mask = np.all((image_array >= lower_color) & (image_array <= upper_color), axis=-1)
enhanced_image_array = np.zeros_like(image_array)
enhanced_image_array[mask] = image_array[mask]


enhanced_image = Image.fromarray(enhanced_image_array)
enhanced_image = ImageOps.invert(enhanced_image)
# enhanced_image = ImageOps.equalize(enhanced_image)
enhancer = ImageEnhance.Contrast(enhanced_image)
enhanced_image = enhancer.enhance(100)

im1 = np.array(enhanced_image)

resize_factor = 0.2
im1 = cv2.resize(im1, (0, 0), fx=resize_factor, fy=resize_factor)


gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)


# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
blurred = cv2.medianBlur(gray, 7)

edges = cv2.Canny(blurred, 100, 150)  # Adjust the thresholds based on your needs


contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_contour_area = 1000
def get_aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / h if h != 0 else float('inf')

# Define aspect ratio range
min_aspect_ratio = 0.9
max_aspect_ratio = 1.1

# Filter contours based on area and aspect ratio
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Extract coordinates of each contour
contour_coordinates = [cnt.reshape(-1, 2) for cnt in filtered_contours]

# Filter contours based on aspect ratio
filtered_contours = [cnt for cnt in filtered_contours if min_aspect_ratio < get_aspect_ratio(cnt) < max_aspect_ratio]

# Draw contours on the original image
image_with_contours = np.copy(im1)
cv2.drawContours(image_with_contours, filtered_contours, -1, (0, 255, 0), 2)

# Display the enhanced image with contours using OpenCV
cv2.imshow('Enhanced Image with Contours', image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()