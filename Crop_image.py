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

# if it is the front side set side to 0, else set it to 1
side = 1
image = Image.open('ColdADC_test_images/Full_test_2.jpg')

image_array = np.array(image)
target_color = np.array([47, 33, 14])
tolerance = 15

lower_color = np.maximum(target_color - tolerance, [0, 0, 0])
upper_color = np.minimum(target_color + tolerance, [255, 255, 255])
mask = np.all((image_array >= lower_color) & (image_array <= upper_color), axis=-1)
enhanced_image_array = np.zeros_like(image_array)
enhanced_image_array[mask] = image_array[mask]


enhanced_image = Image.fromarray(enhanced_image_array)
enhanced_image = ImageOps.invert(enhanced_image)
enhancer = ImageEnhance.Contrast(enhanced_image)
enhanced_image = enhancer.enhance(100)
enhanced_image.show()

# text = tes.pytesseract.image_to_string(enhanced_image)
# print(text)



im1 = np.array(enhanced_image)

resize_factor = 0.2
im1 = cv2.resize(im1, (0, 0), fx=resize_factor, fy=resize_factor)


gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 100, 150)  # Adjust the thresholds based on your needs

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_contour_area = 1000
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

for idx, contour in enumerate(filtered_contours):
    x, y, w, h = cv2.boundingRect(contour)
    print(f"Contour {idx + 1} Bounding Box: (x={x}, y={y}, w={w}, h={h})")
    cv2.drawContours(im1, [contour], -1, (0, 255, 0), 2)
    cv2.putText(im1, f'Contour {idx + 1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


cv2.drawContours(im1, filtered_contours, -1, (0, 255, 0), 2)
cv2.imshow('Processed Image with Contours', im1)

cv2.waitKey(0)
cv2.destroyAllWindows()







