# Import required libraries
import keras_ocr
import matplotlib.pyplot as plt
import Crop_image
from PIL import Image, ImageEnhance
import main 
import numpy as np
import cv2

im = Image.open('ColdADC_test_images/Full Test data/IMG_5152.jpg')
im = Crop_image.contour_image(im)[1]
im = Image.fromarray(im)
enhancer = ImageEnhance.Contrast(im)
im = enhancer.enhance(20)
im = np.array(im)
_, im = cv2.threshold(im, 215, 255, cv2.THRESH_BINARY)

resize_factor = 1
im = cv2.resize(im, (0, 0), fx=resize_factor, fy=resize_factor)

im_sharpened = cv2.addWeighted(im, 2, cv2.GaussianBlur(im, (0, 0), 2), -1.5, 0)
im1 = Image.fromarray(im_sharpened)
im = np.array(im)
pipeline = keras_ocr.pipeline.Pipeline()

# Convert the NumPy array to a list of images
images = [im]

# Perform OCR on the image
predictions = pipeline.recognize(images)

# Display OCR results
for text, box in predictions[0]:
    print(f'Text: {text}, Bounding Box: {box}')

# Visualize the results by drawing bounding boxes on the image
keras_ocr.tools.drawAnnotations(image=images[0], predictions=predictions[0])

# Display the image with annotations
plt.imshow(images[0])
plt.show()