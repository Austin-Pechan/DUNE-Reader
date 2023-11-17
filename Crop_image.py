from PIL import Image
import pytesseract as tes
import numpy as np
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import draw_clusters
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

from sklearn.cluster import MiniBatchKMeans
tes.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

import cv2
import numpy as np


image = Image.open('ColdADC_test_images/Full_test_1.jpg')
image_array = np.array(image)
target_color = np.array([47, 33, 14])
tolerance = 15

lower_color = np.maximum(target_color - tolerance, [0, 0, 0])
upper_color = np.minimum(target_color + tolerance, [255, 255, 255])
mask = np.all((image_array >= lower_color) & (image_array <= upper_color), axis=-1)
enhanced_image_array = np.zeros_like(image_array)
enhanced_image_array[mask] = image_array[mask]


enhanced_image = Image.fromarray(enhanced_image_array)
enhanced_image.show()
enhanced_image = ImageOps.invert(enhanced_image)
enhancer = ImageEnhance.Contrast(enhanced_image)
enhanced_image = enhancer.enhance(100)
enhanced_image.show()

# text = tes.pytesseract.image_to_string(enhanced_image)
# print(text)



image = np.array(enhanced_image)

resize_factor = 0.2
image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 100, 150)  # Adjust the thresholds based on your needs

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_contour_area = 100
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)


cv2.imshow('Detected Chips', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


min_contour_area = 1000
max_contour_area = 10000

for idx, contour in enumerate(filtered_contours):
    contour_area = cv2.contourArea(contour)

    if min_contour_area <= contour_area <= max_contour_area:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], 0, (255), thickness=cv2.FILLED)
        roi = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow(f'ROI {idx}', roi)

cv2.waitKey(0)
cv2.destroyAllWindows()



















# import tensorflow as tf
# import cv2
# import numpy as np

# # Load the model
# model_path = 'path/to/centernet_hourglass104_1024x1024_coco17_tpu-32/saved_model'
# model = tf.saved_model.load(model_path)

# # Load an example image (replace with your own image path)
# image_path = 'One_ASIC_Image.jpg'
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = tf.convert_to_tensor(image, dtype=tf.float32)
# image = tf.expand_dims(image, 0)  # Add batch dimension

# # Perform inference
# detections = model(image)

# # Extract keypoints from the output
# keypoints = detections['keypoints'][0].numpy()

# # Draw keypoints on the image
# for keypoint in keypoints:
#     x, y, score = keypoint
#     if score > 0.5:  # You can adjust the confidence threshold
#         cv2.circle(image[0].numpy(), (int(x), int(y)), 3, (0, 255, 0), -1)

# # Display the image with keypoints
# cv2.imshow('KeyPoint Detection', image[0].numpy())
# cv2.waitKey(0)
# cv2.destroyAllWindows()











# # Load the image
# image = Image.open('ColdADC_test_images/Full_test_2.jpg')

# # Convert the image to a NumPy array
# image_array = np.array(image)

# # Get the dimensions of the image
# width, height = image.size

# # Flatten the 3D image array into a 2D array
# image_array = image_array.reshape((width * height, -1))

# # Find the optimal number of clusters
# n_clusters = 30
# kmeans = MiniBatchKMeans(n_clusters=n_clusters, max_iter=1000, random_state=0, batch_size=100)
# labels = kmeans.fit_predict(image_array)

# # Reshape labels to the original image dimensions
# labels = labels.reshape(height, width)

# # Create a PIL Image from the labels
# clustered_image = Image.fromarray(labels)

# # Show the PIL Image
# clustered_image.show()








# target_color = np.array([28, 28, 28])
# tolerance = 2

# lower_color = np.maximum(target_color - tolerance, [0, 0, 0])
# upper_color = np.minimum(target_color + tolerance, [255, 255, 255])
# mask = np.all((image_array >= lower_color) & (image_array <= upper_color), axis=-1)

# y, x = np.where(mask)

# extra_crop_pixels = 20
# regions = []

# #group pixles

# for x_center, y_center in zip(x, y):
#     x_min = max(0, x_center - extra_crop_pixels)
#     y_min = max(0, y_center - extra_crop_pixels)
#     x_max = min(image_array.shape[1], x_center + extra_crop_pixels)
#     y_max = min(image_array.shape[0], y_center + extra_crop_pixels)

#     if len(regions) > 0:
#         x, y, width, height = regions[-1]

#         if (
#             x_min >= x and x_max <= (x + width) and
#             y_min >= y and y_max <= (y + height)
#         ):
#             # Extend the current region
#             regions[-1] = (x, y, max(x + width, x_max) - x, max(y + height, y_max) - y)
#         else:
#             regions.append((x_min, y_min, x_max - x_min, y_max - y_min))
#     else:
#         regions.append((x_min, y_min, x_max - x_min, y_max - y_min))

# # crop each grouped region

# for i, (x, y, width, height) in enumerate(regions):
#     zoomed_image = image.crop((x, y, x + width, y + height))
#     zoomed_image.show()

# # Create a blank image to composite the regions
# composite_image = Image.new('RGB', image.size)

# # Crop and paste each grouped region onto the composite image
# for (x, y, width, height) in regions:
#     cropped_region = image.crop((x, y, x + width, y + height))
#     composite_image.paste(cropped_region, (x, y))

# # Display the composite image
# composite_image.show()
