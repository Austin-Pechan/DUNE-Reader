from PIL import Image
import pytesseract as tes
import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import platform
import cv2

#Assuming we are only running on either Windows or Linux OS
OSSystem=platform.system()
if OSSystem == 'Windows':
    tes.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
else:
    tes.pytesseract.tesseract_cmd = r'tesseract'


def contour_image(image):
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
    enhancer = ImageEnhance.Contrast(enhanced_image)
    enhanced_image = enhancer.enhance(1000)

    im1 = np.array(enhanced_image)

    resize_factor = 0.7
    im1 = cv2.resize(im1, (0, 0), fx=resize_factor, fy=resize_factor)


    image_float = im1.astype(np.float32)
    brightness_increase = 200
    brightened_image = image_float + brightness_increase
    brightened_image = np.clip(brightened_image, 0, 255)
    brightened_image = brightened_image.astype(np.uint8)

    gray = cv2.cvtColor(brightened_image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 11)

    edges = cv2.Canny(blurred, 100, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 6000
    min_aspect_ratio = 0.9
    max_aspect_ratio = 1.1

    def get_aspect_ratio(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return float(w) / h if h != 0 else float('inf')

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    filtered_contours = [cnt for cnt in filtered_contours if min_aspect_ratio < get_aspect_ratio(cnt) < max_aspect_ratio]


    # image_with_contours = np.copy(im1)
    # cv2.drawContours(image_with_contours, filtered_contours, -1, (0, 255, 0), 2)
    # cv2.imshow('Enhanced Image with Contours', image_with_contours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contour_coordinates = [cv2.boundingRect(cnt) for cnt in filtered_contours]
    im2 = cv2.resize(image_array, (0, 0), fx=resize_factor, fy=resize_factor)
    cropped_images = [im2[y:y+h, x:x+w] for x, y, w, h in contour_coordinates]

    # Sort the images based on their coordinates
    sorted_data = sorted(zip(cropped_images, contour_coordinates), key=lambda x: (x[1][1], x[1][0]))

    sorted_images, sorted_coordinates = zip(*sorted_data)

    return sorted_images

def main():
    image = Image.open('ColdADC_test_images/Full Test data/IMG_5152.jpg')
    cropped_images = contour_image(image)

    for i, cropped_image in enumerate(cropped_images):
        cv2.imshow(f'Cropped Image {i}', cropped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
