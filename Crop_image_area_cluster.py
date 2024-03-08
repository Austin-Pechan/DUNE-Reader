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


def crop_image(image, side):
    cropped_images = []
    if side == 0:
        img = image.crop((400, 100, 2900, 2500))
        cropped_images.append(img)
    else: 
        print("hi")

    return cropped_images

def contour_image(image, tc_lowerbound):
    image_array = np.array(image)
    tolerance = 20

    largest_contour = None
    largest_contour_area = 0

    for i in range(8):
        target_color = np.array([tc_lowerbound[0] + 5 * i, tc_lowerbound[1] + 5 * i, tc_lowerbound[2] + 5 * i])
        lower_color = np.maximum(target_color - tolerance, [0, 0, 0])
        upper_color = np.minimum(target_color + tolerance, [255, 255, 255])
        mask = np.all((image_array >= lower_color) & (image_array <= upper_color), axis=-1)
        enhanced_image_array = np.zeros_like(image_array)
        enhanced_image_array[mask] = image_array[mask]

        enhanced_image = Image.fromarray(enhanced_image_array)
        enhanced_image = ImageOps.invert(enhanced_image)
        enh = np.array(enhanced_image)
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

        for cnt in contours:
            contour_area = cv2.contourArea(cnt)
            if contour_area > largest_contour_area:
                largest_contour = cnt
                largest_contour_area = contour_area

    return largest_contour


def main():
    #"ColdADC_test_images/New_FEMB_photos/FEMB_0PF_0PL_2sidebars_800ms.png" = [38,43,50]
    #Test --> "ColdADC_test_images/New_FEMB_photos/FEMB_71PF_10PL_1s.png" = [48,52,74]
    #Test --> "ColdADC_test_images/New_FEMB_photos/FEMB_88PF_10PL_2sidebars_788ms.png" = [41,45,62]
    #"ColdADC_test_images/New_FEMB_photos/FEMB_BACK_0PF_0PL_2sidebars_800ms.png" = [29,33,38]
    #"ColdADC_test_images/New_FEMB_photos/FEMB_BACK_71PF_10PL_1s.png" = [31,33,47]
    #"ColdADC_test_images/New_FEMB_photos/FEMB_BACK_88PF_10PL_2sidebars_788ms.png" = [32,34,46]

    original_image = Image.open('ColdADC_test_images/New_FEMB_photos/FEMB_71PF_10PL_1s.png')

    # 1 = front of board (10 chips), 2 = back (8 chips) 
    cropped_images = crop_image(original_image, 0)

    for i, cropped_image in enumerate(cropped_images):
        # Apply contour_image function to each cropped image
        refined_images = contour_image(cropped_image, [48, 52, 74])

        # Display the refined images
        for j, refined_image in enumerate(refined_images):
            cv2.imshow(f'Refined Image {i}_{j}', np.array(refined_image))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()