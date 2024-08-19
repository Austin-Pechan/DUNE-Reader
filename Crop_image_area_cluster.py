from PIL import Image
import pytesseract as tes
import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import platform
import cv2
from manually__crop import manually_cropped_image

#Assuming we are only running on either Windows or Linux OS
OSSystem=platform.system()
if OSSystem == 'Windows':
    tes.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
else:
    tes.pytesseract.tesseract_cmd = r'tesseract'


def crop_image(image, side):
    cropped_images = []
    if side == 1:
        #coldata 1
        img = image.crop((500, 100, 1700, 1100))
        cropped_images.append(img)
        #coldata 2
        img = image.crop((2200, 100, 3600, 1100))
        cropped_images.append(img)
        
        #coldADC 1
        img = image.crop((400, 950, 1200, 1600))
        cropped_images.append(img)
        #coldADC 2
        img = image.crop((1100, 950, 1900, 1600))
        cropped_images.append(img)
        #coldADC 3
        img = image.crop((2000, 950, 3000, 1600))
        cropped_images.append(img)
        #coldADC 4
        img = image.crop((2800, 950, 3700, 1600))
        cropped_images.append(img)

        #larasic 1
        img = image.crop((400, 1600, 1200, 2300))
        cropped_images.append(img)
        #larasic 2
        img = image.crop((1100, 1600, 1900, 2300))
        cropped_images.append(img)
        #larasic 3
        img = image.crop((2000, 1600, 3000, 2300))
        cropped_images.append(img)
        #larasic 4
        img = image.crop((2800, 1600, 3700, 2300))
        cropped_images.append(img)
    elif side == 2: 
        #coldADC 1
        img = image.crop((400, 950, 1200, 1600))
        cropped_images.append(img)
        #coldADC 2
        img = image.crop((1100, 950, 1900, 1600))
        cropped_images.append(img)
        #coldADC 3
        img = image.crop((2000, 950, 3000, 1600))
        cropped_images.append(img)
        #coldADC 4
        img = image.crop((2800, 950, 3700, 1600))
        cropped_images.append(img)

        #larasic 1
        img = image.crop((400, 1600, 1200, 2300))
        cropped_images.append(img)
        #larasic 2
        img = image.crop((1100, 1600, 1900, 2300))
        cropped_images.append(img)
        #larasic 3
        img = image.crop((2000, 1600, 3000, 2300))
        cropped_images.append(img)
        #larasic 4
        img = image.crop((2800, 1600, 3700, 2300))
        cropped_images.append(img)
    elif side == 3:
        img = image.crop((1000, 1100, 5350, 2925))
        img = image.crop((1300, 1100, 5350, 2925))
        x_1=1000
        y_1 = 1100
        for i in range(1,16):
            y_1 = 1100
            for i in range(1,7):
                img = image.crop((x_1, y_1, x_1+310, y_1+300))
                img = img.rotate(90)
                cropped_images.append(img)
                y_1 += 300
            x_1 += 285
        print(len(cropped_images))


    return cropped_images

def original_contour_cropping(image, tc_lowerbound):
    image_array = np.array(image)
    tolerance = 20
    largest_contour = None
    largest_contour_area = 0
    min_contour_area = 22000
    min_aspect_ratio = 0.9
    max_aspect_ratio = 1.1

    for i in range(8):
        target_color = np.array([tc_lowerbound[0] + 5 * i, tc_lowerbound[1] + 5 * i, tc_lowerbound[2] + 5 * i])
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

        resize_factor = 1
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

        # Find the largest contour
        if contours:
            for contour in contours:
                current_contour_area = cv2.contourArea(contour)
                current_aspect_ratio = get_aspect_ratio(contour)

                # Filter contours based on minimum area and aspect ratio
                if current_contour_area > min_contour_area and min_aspect_ratio < current_aspect_ratio < max_aspect_ratio:
                    if current_contour_area > largest_contour_area:
                        largest_contour = contour
                        largest_contour_area = current_contour_area

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = Image.fromarray(image_array[y:y+h, x:x+w])
        # cv2.imshow('img', np.array(cropped_image))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return cropped_image

    return None

def contour_image(image, tc_lowerbound):
    original_cropped_image = original_contour_cropping(image, tc_lowerbound)

    if original_cropped_image is not None:
        # Convert the PIL Image to a NumPy array before using cv2.imshow
        return original_cropped_image
    else:
        print("Original contour-based cropping returned None. Performing manual cropping...")
        manual_cropped_image = manually_cropped_image(image)

        if manual_cropped_image is not None:
            # Convert the PIL Image to a NumPy array before using cv2.imshow
            cv2.imshow('Manual Cropped Image', np.array(manual_cropped_image))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return manual_cropped_image

    return None

def get_aspect_ratio(contour):
    _, _, w, h = cv2.boundingRect(contour)
    return float(w) / h if h != 0 else 0

def final_cropping(image, tc_lowerbound, side):
    cropped_images = crop_image(image, side)
    refined_cropped_images = []
    for i, cropped_image in enumerate(cropped_images):
        refined_cropped_images.append(contour_image(cropped_image, tc_lowerbound))
    return refined_cropped_images


def main():
    original_image = Image.open('Irvine_tray_1.JPG')
    # 1 = front of board (10 chips), 2 = back (8 chips) 3 = Irvine tray
    final_cropping(original_image, [75, 80, 95], 3)      
            
if __name__ == "__main__":
    main()
