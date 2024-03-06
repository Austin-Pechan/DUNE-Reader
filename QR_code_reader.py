import pytesseract as tes
from pyzbar.pyzbar import decode
import cv2
import numpy as np
from PIL import Image

def read_qr_code(image):
    qr_codes = decode(image)

    # Check if any QR codes were detected
    if qr_codes:
        # Get the data from the first QR code
        data = qr_codes[0].data.decode('utf-8')

        # Print the QR code data
        print(f"QR Code Data: {data}")

        return data
    else:
        print("No QR code found in the image.")
        return None
    
def preprocess_image(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Increase contrast
    enhancer = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    contrast_img = enhancer.apply(blurred)

    # Apply adaptive thresholding
    _, thresholded = cv2.threshold(contrast_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use morphological operations to improve QR code visibility
    kernel = np.ones((3, 3), np.uint8)
    thresholded = cv2.erode(thresholded, kernel, iterations=1)
    thresholded = cv2.dilate(thresholded, kernel, iterations=1)

    return thresholded

def make_background_black(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask of white pixels
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Create an output image with black background
    result = np.zeros_like(image, dtype=np.uint8)

    # Set the pixels in the output image to white where the mask is white
    result[mask > 0] = [255, 255, 255]

    return result
def add_quiet_zone(image, quiet_zone_size):
    # Get the dimensions of the original image
    height, width, channels = image.shape

    # Calculate the new dimensions including the quiet zone
    new_height = height + 2 * quiet_zone_size
    new_width = width + 2 * quiet_zone_size

    # Create a new image with a quiet zone
    new_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    # Copy the original image into the center of the new image
    new_image[quiet_zone_size:quiet_zone_size+height, quiet_zone_size:quiet_zone_size+width, :] = image

    return new_image
def main():
    qr_image = cv2.imread('ColdADC_test_images/New_FEMB_photos/FEMB_0PF_0PL_2sidebars_800ms.png')
    top_left_x = 1050
    top_left_y = 1500
    bottom_right_x = 1200
    bottom_right_y = 1650
    cropped_image = qr_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    # cv2.imshow('Cropped Section', cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    im = add_quiet_zone(cropped_image, 60)
    original_height, original_width, _ = im.shape

    scaled_width = original_width * 2
    scaled_height = original_height * 2

    im = cv2.resize(im, (scaled_width, scaled_height))
    im = preprocess_image(im)
    cv2.imshow('Preprocessed Image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    read_qr_code(im)

if __name__ == "__main__":
    main()