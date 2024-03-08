import pytesseract as tes
from pyzbar.pyzbar import decode
import cv2
import numpy as np
from PIL import Image
from qreader import QReader

def read_qr_code(image, to_binary=False):
    qreader = QReader()
    qr_image = image
    if to_binary:
        _, qr_image = cv2.threshold(qr_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Decode the QR code from the cropped image
    try:
        # data = read(qr_image)  # Uncomment this line if needed
        data = qreader.detect_and_decode(image=qr_image)
        if data:
            return data
        else:
            return "QR code not detected!"
    except Exception as e:
        print(f"Error decoding QR code: {e}")
        return "QR code not detected!"



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
def qr_code_full(image):
    qr_image = image.crop((600, 1000, 1600, 2000))
    qr_image = np.array(qr_image)
    im = add_quiet_zone(qr_image, 60)
    final_txt = read_qr_code(im)
    return final_txt

def main():
    qr_image = Image.open('ColdADC_test_images/New_FEMB_photos/Test2/With_Polarizer_Ring/FEMB_BACK_2PBars_10PL_88PF_1s.png')
    txt = qr_code_full(qr_image)
    print(txt)

if __name__ == "__main__":
    main()