import pytesseract as tes
from pyzbar.pyzbar import decode
import cv2
import numpy as np

def read_qr_code1(image):

    qcd = cv2.QRCodeDetector()

    retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(image)
    if retval:
        print("the qr scan worked")
    else:
        print("the qr scan failed")

def read_qr_code2(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the pyzbar library to decode QR codes
    qr_codes = decode(gray)

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
    qr_image = cv2.imread('ColdADC_test_images/qr_code_polarized.png')
    # im0 = make_background_black(qr_image)
    im = add_quiet_zone(qr_image, 60)
    original_height, original_width, _ = im.shape

    # Scale up all dimensions by 10
    scaled_width = original_width * 2
    scaled_height = original_height * 2

    # Resize the image
    im = cv2.resize(im, (scaled_width, scaled_height))
    cv2.imshow('Result Image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    read_qr_code1(im)
    read_qr_code2(im)

if __name__ == "__main__":
    main()