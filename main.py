import pytesseract as tes
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import cv2
import numpy as np
import regex as re
from difflib import get_close_matches
from skimage import io, img_as_ubyte, filters, morphology
from skimage.restoration import denoise_tv_bregman
import platform
import Crop_image
from pyzbar.pyzbar import decode


#Assuming we are only running on either Windows or Linux OS
OSSystem=platform.system()
if OSSystem == 'Windows':
    print("Running on Windows OS")
    tes.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
else:
    print("Running on Linux OS")
    tes.pytesseract.tesseract_cmd = r'tesseract'

def convert_image(im):
    im = increase_brightness(im, .75)
    im = Image.fromarray(im)
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(50)
    im = im.convert('L')
    im = np.array(im)
    _, im = cv2.threshold(im, 215, 255, cv2.THRESH_BINARY)

    resize_factor = 1
    im = cv2.resize(im, (0, 0), fx=resize_factor, fy=resize_factor)

    im_sharpened = cv2.addWeighted(im, 2, cv2.GaussianBlur(im, (0, 0), 2), -1.5, 0)
    im1 = Image.fromarray(im_sharpened)


    # im1 = im1.filter(ImageFilter.BoxBlur(1))
    im1 = im1.rotate(-90)
    im1 = im1.filter(ImageFilter.SHARPEN)



    # Invert Image
    im1 = ImageOps.invert(im1)

    # im1.show()
    return im1

def increase_brightness(image, factor=1.5):
    image_float = image.astype(np.float32)
    brightened_image = image_float * factor
    brightened_image = np.clip(brightened_image, 0, 255)
    brightened_image = brightened_image.astype(np.uint8)

    return brightened_image


def text_output(im):
    custom_config = r'--psm 6'
    text = tes.image_to_string(im, lang='eng', config=custom_config)

    # [0] = type of chip, [2] = serial_number, [3] = lot number
    text = text.split("\n")

    text = [re.sub('[^a-zA-Z0-9/ -.]', '', x) for x in text if x.strip() != '']
    text = [re.sub('[^a-zA-Z0-9/ -.]', '', x) for x in text if x.strip() != '']

    # print("this is the pre text: ", text)
    larasic = None
    for i in range(len(text)):
        if get_close_matches(text[i], ["ColdADC", "COLDATA"], cutoff=0.5):
            larasic = False
            break
        elif get_close_matches(text[i], ["LArASIC"], cutoff=0.3):
            larasic = True
            break

    cold = True
    lar = True
    vers = True
    if larasic:
        i = 0
        while i < len(text):
            if lar:
                found_larasic = False
                close_matches_lar = get_close_matches(text[i], ["LArASIC"], cutoff=0.3)
                if close_matches_lar and close_matches_lar[0] == "LArASIC":
                    lar = False
                    text[i] = 'LArASIC'
                    found_larasic = True
                else:
                    text.pop(i)
                    i -= 1
                if found_larasic:
                    text.insert(0, "BNL")
                    i += 1
            elif vers:
                words = text[i].strip("'").split()
                close_matches_vers = get_close_matches(words[0], ["Version"], cutoff=0.5)
                if close_matches_vers and close_matches_vers[0] == "Version":
                    vers = False
                    if len(words) == 2:
                        text[i] = 'Version ' + words[1]
                    if text[i][9] == 'S':
                        text[i] = text[i][:9] + '5' + text[i][10:]
                    if text[i][10] == '8':
                        text[i] = text[i][:10] + 'B'
                    text = [re.sub('[^a-zA-Z0-9/-]', '', x) if j > i else x for j, x in enumerate(text) if x.strip() != '']
                else:
                    text.pop(i)
                    i -= 1                       
            i += 1
        if len(text) >= 5:
            text = text[:5]
            if len(text[3]) == 5 and text[3][2] != '/':
                text[3] = text[3][:2] + '/' + text[3][3:]

    else:
        text = [re.sub('[^a-zA-Z0-9.]', '', x) for x in text if x.strip() != '']
        i = 0
        while i < len(text):
            if cold:
                close_matches = get_close_matches(text[i], ["ColdADC", "COLDATA"], cutoff=0.6)
                if close_matches:
                    cold = False
                    text[i] = close_matches[0]
                else:
                    text.pop(i)
                    i -= 1
            i += 1
        text = text[:4]

    print(text)
    return(text)

def read_qr_code(image_path):
    # Read the image
    image = cv2.imread(image_path)

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


class ImageError(Exception):
    def __init__(self, message):
        super().__init__(message)
        

def full_test(image, side):
    array_of_images = Crop_image.contour_image(image)
    array_of_text = []

    if side == 1:
        read_qr_code(image)
        if len(array_of_images) != 10:
            raise ImageError("contouring failed, please retake the image and try again")
    elif side == 2:
        if len(array_of_images) != 8:
            raise ImageError("contouring failed, please retake the image and try again")

    # for i in array_of_images:
    #     cv2.imshow(f'Cropped Image {i}', i)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # for i in array_of_images:
    #     im1 = convert_image(i, 30)
    #     array_of_text.append(text_output(im1))
    for i in array_of_images:
        im1 = convert_image(i)
        array_of_text.append(text_output(im1))
        # avg = average_texts(array_of_text)
        # need to write this function if decided to go this way

def read_qr_code(image):

    qcd = cv2.QRCodeDetector()

    retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(image)
    print(retval)



def main():
    qr = cv2.imread('ColdADC_test_images/QR_code_test.png')
    read_qr_code(qr)
    image = Image.open('ColdADC_test_images/FEMB_populated_5.png')
    #set parameter two to 1 if it is the front side of the chip or 2 if it is the back side
    full_test(image, 1)

if __name__ == "__main__":
    main()
