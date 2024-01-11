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

#Assuming we are only running on either Windows or Linux OS
OSSystem=platform.system()
if OSSystem == 'Windows':
    print("Running on Windows OS")
    tes.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
else:
    print("Running on Linux OS")
    tes.pytesseract.tesseract_cmd = r'tesseract'

def convert_image(im):
    im = Image.fromarray(im)
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(30)
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


def text_output(im):
    custom_config = r'--psm 6'
    text = tes.image_to_string(im, lang='eng', config=custom_config)

    # [0] = type of chip, [2] = serial_number, [3] = lot number
    text = text.split("\n")

    #sparse out empty array elements
    text = [re.sub('[^a-zA-Z0-9/-]', '', x) for x in text if x.strip() != '']
    text = [re.sub('[^a-zA-Z0-9/-]', '', x) for x in text if x.strip() != '']

    larasic = None
    for i in range(len(text)):
        if get_close_matches(text[i], ["ColdADC", "COLDATA"], cutoff=0.5):
            larasic = False
            break
        elif get_close_matches(text[i], ["LArASIC"], cutoff=0.5):
            larasic = True
            break

    bnl = True 
    lar = True
    if larasic:
        i = 0
        while i < len(text):
            if bnl:
                close_matches = get_close_matches(text[i], ["BNL"], cutoff=0.5)
                if close_matches and close_matches[0] == "BNL":
                    bnl = False
                    text[i] = 'BNL'
                else:
                    text.pop(i)
                    i -= 1
            elif lar:
                close_matches_lar = get_close_matches(text[i], ["LArASIC"], cutoff=0.5)
                if close_matches_lar and close_matches_lar[0] == "LArASIC":
                    lar = False
                    text[i] = 'LArASIC'
                else:
                    text.pop(i)
                    i -= 1
            i += 1

    print(text)
    return(text)

class ImageError(Exception):
    def __init__(self, message):
        super().__init__(message)
        

def full_test(image, side):
    array_of_images = Crop_image.contour_image(image)
    array_of_text = []

    if side == 1:
        if len(array_of_images) != 10:
            raise ImageError("contouring failed, please retake the image and try again")
    elif side == 2:
        if len(array_of_images) != 8:
            raise ImageError("contouring failed, please retake the image and try again")

    # for i in array_of_images:
    #     cv2.imshow(f'Cropped Image {i}', i)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    for i in array_of_images:
        im1 = convert_image(i)
        array_of_text.append(text_output(im1))


def main():
    image = Image.open('ColdADC_test_images/Full_test_2.jpg')
    #set parameter two to 1 if it is the front side of the chip or 2 if it is the back side
    full_test(image, 2)

if __name__ == "__main__":
    main()
