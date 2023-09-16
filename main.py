import pytesseract as tes
from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np
import math
import regex as re
import platform

#Assuming we are only running on either Windows or Linux OS
OSSystem=platform.system()
if OSSystem == 'Windows':
    print("Running on Windows OS")
    tes.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
else:
    print("Running on Linux OS")
    tes.pytesseract.tesseract_cmd = r'tesseract'


def main():
    image = Image.open('COLDATA.jpg')

    width, height = image.size

    # parameters for COLDATA.jpg
    left = width / 2.53
    up = height / 2.9
    right = width / 2.08
    down = height / 2.5

    # parameters for One_Asic (note this does not work as the image is poor quality. Only used for testing)
    # left = width / 6
    # up = height / 3.2
    # right = width / 1.4
    # down = height / 1.7

    im = convert_image(image, left, right, up, down)

    # Convert the image to text
    text = tes.pytesseract.image_to_string(im)


    # [0] = type of chip, [2] = serial_number, [3] = lot number
    text = text.split("\n")

    print("type_of_chip: " + text[0] + ", " + "serial_number: " + text[2] + ", " + "lot_number: " + text[3])


def convert_image(im, left, right, up, down):
    # Size of im in pixels
    
    # Crop
    im1 = im.crop((left, up, right, down))

    # Resize
    im1.resize((im.width * 4, im.height * 4))

    im1 = im1.convert('L')

    # Convert to Binary Image
    (thresh, BWimg) = cv2.threshold(np.array(im1), 117, 255, cv2.THRESH_BINARY)
    im1 = Image.fromarray(BWimg)

    # Invert Image
    im1 = ImageOps.invert(im1)

    # Shows the image in image viewer
    im1.show()

    return im1


if __name__ == "__main__":
    main()
