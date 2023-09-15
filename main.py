import pytesseract as tes
from PIL import Image, ImageOps
import cv2
import numpy as np
import regex as re
tes.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


def main():
    image = Image.open('One_ASIC_Image.jpg')

    im = convert_image(image)

    # Convert the image to text
    text = tes.pytesseract.image_to_string(im)

    # Parses out everything but numbers 0-9
    regex = re.compile(r'[^0-9]')
    text = regex.sub('', text)

    print("ASIC_Serial_Number: " + text)


def convert_image(im):
    # Size of im in pixels
    width, height = im.size
    
    left = width / 6
    top = height / 2.2
    right = width / 2.25
    bottom = height / 1.94
    
    # Crop
    im1 = im.crop((left, top, right, bottom))

    # Resize
    im1.resize((width * 2, height * 2))

    # Convert to Binary Image
    (thresh, BWimg) = cv2.threshold(np.array(im1), 100, 255, cv2.THRESH_BINARY)
    im1 = Image.fromarray(BWimg)

    #Invert Image
    #im1 = ImageOps.invert(im1)

    # Shows the image in image viewer
    #im1.show()

    return im1


if __name__ == "__main__":
    main()