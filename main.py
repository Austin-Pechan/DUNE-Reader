import pytesseract as tes
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import cv2
import numpy as np
import regex as re
from difflib import get_close_matches
from skimage import io, img_as_ubyte, filters, morphology
from skimage.restoration import denoise_tv_bregman
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
    image = Image.open('DUNE-Reader/ColdADC_test_images/Full_test_1.jpg')

    width, height = image.size

    # note: coordinates start in top left of the image

    # parameters for im1: this image is unreadable with my current methods
    # left = width / 7
    # up = height / 6
    # right = width
    # down = height / 1.4

    # parameters for DUNE-Reader/COLDATA.jpg
    # left = width / 2.53
    # up = height / 2.9
    # right = width / 2.08
    # down = height / 2.5

    # parameters for One_Asic (note this does not work as the image is poor quality. Only used for testing)
    # left = width / 6
    # up = height / 3.2
    # right = width / 1.35
    # down = height / 1.7

    # im = convert_image(image, left, right, up, down)

    # # Convert the image to text
    # text_output(im)

    full_test_1(image)


def convert_image(im, left, right, up, down):

    # Crop
    im1 = im.crop((left, up, right, down))

    # Resize
    im1.resize((im.width * 4, im.height * 4))

    im1 = im1.filter(ImageFilter.BoxBlur(1))
    
    # Contrast
    enhancer = ImageEnhance.Contrast(im1)
    im1 = enhancer.enhance(4.5)

    # Grayscale
    im1 = im1.convert('L')

    kernel_set = [1, 4, 1,
                  4, 12, 4,
                  1, 4, 1]
    kernel = ImageFilter.Kernel(size=(3, 3), kernel=kernel_set)

    # Apply the filter to the image
    im1 = im1.filter(kernel)

    # Convert to Binary Image
    (thresh, BWimg) = cv2.threshold(np.array(im1), 117, 255, cv2.THRESH_BINARY)
    im1 = Image.fromarray(BWimg)

    # Invert Image
    im1 = ImageOps.invert(im1)

    # Load the image
    # image = np.array(im1)

    # # Denoise the image
    # denoised_image = denoise_tv_bregman(image, 5)
    # threshold_value = filters.threshold_otsu(denoised_image)
    # binary_image = denoised_image > threshold_value

    # # Perform binary dilation on the binary image
    # dilated_image = morphology.binary_dilation(binary_image)


    # pil_image = Image.fromarray((dilated_image * 255).astype(np.uint8))
    # # Shows the image in image viewer
    # pil_image.show()

    im1.show()
    return im1



def text_output(im):
    text = tes.pytesseract.image_to_string(im)

    # [0] = type of chip, [2] = serial_number, [3] = lot number
    text = text.split("\n")

    #print(text)

    # sparse out empty array elements
    text = [x for x in text if x]

    # Sparse code to remove bad info
    text[2] = re.sub('[^a-zA-Z0-9]', '', text[2])
    text[3] = re.sub('[^a-zA-Z0-9]', '', text[3])

    type_of_chip = get_close_matches(text[0], ["LArASIC", "ColdADC", "COLDATA"], 1)

    final_text = "type_of_chip: " + type_of_chip[0] + ", " + "serial_number: " + text[2] + ", " + "lot_number: " + text[3]

    print(final_text)
    return(final_text)


def full_test_1(image):
    width, height = image.size
    image =  image.rotate(270)
    #im_1 = convert_image(image, width/2.55, width/2.25, height/3.4, height/2.7)
    im_2 = convert_image(image, width/2.55, width/2.25, height/1.66, height/1.52)

    print(tes.pytesseract.image_to_string(im_2))

#     return LArASIC_1, LArASIC_2, LArASIC_3, LArASIC_4, 
# ColdADC_1, ColdADC_2, ColdADC_3, ColdADC_4, ColdATA_1, ColdATA_2


if __name__ == "__main__":
    main()
