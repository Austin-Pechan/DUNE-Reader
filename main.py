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


def main():
    image = Image.open('ColdADC_test_images/Full_test_2.jpg')
    #set parameter two to 1 if it is the front side of the chip or 2 if it is the back side
    full_test(image, 2)

def convert_image(im):
    im = Image.fromarray(im)
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(5)
    im = im.convert('L')
    im = np.array(im)

    resize_factor = 1
    im = cv2.resize(im, (0, 0), fx=resize_factor, fy=resize_factor)

    im_sharpened = cv2.addWeighted(im, 2, cv2.GaussianBlur(im, (0, 0), 2), -1.5, 0)
    im1 = Image.fromarray(im_sharpened)


    # im1 = im1.filter(ImageFilter.BoxBlur(1))
    im1 = im1.rotate(-90)
    im1 = im1.filter(ImageFilter.SHARPEN)


    # # Contrast
    # enhancer = ImageEnhance.Contrast(im1)
    # im1 = enhancer.enhance(5)

    # # Grayscale
    # im1 = im1.convert('L')
    # kernel_set = [3, 6, 3,
    #               6, 24, 6,
    #               3, 6, 3]
    # kernel = ImageFilter.Kernel(size=(3, 3), kernel=kernel_set)

    # # Apply the filter to the image
    # im1 = im1.filter(kernel)

    # # Convert to Binary Image
    # (thresh, BWimg) = cv2.threshold(np.array(im1), 117, 255, cv2.THRESH_BINARY)
    # im1 = Image.fromarray(BWimg)


    # Invert Image
    im1 = ImageOps.invert(im1)

    # # Denoise the image
    # im2 = np.array(im1)
    # denoised_image = denoise_tv_bregman(im2, 5)
    # threshold_value = filters.threshold_otsu(denoised_image)
    # binary_image = denoised_image > threshold_value

    # # Perform binary dilation on the binary image
    # dilated_image = morphology.binary_dilation(binary_image)


    # pil_image = Image.fromarray((dilated_image * 255).astype(np.uint8))

    # pil_image.show()

    im1.show()
    return im1


def text_output(im):
    text = tes.pytesseract.image_to_string(im)

    # [0] = type of chip, [2] = serial_number, [3] = lot number
    text = text.split("\n")

    #sparse out empty array elements
    text = [x for x in text if x]

    # # Sparse code to remove bad info
    # text[2] = re.sub('[^a-zA-Z0-9]', '', text[2])
    # text[3] = re.sub('[^a-zA-Z0-9]', '', text[3])

    # type_of_chip = get_close_matches(text[0], ["LArASIC", "ColdADC", "COLDATA"], 1)
    # final_text = "type_of_chip: " + type_of_chip[0] + ", " + "serial_number: " + text[2] + ", " + "lot_number: " + text[3]

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

if __name__ == "__main__":
    main()
