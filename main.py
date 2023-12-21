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

from scipy.signal import convolve2d
from scipy.signal import wiener
def convert_image(im):
    kernel_size = 4

    # Create a simple averaging kernel
    blur_kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2

    # Convolve the image with the averaging kernel to simulate blurring
    blurred_image = convolve2d(im, blur_kernel, 'same', 'symm')

    # Apply Wiener deconvolution
    deblurred_image = wiener(blurred_image)

    # Ensure the deblurred image is in the valid intensity range [0, 255]
    deblurred_image = np.clip(deblurred_image, 0, 255).astype(np.uint8)

    # Convert the deblurred image to a PIL Image
    im1 = Image.fromarray(deblurred_image)

    # Display the deblurred image
    im1.show()


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
