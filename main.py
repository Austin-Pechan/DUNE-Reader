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

def convert_image(im, resize):
    im = increase_brightness(im, .75)
    im = Image.fromarray(im)
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(50)
    im = im.convert('L')
    im = np.array(im)
    _, im = cv2.threshold(im, 215, 255, cv2.THRESH_BINARY)

    resize_factor = resize
    im = cv2.resize(im, (0, 0), fx=resize_factor, fy=resize_factor)

    im_sharpened = cv2.addWeighted(im, 2, cv2.GaussianBlur(im, (0, 0), 2), -1.5, 0)
    # kernel_size_1 = 1
    # kernel_1 = np.ones((kernel_size_1, kernel_size_1), np.uint8)
    # # kernel_size_2 = 2
    # # kernel_2 = np.ones((kernel_size_2, kernel_size_2), np.uint8)
    # # Erosion
    # eroded_image = cv2.erode(im_sharpened, kernel_1, iterations=1)
    # # Identify contours in the eroded image
    # contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Define a threshold size for black regions to be filtered out
    # min_contour_area = 1  # Adjust as needed

    # # Remove small black regions
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area < min_contour_area:
    #         cv2.drawContours(eroded_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # # Dilation
    # # dilated_image = cv2.dilate(eroded_image, kernel_2, iterations=1)
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
    # print("raw text: ", text)

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
                matched_ver = 0
                for h in range(len(words)):
                    close_matches_vers = get_close_matches(words[h], ["Version"], cutoff=0.5)
                    if close_matches_vers and close_matches_vers[0] == "Version":
                        matched_ver = h
                        break
                close_matches_vers = get_close_matches(words[matched_ver], ["Version"], cutoff=0.5)
                if close_matches_vers and close_matches_vers[0] == "Version":
                    vers = False
                    text[i] = 'Version '
                    if len(words) > matched_ver+1:
                        text[i] = 'Version ' + words[matched_ver+1]
                    if len(text[i]) > 10:
                        if text[i][9] == 'S':
                            text[i] = text[i][:9] + '5' + text[i][10:]
                        if text[i][10] == '8':
                            text[i] = text[i][:10] + 'B'
                        text = [re.sub('[^0-9/-]', '', x) if j > i else x for j, x in enumerate(text) if x.strip() != '']
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
        if len(text) > 1:
            if text[1].count('.') > 1:
                first_period_index = text[1].find('.')
                text[1] = text[1][:first_period_index] + text[1][first_period_index+1:]
            text = text[:4]
            text = [re.sub('[^0-9]', '', x) if i > 1 else x for i, x in enumerate(text) if x.strip() != '']

    print(text)
    return(text)


class ImageError(Exception):
    def __init__(self, message):
        super().__init__(message)
        

def full_test(image, side, tc_lowerbound):
    array_of_images = Crop_image.contour_image(image, tc_lowerbound)
    array_of_text = []

    if side == 1:
        # read_qr_code(image)
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
        im1 = convert_image(i, 1)
        array_of_text.append(text_output(im1))
        # avg = average_texts(array_of_text)
        # need to write this function if decided to go this way



def main():
    # qr = cv2.imread('ColdADC_test_images/QR_code_test.png')
    # read_qr_code(qr)
    image = Image.open('ColdADC_test_images/New_FEMB_photos/FEMB_BACK_0PF_0PL_2sidebars_800ms.png')
    #set parameter two to 1 if it is the front side of the chip or 2 if it is the back side
    full_test(image, 2, [29,33,38])

if __name__ == "__main__":
    main()
