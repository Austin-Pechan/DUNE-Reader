import cv2

class ImageCropper:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.crop_coordinates = []
        self.crop_in_progress = False
        self.crop_image = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.crop_coordinates = [(x, y)]
            self.crop_in_progress = True
        elif event == cv2.EVENT_LBUTTONUP and self.crop_in_progress:
            self.crop_coordinates.append((x, y))
            self.crop_in_progress = False
            self.crop_image = self.crop_selected_region()

    def crop_selected_region(self):
        x_start, y_start = self.crop_coordinates[0]
        x_end, y_end = self.crop_coordinates[-1]
        cropped_image = self.image.copy()
        cv2.rectangle(cropped_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow('Image Cropper', cropped_image)
        return self.image[y_start:y_end, x_start:x_end]

    def run(self):
        cv2.namedWindow('Image Cropper')
        cv2.setMouseCallback('Image Cropper', self.mouse_callback)

        while True:
            preview_image = self.image.copy()

            if self.crop_in_progress:
                x_start, y_start = self.crop_coordinates[0]
                x_end, y_end = self.crop_coordinates[-1]
                cv2.rectangle(preview_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

            cv2.imshow('Image Cropper', preview_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if self.crop_image is not None:
                cv2.imshow('Cropped Image', self.crop_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()
    def perform_cropping(self):
        cv2.namedWindow('Image Cropper')
        cv2.setMouseCallback('Image Cropper', self.mouse_callback)

        while True:
            preview_image = self.image.copy()

            if self.crop_in_progress:
                x_start, y_start = self.crop_coordinates[0]
                x_end, y_end = self.crop_coordinates[-1]
                cv2.rectangle(preview_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

            cv2.imshow('Image Cropper', preview_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if self.crop_image is not None:
                break

        cv2.destroyAllWindows()
        return self.crop_image 

if __name__ == "__main__":
    image_path = 'ColdADC_test_images/New_FEMB_photos/FEMB_88PF_10PL_2sidebars_788ms.png' 
    cropper = ImageCropper(image_path)
    cropped_image = cropper.perform_cropping()
    cv2.imshow('Final cropped image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
