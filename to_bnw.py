import cv2
import os

orignal_image_folder = 'datasets/test/color'
orig_img_paths = []

orig_img_paths += [os.path.join(orignal_image_folder, im) for im in os.listdir(
    orignal_image_folder) if os.path.isfile(os.path.join(orignal_image_folder, im))]

for img_path in orig_img_paths:
    im = cv2.imread(img_path, 0)
    im = cv2.resize(im, (500, 500))
    cv2.imwrite('datasets/test/bnw/' + img_path.split('/')[-1], im)
