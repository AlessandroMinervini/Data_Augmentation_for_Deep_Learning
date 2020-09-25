import numpy as np
import pandas as pd
import os
import cv2
import logging
import imgaug as ia
import imgaug.augmenters as iaa
import imageio


class data_augmentator:
    def __init__(self, input_path):
        self.path = input_path
        self.base = 'augmentated/'
        self.path_rotated = 'rotated/'
        self.path_noised = 'noised/'
        self.path__dark_brightness = 'dark_brightness/'
        self.path_cropped = 'cropped/'
        self.path_flipped = 'flipped/'
        self.images = []
        self.filenames = []

    def load_images(self):
        logging.warning('Loading images')
        for filename in sorted(os.listdir(self.path)):
            self.filenames.append(filename)
            img = cv2.imread(self.path + filename)
            self.images.append(img)
        logging.warning('Done.')

    def make_out_folders(self):
        try:
            os.mkdir(self.base)
            os.mkdir(self.base + self.path_rotated)
            os.mkdir(self.base + self.path_noised)
            os.mkdir(self.base + self.path__dark_brightness)
            os.mkdir(self.base + self.path_cropped)
            os.mkdir(self.base + self.path_flipped)
        except OSError:
            print ("Creation of the directories failed")
        else:
            print ("Successfully created the directories")

    def rotate(self):
        logging.warning('Rotating')
        for img, filename in zip(self.images, self.filenames):
            rotate=iaa.Affine(rotate=(-50, 30))
            rotated_image=rotate.augment_image(img)
            cv2.imwrite(self.base + self.path_rotated + filename + '_rotated.png', rotated_image)
        logging.warning('Done.')

    def noise(self):
        logging.warning('Noising')
        for img, filename in zip(self.images, self.filenames):
            gaussian_noise=iaa.AdditiveGaussianNoise(10,20)
            noise_image=gaussian_noise.augment_image(img)
            cv2.imwrite(self.base + self.path_noised + filename + '_noised.png', noise_image)
        logging.warning('Done.')

    def brightness(self):
        logging.warning('Darking')
        for img, filename in zip(self.images, self.filenames):
            contrast=iaa.GammaContrast(gamma=2.0)
            contrast_image =contrast.augment_image(img)
            cv2.imwrite(self.base + self.path__dark_brightness + filename + '_dark_brightness.png', contrast_image)
        logging.warning('Done.')

    def crop(self):
        logging.warning('Cropping')
        for img, filename in zip(self.images, self.filenames):
            crop = iaa.Crop(percent=(0, 0.3))
            corp_image=crop.augment_image(img)
            cv2.imwrite(self.base + self.path_cropped + filename + '_cropped.png', corp_image)
        logging.warning('Done.')

    def flipup(self):
        logging.warning('Flipping')
        for img, filename in zip(self.images, self.filenames):
            flip_vr=iaa.Flipud(p=1.0)
            flip_vr_image= flip_vr.augment_image(img)
            cv2.imwrite(self.base + self.path_flipped + filename + '_flipped.png', flip_vr_image)
        logging.warning('Done.')

    def start(self):
        self.load_images()
        self.make_out_folders()
        self.rotate()
        self.noise()
        self.brightness()
        self.crop()
        self.flipup()


def main():
    path_img = 'dataset/'

    aug = data_augmentator(path_img)
    aug.start()


if __name__ == "__main__":
    main()
