import numpy as np
import cv2 as cv

class camera:
    def __init__(self, pos, rotation, valid, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width
        self.valid = valid
        self.pos = np.array(pos)
        self.rotation = np.array(rotation)
        self.intrinsics = np.array([(img_width, 0, img_width / 2),
                                          (0, img_height, img_height / 2),
                                          (0, 0, 1)])

        self.rotation_matrix, _ = cv.Rodrigues(self.rotation)

        self.extrinsics = np.concatenate((self.rotation_matrix, np.transpose([self.pos])), axis=1)
        self.homo_extrinsics = np.concatenate((self.extrinsics, [np.array([0, 0, 0, 1])]), axis=0)