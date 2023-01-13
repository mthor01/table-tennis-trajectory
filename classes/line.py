import numpy as np

class line:
    def __init__(self, pos1=[], pos2=[], angle=None):
        self.pos1 = np.array(pos1)
        self.pos2 = np.array(pos2)
        self.angle = angle

    def pos1_tuple(self):
        return (int(self.pos1[0]), int(self.pos1[1]))

    def pos2_tuple(self):
        return (int(self.pos2[0]), int(self.pos2[1]))