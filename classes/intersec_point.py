import numpy as np

class intersec_point:
    def __init__(self, pos=[], line1_index=None, line2_index=None):
        self.pos = np.array(pos)
        self.line1_index = line1_index
        self.line2_index = line2_index

    def pos_tuple(self):
        return (int(self.pos[0]), int(self.pos[1]))