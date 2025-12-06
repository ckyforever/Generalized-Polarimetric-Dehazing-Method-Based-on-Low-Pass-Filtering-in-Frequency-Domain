import numpy as np

class Stokes:
    def __init__(self, i_0, i_45, i_90, i_135):
        self.DoP = None
        self.AoP = None
        self.s_1 = None
        self.s_0 = None
        self.s_2 = None
        self.s_3 = None
        self.i_0 = i_0
        self.i_45 = i_45
        self.i_90 = i_90
        self.i_135 = i_135

    def calculate_stokes(self):
        self.s_0 = self.i_0 + self.i_90
        self.s_1 = self.i_0 - self.i_90
        self.s_2 = self.i_45 - self.i_135
        self.s_3 = self.i_45 + self.i_135
        return self.s_0, self.s_1, self.s_2, self.s_3

    def AoP(self):
        self.AoP = np.arctan(self.s_2/self.s_3) / 2

    def DoP(self):
        self.DoP = np.sqrt(self.s_1 ** 2 + self.s_2 ** 2) / self.s_0
        return self.DoP
