from pen_world.P3.RobinsonTriangle import RobinsonTriangle, psi, psi2
import pen_world.P3.BtileL as BtileL

class BtileS(RobinsonTriangle):
    """
    A class representing a "B_S" Penrose tile in the P3 tiling scheme as
    a "small" Robinson triangle (sides in ratio 1:1:psi).

    """
    def __init__(self, A, B, C):
        self.type = "B_S"
        super(BtileS, self).__init__(A, B, C)

    def inflate(self):
        """
        "Inflate" this tile, returning the two resulting Robinson triangles
        in a list.

        """
        D = psi * self.A + psi2 * self.B
        return [BtileS(D, self.C, self.A),
                BtileL.BtileL(self.C, D, self.B)]
