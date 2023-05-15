from pen_world.P3.RobinsonTriangle import RobinsonTriangle, psi, psi2
import pen_world.P3.BtileS as BtileS

class BtileL(RobinsonTriangle):
    """
    A class representing a "B_L" Penrose tile in the P3 tiling scheme as
    a "large" Robinson triangle (sides in ratio 1:1:phi).

    """
    def __init__(self, A, B, C):
        self.type = "B_L"
        super(BtileL, self).__init__(A, B, C)

    def inflate(self):
        """
        "Inflate" this tile, returning the three resulting Robinson triangles
        in a list.

        """

        # D and E divide sides AC and AB respectively
        D = psi2 * self.A + psi * self.C
        E = psi2 * self.A + psi * self.B
        # Take care to order the vertices here so as to get the right
        # orientation for the resulting triangles.
        return [BtileL(D, E, self.A),
                BtileS.BtileS(E, D, self.B),
                BtileL(self.C, D, self.B)]
