import numpy as np
import random

def onestep(s, stepsize, threshold, scale_level=3,stepnum=1):
    Nums = np.shape(s)[0]
    slist = [i for i in range(Nums)]
    # Choose a random dimension
    samples = random.sample(slist, stepnum)
    # update the dimension with the samples
    for samplepoint in samples:
        if (s[samplepoint] >= 0):
            if (s[samplepoint] > threshold[samplepoint]):
                s[samplepoint] = s[samplepoint] + random.choice([-1]) * stepsize[samplepoint]
            elif ((abs(s[samplepoint] - threshold[samplepoint])) <= 1e-6):
                s[samplepoint] = s[samplepoint] + random.choice([0, -1]) * stepsize[samplepoint]
            else:
                s[samplepoint] = s[samplepoint] + random.choice([-1, 0, 1]) * stepsize[samplepoint]
        elif (s[samplepoint] < 0):
            if (s[samplepoint] < (-threshold[samplepoint])):
                s[samplepoint] = s[samplepoint] + random.choice([1]) * stepsize[samplepoint]
            elif ((abs(s[samplepoint] + threshold[samplepoint])) <= 1e-6):
                s[samplepoint] = s[samplepoint] + random.choice([0, 1]) * stepsize[samplepoint]
            else:
                s[samplepoint] = s[samplepoint] + random.choice([-1, 0, 1]) * stepsize[samplepoint]
    s = np.round(s,scale_level)
    return s


def onemovelist(s, stepsize, threshold, movelength):
    "It is used to generate a move sequence for optical element"
    """s is the initial state of the optical element
    stepsize is the stepsize of each state 
    threshold is the threshold of each state 
    stepnum is the number of state changed in each step
    movelength is the length of the sequence"""
    Nstate = np.shape(s)[0]
    # Pay attentaion to the dimension of the vector
    s_squence = np.zeros([Nstate, movelength])
    # Generate a list of the moving
    for indmove in range(movelength):
        temmove = onestep(s, stepsize, threshold)
        temmove = np.reshape(temmove, [1, Nstate])
        s_squence[:, indmove] = temmove

    return s_squence


def trimstate(s, scale_level):
    """It is used to scale a vectors to predefined grids
    s is a state vector
    scalevector is used to define minimal accuracy"""
    s = s.round(scale_level)
    return s


class MoveState():
    """
    TotalState = 4  # Total State for optical element
    movelength = 5   # Number of move in one sequence
    TotalmoveN = 1  # Total number of movelist in one file
    stepsize = [stepsize1, stepsize1, stepsize2, stepsize2]     # Set minimal step for each move for each dimension
    threshold = [threshold1, threshold1, threshold2, threshold2]     # Set the maximal scale
     scale_level = 3  # Set the scale level to make results around predefined grid
     Totalfiles = 1   # Number of files to be generated
    """

    def __init__(self, start, TotalState, movelength, TotalmoveN, stepsize1, stepsize2, threshold1, threshold2,
                 Totalfiles=1):

        self.TotalState = int(TotalState)
        self.movelength = movelength
        self.TotalmoveN = TotalmoveN
        self.stepsize = [stepsize1, stepsize1, stepsize2, stepsize2]
        self.threshold = [threshold1, threshold1, threshold2, threshold2]
        self.scale_level = 3
        self.Totalfiles = Totalfiles
        self.start = start

    def move(self):
        # Set all move actions
        moves = np.zeros([self.TotalState, self.movelength, self.TotalmoveN])
        for nfile in range(self.Totalfiles):
            for moveind in range(self.TotalmoveN):
                s = trimstate(np.array(self.start), self.scale_level)
                temmovelist = onemovelist(s, self.stepsize, self.threshold, self.movelength)
                moves[:, :, moveind] = temmovelist

        # Reorganize
        Move = []
        for i in range(self.TotalmoveN):
            for j in range(self.movelength):  # 4,5,1000
                move = moves[:, j, i]
                mis = move
                Move.append(mis)

        Moves = np.array(Move)

        return Moves


# if __name__ == "__main__":
#     scale_level = 3  # Set the scale level to make results around predefined grid
#     INFO = [0.02, 0, 0.002, 0.004]
#    # MS = MoveState(INFO, TotalState, movelength, TotalmoveN, stepsize1, stepsize2, threshold1, threshold2)
#     MS = MoveState(INFO, 4, 5, 1, 0.02, 0.002, 0.1, 0.01)
#     move = MS.move()
# 
#     print(move)



















