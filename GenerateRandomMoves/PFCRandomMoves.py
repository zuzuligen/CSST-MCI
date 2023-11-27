import numpy as np
import random


class parameters():
    def __init__(self):
        # Parameters for generating random sequences
        self.TotalState= 4  # Total State for optical element
        self.movelength= 5  # Number of move in one sequence
        self.TotalmoveN= 2000   # Total number of movelist in one file

        self.stepsize1= 0.02  # Set minimal step for each move for each dimension
        self.stepsize2= 0.002

        self.threshold1= 0.1  # Set the maximal scale
        self.threshold2= 0.01  # Set the maximal scale

        self.scale_level= 3  # Set the scale level to make results around predefined grid
        self.Totalfiles= 1  # Number of files to be generated

para = parameters()


def onestep(s, stepsize, threshold, scale_level=para.scale_level,stepnum=1):
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

def onemovelist(s, stepsize, threshold, movelength, stepnum=1):
    "It is used to generate a move sequence for optical element"
    """s is the initial state of the optical element
    stepsize is the stepsize of each state defined as the same as onestep
    threshold is the threshold of each state defined as the same as s
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
    TotalmoveN = 2000  # Total number of movelist in one file
    stepsize = 0.02 * np.ones(np.size(s_min))     # Set minimal step for each move for each dimension
    threshold = 0.1 * np.ones(np.size(s_min))      # Set the maximal scale
    scale_level = 3  # Set the scale level to make results around predefined grid
    Totalfiles = 1   # Number of files to be generated
    """

    def __init__(self,TotalState, movelength, TotalmoveN, stepsize1,stepsize2, threshold1,threshold2,Totalfiles=1):

        self.TotalState = TotalState
        self.movelength = movelength
        self.TotalmoveN = TotalmoveN
        self.stepsize = [stepsize1, stepsize1,stepsize2,stepsize2,stepsize2]
        self.threshold = [threshold1, threshold1,threshold2,threshold2,threshold2]
        self.Totalfiles = Totalfiles

    def move(self):
        # Set all move actions
        moves = np.zeros([self.TotalState, self.movelength, self.TotalmoveN])
        for nfile in range(self.Totalfiles):
            # Generate Initial state, scale would help us to keep scale effective
            tt1 = para.threshold1
            ss1 = para.stepsize1
            tt2 = para.threshold2
            ss2 = para.stepsize2
            XD = np.linspace(-tt1, tt1, 2*int(tt1/ss1)+1)
            XT = np.linspace(-tt2, tt2, 2*int(tt2/ss2)+1)
            [XD, YD, XT, YT] = [XD, XD, XT, XT]
            grid_num1 = 2 * tt1 / ss1      #grid_num1=grid_num2
            i1, i2, i3, i4 = np.random.randint(0, grid_num1, para.TotalState )
            startxd, startyd, startxt, startyt = XD[i1], YD[ i2], XT[ i3], YT[i4]
            # We would set these grids to their closet values according to predefined scale
            s = trimstate(np.array([startxd, startyd, startxt, startyt]), para.scale_level)
            # Now we generate states for all moves
            for moveind in range(self.TotalmoveN):
                temmovelist = onemovelist(s, self.stepsize, self.threshold, self.movelength)
                moves[:, :, moveind] = temmovelist
                # Set the next state
                i1, i2, i3, i4 = np.random.randint(0, grid_num1, para.TotalState)
                startxd, startyd, startxt, startyt = XD[i1], YD[i2], XT[i3], YT[i4]
                s = trimstate(np.array([startxd, startyd, startxt, startyt]), para.scale_level)

        #Reorganize
        Move = []
        for i in range(self.TotalmoveN):
            for j in range(self.movelength):
                move = moves[:, j, i]
                mis = move
                Move.append(mis)
        Moves = np.array(Move)

        return Moves

# if __name__ == "__main__":
#
#     MS = MoveState(para.TotalState, para.movelength, para.TotalmoveN,para.stepsize1,para.stepsize2,para.threshold1,para.threshold2)
#     moves = MS.move()
#     for move in moves:
#         print(move)
#     save = np.array(moves)
#     np.save('./moves',save)

    # x= np.load('./moves.npy')
    # for xx in x:
    #     print(xx)



















