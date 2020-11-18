import numpy as np
from utils.abstractRobotHal import RobotHAL

class Solo8(RobotHAL):
    ''' Define the hardware interface to solo8'''

    def __init__(self, interfaceName="", dt=0.001):
        RobotHAL.__init__(self, interfaceName, dt)

    def InitRobotSpecificParameters(self):
        ''' Definition of the Solo8 paramters '''
        self.nb_motors = 8
        self.motorToUrdf = [0, 1, 3, 2, 5, 4, 6, 7]
        self.gearRatio = np.array(self.nb_motors * [9., ])  # gearbox ratio
        self.motorKt = np.array(self.nb_motors * [0.025, ])  # Nm/A
        self.motorSign = np.array([-1, -1, +1, +1, -1, -1, +1, +1])
        self.maximumCurrent = 6.0  # A
        # To get this offsets, run the calibration with self.encoderOffsets at 0,
        # then manualy move the robot in zero config, and paste the position here (note the negative sign!)
        self.encoderOffsets = - np.array([1.764561414718628, -1.9062448740005493, 2.6938323974609375, 1.0126081705093384, -2.5847601890563965, 1.041310429573059, 1.9418796300888062, 2.2229063510894775])
        
        #self.encoderOffsets *= 0.
        self.rotateImuVectors = lambda x: [x[1], x[0], -x[2]]
        self.rotateImuOrientation = lambda q: [-q[1], -q[0], q[2], -q[3]]


