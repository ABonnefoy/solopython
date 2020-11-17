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
        self.encoderOffsets = - np.array([2.0888984203338623, -2.597313642501831, 2.772291660308838, 1.0124378204345703, -2.6765713691711426, 0.7007977962493896, 2.1985981464385986, 2.1340904235839844])
        
        #self.encoderOffsets *= 0.
        self.rotateImuVectors = lambda x: [x[1], x[0], -x[2]]
        self.rotateImuOrientation = lambda q: [-q[1], -q[0], q[2], -q[3]]


