import numpy as np
from utils.abstractRobotHal import RobotHAL

class TestBench(RobotHAL):
    ''' Define the hardware interface to the test bench'''

    def __init__(self, interfaceName="", dt=0.001):
        RobotHAL.__init__(self, interfaceName, dt)

    def InitRobotSpecificParameters(self):
        ''' Definition of the test bench paramters '''
        self.nb_motors = 2
        self.motorToUrdf = [0, 1]
        self.gearRatio = np.array(self.nb_motors * [1., ])  # gearbox ratio
        self.motorKt = np.array(self.nb_motors * [0.025, ])  # Nm/A
        self.motorSign = np.array([+1, -1])
        
        self.maximumCurrent = 3.0  # A
        # To get this offsets, run the calibration with self.encoderOffsets at 0,
        # then manualy move the robot in zero config, and paste the position here (note the negative sign!)
        self.encoderOffsets = - np.array([1.023422122001648, 0.24071289598941803]) # Zero arrows up
        #self.encoderOffsets *= 0.
        self.rotateImuVectors = lambda x: [x[0], x[1], x[2]] # No IMU for the test benches
        self.rotateImuOrientation = lambda q: [q[0], q[1], q[2], q[3]] # No IMU for the test benches
        