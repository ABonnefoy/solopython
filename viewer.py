# coding: utf8
import argparse
import math
from time import clock, sleep
from masterboard_utils import *
from solo8 import Solo8
from utils.viewerClient import viewerClient
from utils.qualisysClient import QualisysClient
from utils.logger import Logger
import numpy as np
from IPython import embed
import time
experimentTime = 300*1000
logSize = experimentTime
q_viewer = np.array([0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.])
name_interface = "enp4s2f1"
viewer = viewerClient()
qualisys = QualisysClient("140.93.16.160",7)
device = Solo8(name_interface, dt=0.001)
logger = Logger(device,qualisys,logSize)

Kp = 0.3
Kd = 0.0
q_ref = np.array([1,-1,1,-1,-1,1,-1,1])

#From now one, we should not allocate memory !!
device.init(calibrateEncoders=False)

# CONTROL LOOP ***************************************************
while ((not device.hardware.IsTimeout()) and  device.cpt < experimentTime):

    device.UpdateMeasurment()                             # get the latest measurment
    device.SetDesiredJointTorque(Kp*(q_ref-device.q_mes)) # Compute the control
    device.SendCommand(WaitEndOfCycle=True)               # send command and wait for next cycle
    logger.sample(device,qualisys)                        # Log all relevent data

    if ((device.cpt % 50) == 0):
        device.Print()
        quat = device.baseOrientation
        mocapPos = qualisys.getPosition()
        mocapQuat = qualisys.getOrientationQuat()
        #Display the ff with the mocap
        """if not math.isnan(mocapPos[0]):
            q_viewer[0:3] = mocapPos #3d position form mocap
            q_viewer[3:7] = mocapQuat  # IMU Attitude"""

        #display the ff with the Estimation    
        q_viewer[3:7] = quat  # IMU Attitude

        q_viewer[7:] = device.q_mes  # Encoders
        viewer.display(q_viewer)
        print (mocapPos)
# ****************************************************************

#let the user interact with the logger data
embed()
logger.saveAll("data.npz")

# Shut down the interface between the computer and the master board
device.hardware.Stop()