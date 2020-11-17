# coding: utf8
import numpy as np
import argparse
import math
from time import clock, sleep
from utils.viewerClient import viewerClient
from solo8 import Solo8

def example_script(name_interface):
    viewer = viewerClient()
    device = Solo8(name_interface,dt=0.001)
    nb_motors = device.nb_motors

    q_viewer = np.array((7 + nb_motors) * [0.,])

    device.Init(calibrateEncoders=False)
    #CONTROL LOOP ***************************************************
    while ((not device.hardware.IsTimeout()) and (clock() < 200)):
        device.UpdateMeasurment()
        device.SetDesiredJointTorque([0]*nb_motors)
        device.SendCommand(WaitEndOfCycle=True)
        if ((device.cpt % 100) == 0):
            device.Print()

        q_viewer[3:7] = device.baseOrientation  # IMU Attitude
        q_viewer[7:] = device.q_mes  # Encoders
        viewer.display(q_viewer)
    #****************************************************************
    
    # Whatever happened we send 0 torques to the motors.
    device.SetDesiredJointTorque([0]*nb_motors)
    device.SendCommand(WaitEndOfCycle=True)

    if device.hardware.IsTimeout():
        print("Masterboard timeout detected.")
        print("Either the masterboard has been shut down or there has been a connection issue with the cable/wifi.")
    device.hardware.Stop()  # Shut down the interface between the computer and the master board

def main():
    parser = argparse.ArgumentParser(description='Example masterboard use in python.')
    parser.add_argument('-i',
                        '--interface',
                        required=True,
                        help='Name of the interface (use ifconfig in a terminal), for instance "enp1s0"')

    example_script(parser.parse_args().interface)


if __name__ == "__main__":
    main()
