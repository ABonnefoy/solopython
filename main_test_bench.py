# coding: utf8
import numpy as np
import argparse
import math
from time import clock, sleep
from test_bench import TestBench

def example_script(name_interface):
    device = TestBench(name_interface,dt=0.001)
    nb_motors = device.nb_motors

    kp = 0.125 # Proportional gain
    kd = 0.0025 # Derivative gain

    freq = 0.5 # Frequency of the sine wave
    amplitude = math.pi # Amplitude of the sine wave

    device.Init(calibrateEncoders=True)
    device.t = 0 # reset time for the sinus to begin at 0
    #CONTROL LOOP ***************************************************
    while ((not device.hardware.IsTimeout()) and (clock() < 200)):
        device.UpdateMeasurment()

        torques = [0]*nb_motors
        for i in range(device.nb_motors):
            q_ref = amplitude * math.sin(2.0 * math.pi * freq * device.t)
            v_ref = 2.0 * math.pi * freq * amplitude * math.cos(2.0 * math.pi * freq * device.t)

            q_err = q_ref - device.q_mes[i] # Position error
            v_err = v_ref - device.v_mes[i] # Velocity error

            torques[i] = kp * q_err + kd * v_err

        device.SetDesiredJointTorque(torques)
        device.SendCommand(WaitEndOfCycle=True)
        if ((device.cpt % 100) == 0):
            device.Print()
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
