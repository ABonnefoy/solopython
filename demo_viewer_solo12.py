# coding: utf8
import numpy as np
import argparse
from solo12 import Solo12
from pynput import keyboard

from utils.qualisysClient import QualisysClient
from utils.viewerClient import viewerClient, NonBlockingViewerFromRobot
DT = 0.001

key_pressed = False


def on_press(key):
    """Wait for a specific key press on the keyboard

    Args:
        key (keyboard.Key): the key we want to wait for
    """
    global key_pressed
    try:
        if key == keyboard.Key.enter:
            key_pressed = True
            # Stop listener
            return False
    except AttributeError:
        print('Unknown key {0} pressed'.format(key))


def put_on_the_floor(device, q_init):
    """Make the robot go to the default initial position and wait for the user
    to press the Enter key to start the main control loop

    Args:
        device (robot wrapper): a wrapper to communicate with the robot
        q_init (array): the default position of the robot
    """
    global key_pressed
    key_pressed = False
    Kp_pos = 3.
    Kd_pos = 0.01
    imax = 3.0
    pos = np.zeros(device.nb_motors)
    for motor in range(device.nb_motors):
        pos[motor] = q_init[device.motorToUrdf[motor]] * device.gearRatioSigned[motor]
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("Put the robot on the floor and press Enter")
    while not key_pressed:
        device.UpdateMeasurment()
        for motor in range(device.nb_motors):
            ref = Kp_pos*(pos[motor] - device.hardware.GetMotor(motor).GetPosition() - Kd_pos*device.hardware.GetMotor(motor).GetVelocity())
            ref = min(imax, max(-imax, ref))
            device.hardware.GetMotor(motor).SetCurrentReference(ref)
        device.SendCommand(WaitEndOfCycle=True)

    print("Start the motion.")


def mcapi_playback(name_interface):
    """Main function that calibrates the robot, get it into a default waiting position then launch
    the main control loop once the user has pressed the Enter key

    Args:
        name_interface (string): name of the interface that is used to communicate with the robot
    """
    device = Solo12(name_interface, dt=DT)
    nb_motors = device.nb_motors

    q_viewer = np.array((7 + nb_motors) * [0.,])

    v = viewerClient()
    v.display(q_viewer)

    # Default position after calibration
    q_init = np.zeros(12)

    # Calibrate encoders
    device.Init(calibrateEncoders=True, q_init=q_init)

    # Wait for Enter input before starting the control loop
    put_on_the_floor(device, q_init)

    # CONTROL LOOP ***************************************************
    t = 0.0
    t_max = 300.0

    while ((not device.hardware.IsTimeout()) and (t < t_max)):

        device.UpdateMeasurment()  # Retrieve data from IMU and Motion capture

        # Zero desired torques
        tau = np.zeros(12)

        # Set desired torques for the actuators
        device.SetDesiredJointTorque(tau)

        # Send command to the robot
        device.SendCommand(WaitEndOfCycle=True)
        if ((device.cpt % 100) == 0):
            device.Print()

        # Gepetto GUI
        q_viewer[3:7] = device.baseOrientation  # IMU Attitude
        q_viewer[7:] = device.q_mes  # Encoders
        q_viewer[2] = 0.5
        v.display(q_viewer)

        t += DT

    # ****************************************************************

    # Whatever happened we send 0 torques to the motors.
    device.SetDesiredJointTorque([0]*nb_motors)
    device.SendCommand(WaitEndOfCycle=True)

    if device.hardware.IsTimeout():
        print("Masterboard timeout detected.")
        print("Either the masterboard has been shut down or there has been a connection issue with the cable/wifi.")
    device.hardware.Stop()  # Shut down the interface between the computer and the master board


def main():
    """Main function
    """

    parser = argparse.ArgumentParser(description='Playback trajectory to show the extent of solo12 workspace.')
    parser.add_argument('-i',
                        '--interface',
                        required=True,
                        help='Name of the interface (use ifconfig in a terminal), for instance "enp1s0"')

    mcapi_playback(parser.parse_args().interface)


if __name__ == "__main__":
    main()
