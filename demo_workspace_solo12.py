# coding: utf8
import numpy as np
import argparse
from solo12 import Solo12
from pynput import keyboard

from utils.logger import Logger
from utils.qualisysClient import QualisysClient

DT = 0.001

key_pressed = False

# Variables for motion range test of Solo12
# t_switch contains the timestamp associated with the target positions
# q_switch contains the target position for the 3 motors of the front left leg of the robot
# Column i corresponds to i-th target position (contains 3 angular positions) that the robot should reach at time t_switch[i]
# There is an interpolation between targets to have a smooth motion between them
t_switch = 0.3 * np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 33.0, 36.0, 39.0, 42.0])
q_switch = np.array([[0.0, 0.66 * np.pi, 0.0, 0.66 * np.pi, 0.0, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi, 0.0, -0.1 * np.pi, 0.0, 0.22 * np.pi, 0.0, 0.0],
                     [0.8, 0.8, -0.8, -0.8, 0.8, 0.8, 0.0, -0.8, -0.8, -1.4, np.pi*0.5, np.pi*0.5, np.pi*0.5, 0.0],
                     [-1.6, -1.6, 1.6, 1.6, -1.6, -1.6, +1.0, -2.0, -2.0, 2.0, -np.pi, -np.pi*1.4, -np.pi, 0.0]])


def demo_solo12(t):
    """Compute the desired position and velocity of actuators at time t depending on the targets
    defined in t_switch and q_switch

    Args:
        t (float): time elapsed since the start of the demo
    """

    i = 1
    while (i < t_switch.shape[0]) and (t_switch[i] <= t):
        i += 1

    if (i != t_switch.shape[0]):
        return apply_q_v_change(t, i)
    else:
        N = q_switch.shape[1]
        return np.tile(q_switch[:, (N-1):N], (4, 1)) * np.array([[1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1]]).transpose(), np.zeros((12,))


def apply_q_v_change(t, i):
    """Interpolate position and velocity trajectories with polynomials to smoothly go from
    one target to another

    Args:
        t (float): time elapsed since the start of the demo
        i (int): numero of the target that the robot is trying to reach
    """

    # Interpolation of q and v for the front left leg
    ev = t - t_switch[i-1]
    t1 = t_switch[i] - t_switch[i-1]
    A3 = 2 * (q_switch[:, (i-1):i] - q_switch[:, i:(i+1)]) / t1**3
    A2 = (-3/2) * t1 * A3
    q = q_switch[:, (i-1):i] + A2*ev**2 + A3*ev**3
    v = 2 * A2 * ev + 3 * A3 * ev**2

    # Tiling the result of the front left leg for the three other legs
    q = np.tile(q, (4, 1)) * np.array([[1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1]]).transpose()
    v = np.tile(v, (4, 1)) * np.array([[1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1]]).transpose()

    return q, v


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
    qc = QualisysClient(ip="140.93.16.160", body_id=0)  # QualisysClient
    logger = Logger(device, qualisys=qc)  # Logger object
    nb_motors = device.nb_motors

    # Default position after calibration
    q_init = np.array([0.0, 0.8, -1.6, 0, 0.8, -1.6, 0, -0.8, 1.6, 0, -0.8, 1.6])

    # Calibrate encoders
    device.Init(calibrateEncoders=True, q_init=q_init)

    # Wait for Enter input before starting the control loop
    put_on_the_floor(device, q_init)

    # CONTROL LOOP ***************************************************
    t = 0.0
    t_max = t_switch[-1]

    # Parameters of the PD controller
    KP = 2.
    KD = 0.05
    tau_max = 5. * np.ones(12)

    while ((not device.hardware.IsTimeout()) and (t < t_max)):

        device.UpdateMeasurment()  # Retrieve data from IMU and Motion capture

        # Desired position and velocity for this loop and resulting torques
        q_desired, v_desired = demo_solo12(t)
        pos_error = q_desired.ravel() - device.q_mes.ravel()
        vel_error = v_desired.ravel() - device.v_mes.ravel()
        tau = KP * pos_error + KD * vel_error
        tau = np.maximum(np.minimum(tau, tau_max), -tau_max)

        # Set desired torques for the actuators
        device.SetDesiredJointTorque(tau)

        # Call logger
        # logger.sample(device, qualisys=qc)

        # Send command to the robot
        device.SendCommand(WaitEndOfCycle=True)
        if ((device.cpt % 100) == 0):
            device.Print()

        t += DT

    # ****************************************************************

    # Whatever happened we send 0 torques to the motors.
    device.SetDesiredJointTorque([0]*nb_motors)
    device.SendCommand(WaitEndOfCycle=True)

    if device.hardware.IsTimeout():
        print("Masterboard timeout detected.")
        print("Either the masterboard has been shut down or there has been a connection issue with the cable/wifi.")
    device.hardware.Stop()  # Shut down the interface between the computer and the master board

    # Save the logs of the Logger object
    # logger.saveAll()


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
