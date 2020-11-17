# coding: utf8
import numpy as np
import argparse
import math
import time as tmp
from time import clock, sleep
from utils.logger import Logger
from solo8 import Solo8
from utils.qualisysClient import QualisysClient
from datetime import datetime as datetime


import os
import matplotlib.pyplot as plt

from controllers.safety_control import Safety_Control

from controllers.sinu_control import Sinu_Control
from controllers.FF_sinu_control import FF_Sinu_Control

from controllers.tsid_sinu_control import TSID_Sinu_Control
from controllers.tsid_FF_sinu_control import TSID_FF_Sinu_Control

from controllers.direct_tsid_sinu_control import Direct_TSID_Sinu_Control

from controllers.freq_tsid_static_control import Freq_TSID_Static_Control

########## VARIABLES INIT ##########

# Simulation 
PRINT_N = 500                     
DISPLAY_N = 25                    
N_SIMULATION = 10000 
DT = 0.001  
dof = 8

# Order 1 Low-pass filter parameters 
R = 0.1
C = 0.01
RC = R * C # Si RC=0: pas d'incidence
alpha = DT / (RC + DT)

# PD Gains
Kp = 3.0
Kd = 0.05


def tsid_control(name_interface, filter=False, mocap=False, experiment=0):
    device = Solo8(name_interface,dt=DT)
    nb_motors = device.nb_motors

    
    if mocap:
        qualisys = QualisysClient()
        log = Logger(device, qualisys = qualisys, logSize = N_SIMULATION)
    else:    
        log = Logger(device, logSize = N_SIMULATION)
    
    
    if experiment==1:
        controller = Safety_Control(logSize=N_SIMULATION)

    if experiment==2:
        controller = Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.8, amp_knee = 0.6) # Change freq and amp

    if experiment==3:
        controller = Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.8, amp_knee = 0.6) # CHange Kp Kd; freq and amp optimal value

    if experiment==4:
        controller = Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0)
    if experiment==5:
        controller = Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.8, amp_knee = 0.6)
    if experiment==6:
        controller = Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.8, amp_hip = 0.4, freq_knee = 0.8, amp_knee = 0.6)

    if experiment==7:
        controller = FF_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0)
    if experiment==8:
        controller = FF_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.8, amp_knee = 0.6)
    if experiment==9:
        controller = FF_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.8, amp_hip = 0.4, freq_knee = 0.8, amp_knee = 0.6)

    if experiment==10:
        controller = TSID_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0)
    if experiment==11:
        controller = TSID_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.8, amp_knee = 0.6)
    if experiment==12:
        controller = TSID_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.8, amp_hip = 0.4, freq_knee = 0.8, amp_knee = 0.6)

    if experiment==13:
        controller = TSID_FF_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0)
    if experiment==14:
        controller = TSID_FF_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.8, amp_knee = 0.6)
    if experiment==15:
        controller = TSID_FF_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.8, amp_hip = 0.4, freq_knee = 0.8, amp_knee = 0.6)

    if experiment==16:
        controller = Direct_TSID_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0)
    if experiment==17:
        controller = Direct_TSID_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.8, amp_knee = 0.6)
    if experiment==18:
        controller = Direct_TSID_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.8, amp_hip = 0.4, freq_knee = 0.8, amp_knee = 0.6)

    
    device.Init(calibrateEncoders=True, q_init=controller.q_init)

    qmes = device.q_mes
    vmes = device.v_mes
    vmes_prev = np.zeros(dof)

    controller.Init(qmes, vmes)
    
    t_list = []	#list of the time of each simulation step
    i = 0
    
    #CONTROL LOOP ***************************************************
    while ((not device.hardware.IsTimeout()) and (i < N_SIMULATION)):
        device.UpdateMeasurment()

        
        time_start = tmp.time()      

     
        # Measured state
        qmes = device.q_mes
        vmes = device.v_mes  
            
        # Low-Pass Filtering
        if filter:
            vmes[:] = alpha * vmes[:] + (1-alpha) * vmes_prev[:] # 1st Order
                
        jointTorques = controller.control(qmes, vmes, i, Kp, Kd)
        
        device.SetDesiredJointTorque(jointTorques)
        device.SendCommand(WaitEndOfCycle=True)        
        
        
        # Variables update
        vmes_prev = vmes.copy()  # Previous velocity for filtering    
        time_spent = tmp.time() - time_start
        t_list.append(time_spent)
        log.sample(device)
        controller.sample(i)
        i += 1

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
    
    
    ########## PLOTS ##########          
    plotAll(controller, log, t_list, experiment, Kp, Kd)
    
    log.saveAll(fileName = "../Results/Latest/data/expe_%i_logger_data_Kp%f_Kd%f" %(experiment, Kp, Kd))
    controller.saveAll(filename = "../Results/Latest/data/expe_%i_controller_data_Kp%f_Kd%f" %(experiment, Kp, Kd))
    
    
    
def plotAll(controller, log, t_list, experiment, Kp, Kd):
    date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')
    savefile = '../Results/Latest/Images/'
    fontsize = 10
    frame_names = controller.frame_names
    
    plt.figure(1)
    plt.suptitle('Torques tracking', fontsize = fontsize+2)
    for k in range(8):
        plt.subplot(4,2,k+1)
        plt.title(frame_names[k+1],fontsize = fontsize)
        plt.plot(controller.jointTorques_list[:,k], 'b', linestyle = 'dotted', label = 'Command')
        plt.plot(log.torquesFromCurrentMeasurment[:,k], 'c', linestyle = 'dashdot', label = 'Measured')
        axes = plt.gca()
        axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
        if (k==6): 
            axes.set_ylabel('Torque [Nm]',fontsize = fontsize)
            axes.set_xlabel('Time [ms]', fontsize = fontsize)
    axes.set_xlabel('Time [ms]', fontsize = fontsize)
    axes.legend(loc = 'lower right')
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'expe_%i_TORQUES_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')    
    
    plt.figure(2)
    plt.suptitle('Joints position tracking', fontsize = fontsize+2)
    for k in range(8):
        plt.subplot(4,2,k+1)
        plt.title(frame_names[k+1],fontsize = fontsize)
        plt.plot(log.q_mes[:,k], '-b', linestyle = 'dotted', label = 'Measured')  
        plt.plot(controller.q_list[:,k], '-c', linestyle = 'dashdot', label = 'Desired')  
        axes = plt.gca()
        axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
        if (k==6): 
            axes.set_ylabel('Position [rad]',fontsize = fontsize)
            axes.set_xlabel('Time [ms]', fontsize = fontsize)
    axes.set_xlabel('Time [ms]', fontsize = fontsize)
    axes.legend(loc = 'lower right')
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'expe_%i_Q_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
    
    plt.figure(3)
    plt.suptitle('Joints velocity tracking', fontsize = fontsize+2)
    for k in range(8):
        plt.subplot(4,2,k+1)
        plt.title(frame_names[k+1],fontsize = fontsize)
        plt.plot(log.v_mes[:,k], '-b', linestyle = 'dotted', label = 'Measured')
        plt.plot(controller.v_list[:,k], '-c', linestyle = 'dashdot', label = 'Desired')
        axes = plt.gca()
        axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
        if (k==6): 
            axes.set_ylabel('Velocity [rad/s]',fontsize = fontsize)
            axes.set_xlabel('Time [ms]', fontsize = fontsize)
    axes.set_xlabel('Time [ms]', fontsize = fontsize)
    #plt.show()
    axes.legend(loc = 'lower right')
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'expe_%i_V_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
    
    plt.figure(4)
    plt.suptitle('Computation time', fontsize = fontsize+2)
    plt.plot(t_list, 'c+')
    axes = plt.gca()
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_ylabel('Time[s]',fontsize = fontsize)
    axes.set_xlabel('Iteration', fontsize = fontsize)
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'expe_%i_TIME_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
    
    

def main():
    parser = argparse.ArgumentParser(description='Example masterboard use in python.')
    parser.add_argument('-i',
                        '--interface',
                        required=True,
                        help='Name of the interface (use ifconfig in a terminal), for instance "enp1s0"')
    parser.add_argument('-rc', 
                        '--filter',
                        type=bool, 
                        required=False, 
                        help='Choosing to to add a RC filter to remove velocity measurement noise (bool)')
    parser.add_argument('-mc',
                        '--mocap',
                        type=bool,
                        required=False,
                        help='Choosing to use Motion Capture (bool)')
    parser.add_argument('-exp',
                        '--experiment',
                        type=int,
                        required=True,
                        help='Chosen experiment (detailed list to come)')
    

    tsid_control(parser.parse_args().interface, parser.parse_args().filter, parser.parse_args().mocap, parser.parse_args().experiment)


if __name__ == "__main__":
    main()
