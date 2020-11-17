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


from controllers.freq_tsid_feet_sinu_control import Freq_TSID_Feet_Sinu_Control
from controllers.freq_ik_feet_sinu_control import Freq_IK_Feet_Sinu_Control


########## VARIABLES INIT ##########

# Simulation 
PRINT_N = 500                     
DISPLAY_N = 25                    
N_SIMULATION = 10000 
DT = 0.001  
dof = 8

# PD Gains
kp = 3.0
kd = 0.05


def tsid_control(name_interface, experiment=0):
    device = Solo8(name_interface,dt=DT)
    nb_motors = device.nb_motors


    log = Logger(device, logSize = N_SIMULATION)
    

    if experiment==1:
        controller = Freq_TSID_Feet_Sinu_Control(logSize=N_SIMULATION, dt = DT)
    if experiment==2:
        controller = Freq_IK_Feet_Sinu_Control(logSize=N_SIMULATION, dt = DT)


    
    device.Init(calibrateEncoders=True, q_init=controller.q_init)

    qmes = device.q_mes
    vmes = device.v_mes
    vmes_prev = np.zeros(dof)

    controller.Init(qmes, vmes)
   
    Kp = kp * np.ones(nb_motors)
    Kd = kd * np.ones(nb_motors)
    
    t_list = []	#list of the time of each simulation step
    i = 0
    
    #CONTROL LOOP ***************************************************
    while ((not device.hardware.IsTimeout()) and (i < N_SIMULATION)):
        device.UpdateMeasurment()

        
        time_start = tmp.time()      

     
        # Measured state
        qmes = device.q_mes
        vmes = device.v_mes  

        torque_FF, q_des, v_des = controller.low_level(qmes, vmes, i)

        device.SetDesiredJointTorque(torque_FF)        
        device.SetDesiredJointPDgains(Kp, Kd)
        device.SetDesiredJointPosition(q_des)
        device.SetDesiredJointVelocity(v_des)

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
    
    log.saveAll(fileName = "../Results/Latest/data/expe_low_level_%i_logger_data_Kp%f_Kd%f" %(experiment, Kp, Kd))
    controller.saveAll(filename = "../Results/Latest/data/expe_low_level_%i_controller_data_Kp%f_Kd%f" %(experiment, Kp, Kd))
    
    
    
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
    plt.savefig(savefile + 'expe_low_level_%i_TORQUES_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')    
    
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
    plt.savefig(savefile + 'expe_low_level_%i_Q_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
    
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
    plt.savefig(savefile + 'expe_low_level_%i_V_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
    
    plt.figure(4)
    plt.suptitle('Computation time', fontsize = fontsize+2)
    plt.plot(t_list, 'c+')
    axes = plt.gca()
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_ylabel('Time[s]',fontsize = fontsize)
    axes.set_xlabel('Iteration', fontsize = fontsize)
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'expe_low_level_%i_TIME_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
    
    

def main():
    parser = argparse.ArgumentParser(description='Example masterboard use in python.')
    parser.add_argument('-i',
                        '--interface',
                        required=True,
                        help='Name of the interface (use ifconfig in a terminal), for instance "enp1s0"')
    parser.add_argument('-exp',
                        '--experiment',
                        type=int,
                        required=True,
                        help='Chosen experiment (detailed list to come)')
    

    tsid_control(parser.parse_args().interface, parser.parse_args().experiment)


if __name__ == "__main__":
    main()
