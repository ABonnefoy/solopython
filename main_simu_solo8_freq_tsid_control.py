# coding: utf8
import numpy as np
import argparse
import math
import time as tmp
from time import clock, sleep
from utils.viewerClient import viewerClient
from utils.logger import Logger
from datetime import datetime as datetime

import sys
import os

from simulators.solo_simu_pybullet import Solo_Simu_Pybullet
from simulators.solo_simu import Solo_Simu

from controllers.safety_control import Safety_Control

from controllers.freq_tsid_sinu_control import Freq_TSID_Sinu_Control
from controllers.freq_FF_sinu_control import Freq_FF_Sinu_Control
from controllers.freq_tsid_feet_sinu_control import Freq_TSID_Feet_Sinu_Control
from controllers.freq_ik_tsid_feet_control import Freq_IK_TSID_Feet_Control
from controllers.freq_ik_feet_control import Freq_IK_Feet_Control



def tsid_control(experiment=0):

    ########## VARIABLES INIT ##########

    # Simulation 
    PRINT_N = 500                     
    DISPLAY_N = 25                    
    N_SIMULATION = 10000
    dof = 8
    dt = 0.01   # Controller timestep
    DT = 0.0005 # Device timestep
    ratio = dt/DT

    ### Integrated feedback, without Feedforward Torque, for posture control
    '''if experiment==1:
        controller = Freq_TSID_Sinu_Control(logSize=N_SIMULATION, dt = dt, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0)
        device = Solo_Simu(dt=DT, logSize=N_SIMULATION)
    if experiment==2:
        controller = Freq_TSID_Sinu_Control(logSize=N_SIMULATION, dt = dt, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.5, amp_knee = 0.6)
        device = Solo_Simu(dt=DT, logSize=N_SIMULATION)
    if experiment==3:
        controller = Freq_TSID_Sinu_Control(logSize=N_SIMULATION, dt = dt, freq_hip = 0.5, amp_hip = 0.4, freq_knee = 0.5, amp_knee = 0.6)
        device = Solo_Simu(dt=DT, logSize=N_SIMULATION)'''

    ### Integrated feedback simulations, with Feedforward Torque, for posture control
    '''if experiment==4:
        controller = Freq_FF_Sinu_Control(logSize=N_SIMULATION, dt = dt, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0)
        device = Solo_Simu(dt=DT, logSize=N_SIMULATION)
    if experiment==5:
        controller = Freq_FF_Sinu_Control(logSize=N_SIMULATION, dt = dt, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.5, amp_knee = 0.6)
        device = Solo_Simu(dt=DT, logSize=N_SIMULATION)
    if experiment==6:
        controller = Freq_FF_Sinu_Control(logSize=N_SIMULATION, dt = dt, freq_hip = 0.5, amp_hip = 0.4, freq_knee = 0.5, amp_knee = 0.6)
        device = Solo_Simu(dt=DT, logSize=N_SIMULATION)'''

    ### Foot control
    if experiment==7:  # Integrated Feedback, TSID motion
        controller = Freq_TSID_Feet_Sinu_Control(logSize=N_SIMULATION, dt = dt)
        device = Solo_Simu(dt=DT, logSize=N_SIMULATION)
    if experiment==8:  # IK computed Feedback, TSID motion
        controller = Freq_IK_TSID_Feet_Control(logSize=N_SIMULATION, dt = dt)
        device = Solo_Simu(dt=DT, logSize=N_SIMULATION)
    if experiment==9:  # IK computed Feedback, non TSID motion
        controller = Freq_IK_Feet_Control(logSize=N_SIMULATION, dt = dt)
        device = Solo_Simu(dt=DT, logSize=N_SIMULATION)

    ### Foot control - PyBullet simulator
    '''if experiment==7:  # Integrated Feedback, TSID motion
        controller = Freq_TSID_Feet_Sinu_Control(logSize=N_SIMULATION, dt = dt)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)
    if experiment==8:  # IK computed Feedback, TSID motion
        controller = Freq_IK_TSID_Feet_Control(logSize=N_SIMULATION, dt = dt)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)
    if experiment==9:  # IK computed Feedback, non TSID motion
        controller = Freq_IK_Feet_Control(logSize=N_SIMULATION, dt = dt)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)'''

    ### Foot control - Noise in velocity measurement
    if experiment==10:  # Integrated Feedback, TSID motion
        controller = Freq_TSID_Feet_Sinu_Control(logSize=N_SIMULATION, dt = dt)
        device = Solo_Simu(dt=DT, logSize=N_SIMULATION, noise=True)
    if experiment==11:  # IK computed Feedback, TSID motion
        controller = Freq_IK_TSID_Feet_Control(logSize=N_SIMULATION, dt = dt)
        device = Solo_Simu(dt=DT, logSize=N_SIMULATION, noise=True)
    if experiment==12:  # IK computed Feedback, non TSID motion
        controller = Freq_IK_Feet_Control(logSize=N_SIMULATION, dt = dt)
        device = Solo_Simu(dt=DT, logSize=N_SIMULATION, noise=True)

    nb_motors = device.nb_motors

    # PD Gains
    Kp = 10.0 * np.ones(nb_motors)
    Kd = 0.01 * np.ones(nb_motors)
        
    device.Init(q_init=controller.q_init.copy()) #Initialize device with reference position
    
    controller.Init(q_mes=device.q_mes, v_mes=device.v_mes) # Initialize controller with measured reference position
    
    t_list = []	#list of the time of each simulation step
    i = 0
    t = 0
    
    
    
    #CONTROL LOOP ***************************************************
    while (i < N_SIMULATION):
        
        time_start = tmp.time()      

        if i % ratio == 0:  # Set to controller frequency
            torque_FF, Kp, Kd, q_des, v_des = controller.low_level(device.q_mes, device.v_mes, Kp, Kd, i) # Feedback and Feedforward computing

            device.SetDesiredJointTorque(torque_FF)              # With Feedforward Torque
            #device.SetDesiredJointTorque(np.zeros(nb_motors))   # Without Feedforward Torque
      
            device.SetDesiredJointPDgains(Kp, Kd)
            device.SetDesiredJointPosition(q_des)
            device.SetDesiredJointVelocity(v_des)

        device.SendCommand(WaitEndOfCycle=True) # Low-Level impedance control          
        
        
        # Variables update  
        time_spent = tmp.time() - time_start
        t_list.append(time_spent)
        device.sample(i)
        controller.sample(i)        
        i += 1
        t += DT
        
        
        
    
    #****************************************************************
        
    
    ########## PLOTS ##########          
    plotAll(controller, device, t_list, experiment, Kp[0], Kd[0])
    
    device.saveAll(filename = "../Results/Latest/data/simu_%i_device_data_Kp%f_Kd%f" %(experiment, Kp[0], Kd[0]))
    controller.saveAll(filename = "../Results/Latest/data/simu_%i_tsid_data_Kp%f_Kd%f" %(experiment, Kp[0], Kd[0]))
    
    
    
def plotAll(controller, device, t_list, experiment, Kp, Kd):

    import matplotlib.pyplot as plt
    date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')
    savefile = '../Results/Latest/Images/'
    fontsize = 10
    frame_names = controller.frame_names
    axis = ['x-axis', 'y-axis', 'z-axis']
    
    plt.figure(1) ### Torque Command 
    plt.suptitle('Torques tracking', fontsize = fontsize+2)
    for k in range(2):
        plt.subplot(2,1,k+1)
        plt.title(frame_names[k+3],fontsize = fontsize)
        plt.plot(device.torque_cmd_list[:,k+2], 'b', linestyle = 'dotted', label = 'Command')
        plt.plot(controller.torque_des_list[:,k+2], 'c', linestyle = 'dashdot', label = 'Feedforward')
        axes = plt.gca()
        axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_ylabel('Torque [Nm]',fontsize = fontsize)
    axes.set_xlabel('Time [ms]', fontsize = fontsize)
    axes.legend(loc = 'lower right')
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'simu_%i_TORQUES_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
    plt.show()    
    
    plt.figure(2) ### Joint position - desired and measured
    plt.suptitle('Joints position tracking', fontsize = fontsize+2)
    for k in range(2):
        plt.subplot(2,1,k+1)
        plt.title(frame_names[k+3],fontsize = fontsize)
        plt.plot(device.q_mes_list[:,k+2], '-b', linestyle = 'dotted', label = 'Measured')  
        plt.plot(controller.q_cmd_list[:,k+2], '-c', linestyle = 'dashdot', label = 'Desired')  
        axes = plt.gca()
        axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_ylabel('Position [rad]',fontsize = fontsize)
    axes.set_xlabel('Time [ms]', fontsize = fontsize)
    axes.legend(loc = 'lower right')
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'simu_%i_Q_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
    plt.show()
    
    plt.figure(3) ### Joint velocity - desired and measured
    plt.suptitle('Joints velocity tracking', fontsize = fontsize+2)
    for k in range(2):
        plt.subplot(2,1,k+1)
        plt.title(frame_names[k+3],fontsize = fontsize)
        plt.plot(device.v_mes_list[:,k+2], '-b', linestyle = 'dotted', label = 'Measured')
        plt.plot(controller.v_cmd_list[:,k+2], '-c', linestyle = 'dashdot', label = 'Desired')
        axes = plt.gca()
        axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_ylabel('Velocity [rad/s]',fontsize = fontsize)
    axes.set_xlabel('Time [ms]', fontsize = fontsize)
    axes.legend(loc = 'lower right')
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'simu_%i_V_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
    plt.show()

    if (experiment>=7):
        plt.figure(4) ### Foot position - desired and measured
        plt.suptitle('Foot position tracking', fontsize = fontsize+2)
        for k in range(3):
            plt.subplot(3,1,k+1)
            plt.title(axis[k],fontsize = fontsize)
            plt.plot(device.p_mes_list[:,k], '-b', linestyle = 'dotted', label = 'Measured')
            plt.plot(controller.p_des_list[:,k], '-c', linestyle = 'dashdot', label = 'Desired')
            axes = plt.gca()
            axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
        axes.set_ylabel('Position [m]',fontsize = fontsize)
        axes.set_xlabel('Time [ms]', fontsize = fontsize)
        axes.legend(loc = 'lower right')
        plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
        plt.savefig(savefile + 'simu_%i_FOOT_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
        plt.show()
        
    '''plt.figure(5) ### Computation Time
    plt.suptitle('Computation time', fontsize = fontsize+2)
    plt.plot(t_list, 'c+')
    axes = plt.gca()
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_ylabel('Time[s]',fontsize = fontsize)
    axes.set_xlabel('Iteration', fontsize = fontsize)
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'simu_%i_TIME._Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
    plt.show()'''
    
    

def main():
    parser = argparse.ArgumentParser(description='Example masterboard use in python.')
    parser.add_argument('-exp',
                        '--experiment',
                        type=int,
                        required=True,
                        help='Chosen experiment')

    

    tsid_control(parser.parse_args().experiment)


if __name__ == "__main__":
    main()
