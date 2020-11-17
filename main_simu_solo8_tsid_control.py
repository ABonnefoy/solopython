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

from solo_simu_pybullet import Solo_Simu_Pybullet
from solo_simu import Solo_Simu

from controllers.safety_control import Safety_Control

from controllers.sinu_control import Sinu_Control
from controllers.FF_sinu_control import FF_Sinu_Control

from controllers.tsid_sinu_control import TSID_Sinu_Control
from controllers.tsid_FF_sinu_control import TSID_FF_Sinu_Control

from controllers.direct_tsid_sinu_control import Direct_TSID_Sinu_Control

from controllers.tsid_FF_feet_control import TSID_FF_Feet_Control






def tsid_control(filter=False, experiment=0):

    ########## VARIABLES INIT ##########

    # Simulation 
    PRINT_N = 500                     
    DISPLAY_N = 25                    
    N_SIMULATION = 10000
    dof = 8
    DT = 0.001 # Simulation timestep

    pybullet = True

    if experiment==1:
        controller = Safety_Control(logSize=N_SIMULATION)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)

    if experiment==2:
        controller = Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0) # Modifier freq et amp
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)

    if experiment==3:
        controller = Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.5, amp_knee = 0.6) # Modifier Kp Kd
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)

    if experiment==4:
        controller = Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)
    if experiment==5:
        controller = Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.5, amp_knee = 0.6)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)
    if experiment==6:
        controller = Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.8, amp_knee = 0.8)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)
        controller = Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.5, amp_hip = 0.4, freq_knee = 0.5, amp_knee = 0.6)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)

    if experiment==7:
        controller = FF_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)
    if experiment==8:
        controller = FF_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.5, amp_knee = 0.6)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)
    if experiment==9:
        controller = FF_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.5, amp_hip = 0.4, freq_knee = 0.5, amp_knee = 0.6)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)

    if experiment==10:
        controller = TSID_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)
    if experiment==11:
        controller = TSID_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.5, amp_knee = 0.6)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)
    if experiment==12:
        controller = TSID_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.5, amp_hip = 0.4, freq_knee = 0.5, amp_knee = 0.6)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)

    if experiment==13:
        controller = TSID_FF_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)
    if experiment==14:
        controller = TSID_FF_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.5, amp_knee = 0.6)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)
    if experiment==15:
        controller = TSID_FF_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.5, amp_hip = 0.4, freq_knee = 0.5, amp_knee = 0.6)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)

    if experiment==16:
        controller = Direct_TSID_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)
    if experiment==17:
        controller = Direct_TSID_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.5, amp_knee = 0.6)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)
    if experiment==18:
        controller = Direct_TSID_Sinu_Control(logSize=N_SIMULATION, freq_hip = 0.5, amp_hip = 0.4, freq_knee = 0.5, amp_knee = 0.6)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)

    if experiment==19:
        controller = TSID_FF_Feet_Control(logSize=N_SIMULATION)
        device = Solo_Simu_Pybullet(dt=DT, logSize=N_SIMULATION)
        ratio = None



    # PD
    Kp = 2.0
    Kd = 0.03

    nb_motors = device.nb_motors
        
    device.Init(q_init=controller.q_init.copy())
    
    qmes = device.q_mes.copy()
    vmes = device.v_mes.copy()
    vmes_prev = np.zeros(dof)   
    
    q_prev = np.zeros(8)
    v_prev = np.zeros(8)
    i_prev = 0
    
    controller.Init(qmes=qmes, vmes=vmes)
    
    t_list = []	#list of the time of each simulation step
    i = 0
    t = 0
    
    
    
    #CONTROL LOOP ***************************************************
    while (i < N_SIMULATION):
        
        time_start = tmp.time()      

     
        # Measured state
        qmes = device.q_mes
        vmes = device.v_mes

        jointTorques = controller.control(qmes, vmes, i, Kp, Kd)

        device.runSimulation(jointTorques)
        
        
        
        # Variables update  
        time_spent = tmp.time() - time_start
        t_list.append(time_spent)
        device.sample(i)
        controller.sample(i)        
        i += 1
        t += DT
        
        
        
    
    #****************************************************************
        
    
    ########## PLOTS ##########          
    plotAll(controller, device, t_list, experiment, Kp, Kd)
    
    device.saveAll(filename = "data/simu_%i_device_data_Kp%f_Kd%f" %(experiment, Kp, Kd))
    controller.saveAll(filename = "data/simu_%i_tsid_data_Kp%f_Kd%f" %(experiment, Kp, Kd))
    
    
    
def plotAll(controller, device, t_list, experiment, Kp, Kd):

    import matplotlib.pyplot as plt
    date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')
    path = os.path.dirname(__file__) 
    savefile = os.path.join(path, 'Images/')
    fontsize = 10
    frame_names = controller.frame_names
    
    plt.figure(1)
    plt.suptitle('Torques tracking', fontsize = fontsize+2)
    for k in range(8):
        plt.subplot(4,2,k+1)
        plt.title(frame_names[k+1],fontsize = fontsize)
        plt.plot(controller.jointTorques_list[:,k], 'b', linestyle = 'dotted', label = 'Command')
        axes = plt.gca()
        axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
        if (k==6): 
            axes.set_ylabel('Torque [Nm]',fontsize = fontsize)
            axes.set_xlabel('Time [ms]', fontsize = fontsize)
    axes.set_xlabel('Time [ms]', fontsize = fontsize)
    axes.legend(loc = 'lower right')
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'simu_%i_TORQUES_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
    plt.show()    
    
    plt.figure(2)
    plt.suptitle('Joints position tracking', fontsize = fontsize+2)
    for k in range(8):
        plt.subplot(4,2,k+1)
        plt.title(frame_names[k+1],fontsize = fontsize)
        plt.plot(device.q_mes_list[:,k], '-b', linestyle = 'dotted', label = 'Measured')  
        plt.plot(controller.q_list[:,k], '-c', linestyle = 'dashdot', label = 'Desired')  
        axes = plt.gca()
        axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
        if (k==6): 
            axes.set_ylabel('Position [rad]',fontsize = fontsize)
            axes.set_xlabel('Time [ms]', fontsize = fontsize)
    axes.set_xlabel('Time [ms]', fontsize = fontsize)
    axes.legend(loc = 'lower right')
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'simu_%i_Q_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
    plt.show()
    
    plt.figure(3)
    plt.suptitle('Joints velocity tracking', fontsize = fontsize+2)
    for k in range(8):
        plt.subplot(4,2,k+1)
        plt.title(frame_names[k+1],fontsize = fontsize)
        plt.plot(device.v_mes_list[:,k], '-b', linestyle = 'dotted', label = 'Measured')
        plt.plot(controller.v_list[:,k], '-c', linestyle = 'dashdot', label = 'Desired')
        axes = plt.gca()
        axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
        if (k==6): 
            axes.set_ylabel('Velocity [rad/s]',fontsize = fontsize)
            axes.set_xlabel('Time [ms]', fontsize = fontsize)
    axes.set_xlabel('Time [ms]', fontsize = fontsize)
    axes.legend(loc = 'lower right')
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'simu_%i_V_Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
    plt.show()
    
    plt.figure(4)
    plt.suptitle('Computation time', fontsize = fontsize+2)
    plt.plot(t_list, 'c+')
    axes = plt.gca()
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_ylabel('Time[s]',fontsize = fontsize)
    axes.set_xlabel('Iteration', fontsize = fontsize)
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'simu_%i_TIME._Kp%f_Kd%f' %(experiment, Kp, Kd) + date_str + '.svg')
    plt.show()
    
    

def main():
    parser = argparse.ArgumentParser(description='Example masterboard use in python.')
    parser.add_argument('-rc', 
                        '--filter',
                        type=bool, 
                        required=False, 
                        help='Choosing to to add a RC filter to remove velocity measurement noise (bool)')
    parser.add_argument('-exp',
                        '--experiment',
                        type=int,
                        required=True,
                        help='Chosen experiment (detailed list to come)')

    

    tsid_control(parser.parse_args().filter, parser.parse_args().experiment)


if __name__ == "__main__":
    main()
