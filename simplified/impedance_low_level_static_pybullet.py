# coding: utf8
import numpy as np
import argparse
import math
import time as tmp
from time import clock, sleep
from utils.viewerClient import viewerClient
from utils.logger import Logger
from datetime import datetime as datetime

import pybullet as pyb
import pybullet_data

import pinocchio as pin
import tsid

import sys
import os


def tsid_control():

    ########## VARIABLES INIT ##########

    # Simulation 
    PRINT_N = 500                     
    DISPLAY_N = 25                    
    N_SIMULATION = 100000
    dof = 8
    dt = 0.01 # Controller
    DT = 0.0002 # Simulator
    ratio = dt/DT
    error = False
    maximumTorque = 3.0
    last = 0.0
    wait = True
    security = -0.05

    # Robot creation
    path = "/opt/openrobots/share/example-robot-data/robots/solo_description"
    urdf = path + "/robots/solo.urdf"
    vector = pin.StdVec_StdString()
    srdf = path + "/srdf/solo.srdf"
   
    robot = tsid.RobotWrapper(urdf, vector, False) # with tsid
    model = robot.model()
    data = robot.data()
    robot_display = pin.RobotWrapper.BuildFromURDF(urdf, [path, ]) #with pinocchio
    model_display = robot_display.model
    data_display = robot_display.data
    frames = model_display.names
    frame_names = [ name for i,name in enumerate(frames)] 

    pin.loadReferenceConfigurations(model_display, srdf)
    q_init = model_display.referenceConfigurations["standing"]

    ########## VIEWER CREATION ##########  

    physicsClient = pyb.connect(pyb.GUI) # or p.DIRECT for non-graphical version
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = pyb.loadURDF("plane.urdf")
    pyb.setGravity(0,0,-9.81)
    robotStartPos = [0,0,0.5]
    robotStartOrientation = pyb.getQuaternionFromEuler([0,0,0])
    robotId = pyb.loadURDF(urdf,robotStartPos, robotStartOrientation)
    revoluteJointIndices = [0,1, 3,4, 6,7, 9,10]
    nb_motors = len(revoluteJointIndices)
    pyb.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=(0.0*(180/3.1415)+45), cameraPitch=-39.9,
                                   cameraTargetPosition=robotStartPos)
    pyb.setJointMotorControlArray(robotId, jointIndices = revoluteJointIndices, controlMode = pyb.VELOCITY_CONTROL,targetVelocities = [0.0 for m in revoluteJointIndices], forces = [0.0 for m in revoluteJointIndices]) 

    initial_joint_positions = [ joint for i,joint in enumerate(q_init)]
    for i in range (len(initial_joint_positions)):
        pyb.resetJointState(robotId, revoluteJointIndices[i], initial_joint_positions[i])
    jointTorques = [0.0 for m in revoluteJointIndices]
        
    pyb.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)
        
    pyb.createConstraint(robotId, -1, -1, -1, pyb.JOINT_FIXED, [0, 0, 0], [0, 0, 0], robotStartPos) # Fixed base
    jointStates = pyb.getJointStates(robotId, revoluteJointIndices)
            
    q_mes = np.array([jointStates[i_joint][0] for i_joint in range(len(jointStates))])
    v_mes = np.array([jointStates[i_joint][1] for i_joint in range(len(jointStates))])
        
    # Set time step for the simulation
    pyb.setTimeStep(DT)


    ########## CONTROLLER INITIALIZATION ##########

    # PD
    Kp = 1.0 * np.ones(nb_motors)
    Kd = 0.005 * np.ones(nb_motors)

    
    # Initial controller configuration
    q_des = q_init.copy()
    v_des = np.zeros(dof)
    dv_des = np.zeros(dof)
    jointTorques = np.zeros(dof)
    torque_FF = np.zeros(dof)

    
    t_list = []	#list of the time of each simulation step
    i = 0
    t = 0
    
    
    
    #CONTROL LOOP ***************************************************
    while (i < N_SIMULATION):
        
        time_start = tmp.time()     

        if i % ratio == 0:

            for index in range(len(q_mes)):
                if error or (q_mes[index]<-3.14) or (q_mes[index]>3.14) or (v_mes[index]<-30) or (v_mes[index]>30): 
                    error = True
                    torque_FF = -security * v_mes
                    Kp = 0.0
                    Kd = 0.0

            if error == False: 
                pin.framesForwardKinematics(model, data, q_mes)
                torque_FF = pin.rnea(model, data, q_des, v_des, dv_des)
            
    
        torque = np.array(Kp * (q_des - q_mes) + Kd * (v_des - v_mes) + torque_FF)
        jointTorques = np.clip(torque, -maximumTorque * np.ones(8), maximumTorque * np.ones(8))

        pyb.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)
        pyb.stepSimulation()
        jointStates = pyb.getJointStates(robotId, revoluteJointIndices)
        q_mes = np.array([jointStates[i_joint][0] for i_joint in range(len(jointStates))])
        v_mes = np.array([jointStates[i_joint][1] for i_joint in range(len(jointStates))]) 


        while(wait):
            if((clock() - last) >= DT):
                last = clock()
                wait = False             
        
        
        # Variables update  
        time_spent = tmp.time() - time_start
        t_list.append(time_spent)
        i += 1
        t += DT 
        wait = True
        
    
    #****************************************************************
 

def main():
    tsid_control()


if __name__ == "__main__":
    main()
