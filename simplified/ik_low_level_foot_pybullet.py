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

    # SE3 Task         
    w_se3 = 1.0
    kp_se3 = 30.0
    level_se3 = 1
    foot = 'FR_FOOT'
    foot_index = model.getFrameId(foot)

    # PD
    Kp = 6.0 * np.ones(nb_motors)
    Kd = 0.05 * np.ones(nb_motors)

    
    # Initial controller configuration
    q_des = q_init.copy()
    v_des = np.zeros(dof)
    dv_des = np.zeros(dof)
    jointTorques = np.zeros(dof)
    torque_FF = np.zeros(dof)

    # Dynamics Problem initialization
    t = 0.0 # time
    invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
    invdyn.computeProblemData(t, q_des, v_des)
        
    se3Task = tsid.TaskSE3Equality("task-se3", robot, foot)
    se3Task.setKp(kp_se3 * np.ones(6))
    se3Task.setKd(2.0 * np.sqrt(kp_se3) * np.ones(6))
    mask = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    se3Task.setMask(mask)
    se3Task.useLocalFrame(False)
    invdyn.addMotionTask(se3Task, w_se3, level_se3, 0.0)
    se3_ref = robot.framePosition(invdyn.data(), foot_index)
    se3_target = se3_ref.copy()
    trajSE3 = tsid.TrajectorySE3Constant("traj_foot", se3_target)
    sampleSE3 = trajSE3.computeNext()


    offset               = np.zeros(6)
    offset[:3]           = se3_ref.translation
    amp                  = np.zeros(6)
    amp[2]               = 0.1
    two_pi_f             = np.zeros(6)
    two_pi_f[2]          = 2*np.pi*0.5
    two_pi_f_amp         = np.multiply(two_pi_f,amp)
    two_pi_f_squared_amp = np.multiply(two_pi_f, two_pi_f_amp)

    # Solver
    solver = tsid.SolverHQuadProgFast("qp solver")
    solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)
    HQPData = invdyn.computeProblemData(t, q_mes, v_mes)
    HQPData.print_all()        
    sol = solver.solve(HQPData)
    if(sol.status!=0):
        print ("QP problem could not be solved! Error code:", sol.status)
        error = True


    
    t_list = []	#list of the time of each simulation step
    i = 0
    
    
    
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

                q_prev = q_des.copy()
                v_prev = v_des.copy()

                # TSID computation        

                sampleSE3.pos(offset + np.multiply(amp, np.sin(two_pi_f*t)))
                sampleSE3.vel(np.multiply(two_pi_f_amp, np.cos(two_pi_f*t)))
                sampleSE3.acc(np.multiply(two_pi_f_squared_amp, -np.sin(two_pi_f*t)))
                se3Task.setReference(sampleSE3) 

                if i == 0:
                    y = sampleSE3.pos()
                    dy = sampleSE3.vel()
        
                HQPData = invdyn.computeProblemData(t, q_mes, v_mes)     
                sol = solver.solve(HQPData)
                if(sol.status!=0):
                    print ("QP problem could not be solved! Error code:", sol.status)
                    error = True

                pin.framesForwardKinematics(model, data, q_mes)
                pin.crba(model, data, q_mes)
                J_foot = pin.computeFrameJacobian(model, data, q_prev, foot_index)
                M = data.M
                y_prev = y.copy()
                dy_prev = dy.copy()

                ddy = sampleSE3.acc()
                #dv_des = invdyn.getAccelerations(sol)
                ddy_cmd = kp_se3 * (y - y_prev) + 2.0 * np.sqrt(kp_se3) * (dy - dy_prev) + ddy
                dv_des = np.linalg.pinv(J_foot) @ ddy_cmd

                dy = sampleSE3.vel()
                v_des = np.linalg.pinv(J_foot) @ dy

                y = sampleSE3.pos()
                q_des = q_prev + np.linalg.pinv(J_foot) @ (y - y_prev)

                #torque_FF = invdyn.getActuatorForces(sol)
                torque_FF = M @ dv_des + pin.rnea(model, data, q_des, v_des, dv_des)
            
    
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
