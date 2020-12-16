#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 17:26:17 2020

@author: abonnefoy
"""


import pinocchio as pin
import tsid

import numpy as np
from numpy.linalg import norm as norm
from datetime import datetime as datetime

import os

from controllers.safety_control import Safety_Control


class Freq_TSID_Sinu_Control:
    
    def __init__(self, logSize = None, dt = 0.01, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0):
        
        self.DT = dt
        self.PRINT_N = 500                     
        self.DISPLAY_N = 25      
        
        # Robot model 
        self.dof = 8

        # Desired motion
        self.amp_hip = amp_hip
        self.amp_knee = amp_knee
        self.freq_hip = freq_hip
        self.freq_knee = freq_knee

        # Posture Task         
        self.w_posture = 1e3
        self.kp_posture = 10.0
        self.level_posture = 1
        self.q_disp = np.zeros(self.dof)
           
        # Safety 
        self.error = False
        self.joint_error_list = []
        self.time_error = False
        self.safety_controller = Safety_Control()
        self.tau_max = 3.0
        self.security = 0.05    
        
        
        ########## ROBOT MODEL CREATION ##########
    
        # Definition of the path for the urdf and srdf files of the robot
        path = os.path.dirname(__file__)
        path = os.path.join(path, '../.')
        urdf = os.path.join(path, 'solo_description/robots/solo.urdf')
        vector = pin.StdVec_StdString()
        vector.extend(item for item in path)        
        srdf = os.path.join(path, 'solo_description/srdf/solo.srdf')
                  
        # Creation of the robot models
        self.robot = tsid.RobotWrapper(urdf, vector, False) # with tsid
        self.model = self.robot.model()
        self.data = self.robot.data()
        robot_display = pin.RobotWrapper.BuildFromURDF(urdf, [path, ]) #with pinocchio
        self.model_display = robot_display.model
        self.data_display = robot_display.data
        self.frames = self.model_display.names
        self.frame_names = [ name for i,name in enumerate(self.frames)] 
        
        # Reference configuration
        pin.loadReferenceConfigurations(self.model_display, srdf)
        self.q_init = self.model_display.referenceConfigurations["standing"]
                
        if logSize is not None:
            self.tau_list = np.zeros((logSize,self.dof))
            self.torque_des_list = np.zeros((logSize,self.dof))
            self.q_cmd_list = np.zeros((logSize,self.dof))
            self.v_cmd_list = np.zeros((logSize,self.dof))
        
        
    def Init(self, q_mes, v_mes):
        # Initial configuration
        self.q_cmd = q_mes.copy()
        self.v_cmd = v_mes.copy()
        self.dv_cmd = np.zeros(self.dof)
        self.torque_des = np.zeros(self.dof)
        self.tau = np.zeros(self.dof)
	
        self.safety_controller.Init(q_mes, v_mes)

	# Dynamics Problem initialization
        self.t = 0.0 # time
        self.invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", self.robot, False)
        self.invdyn.computeProblemData(self.t, self.q_cmd, self.v_cmd)
        
        # Task definition
        self.postureTask = tsid.TaskJointPosture("task-posture", self.robot)
        self.postureTask.setKp(self.kp_posture * np.ones(self.dof)) 
        self.postureTask.setKd(2.0 * np.sqrt(self.kp_posture) * np.ones(self.dof))
        self.invdyn.addMotionTask(self.postureTask, self.w_posture, self.level_posture, 0.0)
        q_ref = self.q_cmd.copy() 
        self.trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref) # Goal = static
        self.samplePosture = self.trajPosture.computeNext()

        self.offset = np.reshape(self.q_init, 8)
        self.amp                  = np.zeros(8)
        self.amp[2]               = self.amp_hip
        self.amp[3]               = self.amp_knee
        self.two_pi_f             = np.zeros(8)
        self.two_pi_f[2]          = 2*np.pi*self.freq_hip
        self.two_pi_f[3]          = 2*np.pi*self.freq_knee
        self.two_pi_f_amp         = np.multiply(self.two_pi_f,self.amp)
        self.two_pi_f_squared_amp = np.multiply(self.two_pi_f, self.two_pi_f_amp)

        # Solver
        self.solver = tsid.SolverHQuadProgFast("qp solver")
        self.solver.resize(self.invdyn.nVar, self.invdyn.nEq, self.invdyn.nIn)
        HQPData = self.invdyn.computeProblemData(self.t, self.q_cmd, self.v_cmd)
        HQPData.print_all()        
        self.sol = self.solver.solve(HQPData)
        if(self.sol.status!=0):
            print ("QP problem could not be solved! Error code:", self.sol.status)
            self.error = True



##### LOW LEVEL CONTROL

    def low_level(self, v_mes, q_mes, Kp, Kd, i):


        for index in range(len(q_mes)):
            if self.error or (q_mes[index]<-3.14) or (q_mes[index]>3.14) or (v_mes[index]<-30) or (v_mes[index]>30): 
                self.error = True
                self.torque_des = -self.security * v_mes
                self.t += self.DT
                return(self.torque_des, np.zeros(self.dof), np.zeros(self.dof), q_mes, v_mes)
     
        '''self.v_cmd = v_mes.copy()
        self.q_cmd = q_mes.copy()  '''
    

        # TSID computation
        self.samplePosture.pos(self.offset + np.multiply(self.amp, np.sin(self.two_pi_f*self.t)))
        self.samplePosture.vel(np.multiply(self.two_pi_f_amp, np.cos(self.two_pi_f*self.t)))
        self.samplePosture.acc(np.multiply(self.two_pi_f_squared_amp, -np.sin(self.two_pi_f*self.t)))
        self.postureTask.setReference(self.samplePosture)        
        
        HQPData = self.invdyn.computeProblemData(self.t, self.q_cmd, self.v_cmd)     
        self.sol = self.solver.solve(HQPData)
        if(self.sol.status!=0):
            print ("QP problem could not be solved! Error code:", self.sol.status)
            self.error = True

        pin.framesForwardKinematics(self.model, self.data, q_mes)

        self.dv_cmd = self.invdyn.getAccelerations(self.sol)
        self.v_cmd += self.dv_cmd * self.DT
        self.q_cmd = pin.integrate(self.model, self.q_cmd, self.v_cmd*self.DT)

        self.torque_des = self.invdyn.getActuatorForces(self.sol)
            
        self.t += self.DT
            
        return(self.torque_des, Kp, Kd, self.q_cmd, self.v_cmd)
    
    def sample(self, i):
        self.torque_des_list[i,:] = self.torque_des[:]
        self.q_cmd_list[i,:] = self.q_cmd[:]
        self.v_cmd_list[i,:] = self.v_cmd[:]
        
    def saveAll(self, filename = "data"):
        date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')
        np.savez(filename + date_str + ".npz",
                 q=self.q_cmd_list, 
                 v=self.v_cmd_list,
                 torque_des=self.torque_des_list)
   

