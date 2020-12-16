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
import sys



class Freq_TSID_Feet_Sinu_Control:
    
    def __init__(self, logSize = None, dt = 0.001):
        
        self.DT = dt
        self.PRINT_N = 500                     
        self.DISPLAY_N = 25      
        
        # Robot model 
        self.dof = 8
 
        # SE3 Task         
        self.w_se3 = 1.0
        self.kp_se3 = 30.0
        self.level_se3 = 1
        self.foot = 'FR_FOOT'
          
        # Safety 
        self.error = False
        self.joint_error_list = []
        self.time_error = False
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
        self.foot_index = self.model.getFrameId(self.foot)
        robot_display = pin.RobotWrapper.BuildFromURDF(urdf, [path, ]) #with pinocchio
        self.model_display = robot_display.model
        self.data_display = robot_display.data
        self.frames = self.model_display.names
        self.frame_names = [ name for i,name in enumerate(self.frames)] 
        
        # Reference configuration
        pin.loadReferenceConfigurations(self.model_display, srdf)
        self.q_init = self.model_display.referenceConfigurations["standing"]
        
        
        if logSize is not None:
            self.torque_des_list = np.zeros((logSize,self.dof))
            self.q_cmd_list = np.zeros((logSize,self.dof))
            self.v_cmd_list = np.zeros((logSize,self.dof))
            self.p_des_list = np.zeros((logSize, 3))
        
        
    def Init(self, q_mes, v_mes):
        # Initial configuration
        self.q_cmd = q_mes.copy()
        self.v_cmd = v_mes.copy()
        self.dv_cmd = np.zeros(self.dof)
        self.torque_des = np.zeros(self.dof)


	    # Dynamics Problem initialization
        self.t = 0.0 # time
        self.invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", self.robot, False)
        self.invdyn.computeProblemData(self.t, self.q_cmd, self.v_cmd)

        self.se3Task = tsid.TaskSE3Equality("task-se3", self.robot, self.foot)
        self.se3Task.setKp(self.kp_se3 * np.ones(6))
        self.se3Task.setKd(2.0 * np.sqrt(self.kp_se3) * np.ones(6))
        mask = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.se3Task.setMask(mask)
        self.se3Task.useLocalFrame(False)
        self.invdyn.addMotionTask(self.se3Task, self.w_se3, self.level_se3, 0.0)
        se3_ref = self.robot.framePosition(self.invdyn.data(), self.foot_index)
        se3_target = se3_ref.copy()
        self.trajSE3 = tsid.TrajectorySE3Constant("traj_foot", se3_target)
        self.sampleSE3 = self.trajSE3.computeNext()

        self.offset = np.zeros(6)
        self.offset[:3] = se3_ref.translation
        self.amp                  = np.zeros(6)
        self.amp[0]               = 0.05
        self.two_pi_f             = np.zeros(6)
        self.two_pi_f[0]          = 2*np.pi*2.0
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
        


    def low_level(self, v_mes, q_mes, Kp, Kd, i):


        '''for index in [2,3]:
            if self.error or (q_mes[index]<-3.14) or (q_mes[index]>3.14) or (v_mes[index]<-30) or (v_mes[index]>30): 
                self.error = True
                self.torque_des = -self.security * v_mes
                self.t += self.DT
                print(i)
                return(self.torque_des, np.zeros(self.dof), np.zeros(self.dof), q_mes, v_mes)'''


        # TSID computation        

        self.sampleSE3.pos(self.offset + np.multiply(self.amp, np.sin(self.two_pi_f*self.t)))
        self.sampleSE3.vel(np.multiply(self.two_pi_f_amp, np.cos(self.two_pi_f*self.t)))
        self.sampleSE3.acc(np.multiply(self.two_pi_f_squared_amp, -np.sin(self.two_pi_f*self.t)))
        self.se3Task.setReference(self.sampleSE3)

        if i == 0:
            self.p_des = self.sampleSE3.pos() 
        
        HQPData = self.invdyn.computeProblemData(self.t, self.q_cmd, self.v_cmd)     
        self.sol = self.solver.solve(HQPData)
        if(self.sol.status!=0):
            print ("QP problem could not be solved! Error code:", self.sol.status)
            self.error = True

        ### Desired State
        self.dv_cmd = self.invdyn.getAccelerations(self.sol)
        self.v_cmd += self.dv_cmd * self.DT
        self.q_cmd = pin.integrate(self.model, self.q_cmd, self.v_cmd*self.DT)

        ### Feedforward Torque
        self.torque_des = self.invdyn.getActuatorForces(self.sol)
            
        self.p_des = self.sampleSE3.pos()
        self.t += self.DT
            
        return(self.torque_des, Kp, Kd, self.q_cmd, self.v_cmd)
     

    ### DATA LOGS AND PLOTS     
    def sample(self, i):
        self.torque_des_list[i,:] = self.torque_des[:]
        self.q_cmd_list[i,:] = self.q_cmd[:]
        self.v_cmd_list[i,:] = self.v_cmd[:]
        self.p_des_list[i,:] = self.p_des[:3]
        
    def saveAll(self, filename = "data"):
        date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')
        np.savez(filename + date_str + ".npz",
                 q=self.q_cmd_list, 
                 v=self.v_cmd_list,
                 p=self.p_des_list,
                 torque_des=self.torque_des_list)
   

