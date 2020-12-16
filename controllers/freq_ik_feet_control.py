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


class Freq_IK_Feet_Control:
    
    def __init__(self, logSize = None, dt = 0.001):
        
        self.DT = dt
        self.PRINT_N = 500                     
        self.DISPLAp_N = 25    
        self.t = 0.0  
        
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
            self.tau_list = np.zeros((logSize,self.dof))
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
        self.dq = np.zeros(self.dof)

        pin.forwardKinematics(self.model, self.data, q_mes, v_mes)
        pin.updateFramePlacements(self.model, self.data)
        self.p_des = self.data.oMf[self.foot_index].translation
        self.dp_des = pin.getFrameVelocity(self.model, self.data, self.foot_index).linear

        # Desired trajectory initialisation
        self.offset               = self.p_des.copy()
        self.amp                  = np.zeros(3)
        self.amp[0]               = 0.05
        self.two_pi_f             = np.zeros(3)
        self.two_pi_f[0]          = 2*np.pi*2.0
        self.two_pi_f_amp         = np.multiply(self.two_pi_f,self.amp)
        self.two_pi_f_squared_amp = np.multiply(self.two_pi_f, self.two_pi_f_amp)


########## LOW-LEVEL CONTROL ##########

    def low_level(self, q_mes, v_mes, Kp, Kd, i):

        # Safety Controller
        for index in range(len(q_mes)):
            if self.error or (q_mes[index]<-3.14) or (q_mes[index]>3.14) or (v_mes[index]<-30) or (v_mes[index]>30): 
                self.error = True
                self.torque_des = -self.security * v_mes
                self.t += self.DT
                return(self.torque_des, np.zeros(self.dof), np.zeros(self.dof), q_mes, v_mes)
     
        # Previous Values
        q_prev = self.q_cmd.copy()
        v_prev = self.v_cmd.copy()
        dv_prev = self.dv_cmd.copy()
        p_prev = self.p_des.copy()
        dp_prev = self.dp_des.copy()
 
        
        # Desired trajectory update        

        self.p_des = self.offset + np.multiply(self.amp, np.sin(self.two_pi_f*self.t))
        self.dp_des = np.multiply(self.two_pi_f_amp, np.cos(self.two_pi_f*self.t))
        self.ddp_des = np.multiply(self.two_pi_f_squared_amp, -np.sin(self.two_pi_f*self.t))

        ### Inverse Kinematics

        J_foot = pin.computeFrameJacobian(self.model, self.data, q_prev, self.foot_index, pin.LOCAL_WORLD_ALIGNED)
        self.J_foot = J_foot[:3,:]

        # Desired Velocity
        self.v_cmd = np.linalg.pinv(self.J_foot) @ self.dp_des

        # Desired Configuration
        self.q_cmd = q_prev + np.linalg.pinv(self.J_foot) @ (self.p_des - p_prev)

        # Desired acceleration
        pin.forwardKinematics(self.model, self.data, q_mes, v_mes)
        pin.updateFramePlacements(self.model, self.data)
        self.p_mes = self.data.oMf[self.foot_index].translation
        self.dp_mes = pin.getFrameVelocity(self.model, self.data, self.foot_index).linear
        ddp_cmd = self.kp_se3 * (self.p_des - self.p_mes) + 2.0 * np.sqrt(self.kp_se3) * (self.dp_des - self.dp_mes) + self.ddp_des
        self.dv_cmd = np.linalg.pinv(self.J_foot) @ ddp_cmd

        # Feedforward Torque
        self.torque_des = pin.rnea(self.model, self.data, q_mes, v_mes, self.dv_cmd)
            
        self.t += self.DT
            
        return(self.torque_des, Kp, Kd, self.q_cmd, self.v_cmd)
        



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
   

