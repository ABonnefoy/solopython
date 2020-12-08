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
        self.DISPLAY_N = 25    
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
            self.jointTorques_list = np.zeros((logSize,self.dof))
            self.q_list = np.zeros((logSize,self.dof))
            self.v_list = np.zeros((logSize,self.dof))
            self.y_list = np.zeros((logSize, 3))
        
        
    def Init(self, qmes, vmes):
        # Initial configuration
        self.q = qmes.copy()
        self.v = vmes.copy()
        self.dv = np.zeros(self.dof)
        self.jointTorques = np.zeros(self.dof)
        self.tau = np.zeros(self.dof)
        self.dq = np.zeros(self.dof)

        pin.forwardKinematics(self.model, self.data, self.q, self.v, self.dv)
        pin.updateFramePlacements(self.model, self.data)
        y_mes = self.data.oMf[self.foot_index].translation
        self.y_mes = np.zeros(6)
        self.y_mes[:3] = y_mes
        self.y = self.y_mes.copy()
        self.dy = np.zeros(6)

        self.offset               = self.y_mes.copy()
        self.amp                  = np.zeros(6)
        self.amp[0]               = 0.05
        self.two_pi_f             = np.zeros(6)
        self.two_pi_f[0]          = 2*np.pi*1.0
        self.two_pi_f_amp         = np.multiply(self.two_pi_f,self.amp)
        self.two_pi_f_squared_amp = np.multiply(self.two_pi_f, self.two_pi_f_amp)


########## LOW-LEVEL CONTROL ##########

    def low_level(self, qmes, vmes, Kp, Kd, i):

        for index in range(len(qmes)):
            if self.error or (qmes[index]<-3.14) or (qmes[index]>3.14) or (vmes[index]<-30) or (vmes[index]>30): 
                self.error = True
                self.jointTorques = -self.security * vmes
                self.t += self.DT
                return(self.jointTorques, np.zeros(self.dof), np.zeros(self.dof), qmes, vmes)
     
        qp = self.q.copy()
        vp = self.v.copy()
        dvp = self.dv.copy()
        y_prev = self.y.copy()
        dy_prev = self.dy.copy()
        dqp = self.dq.copy()
 
        
        # TSID computation        

        self.y = self.offset + np.multiply(self.amp, np.sin(self.two_pi_f*self.t))
        self.dy = np.multiply(self.two_pi_f_amp, np.cos(self.two_pi_f*self.t))
        self.ddy = np.multiply(self.two_pi_f_squared_amp, -np.sin(self.two_pi_f*self.t))



        pin.forwardKinematics(self.model, self.data, qp, vp, dvp)
        pin.updateFramePlacements(self.model, self.data)
        J_foot = pin.computeFrameJacobian(self.model, self.data, qp, self.foot_index)

        self.v = np.linalg.pinv(J_foot) @ self.dy

        dJ = self.data.dJ
        ddy_cmd = self.kp_se3 * (self.y - y_prev) + 2.0 * np.sqrt(self.kp_se3) * (self.dy - dy_prev) + self.ddy
        self.dv = np.linalg.pinv(J_foot) @ (ddy_cmd + dJ @ self.v)

        #self.dq = dqp + np.linalg.pinv(J_foot) @ ( self.y - y_prev - J_foot @ dqp )
        self.q = qp + np.linalg.pinv(J_foot) @ (self.y - y_prev)
        #self.q = qp + self.dq

  
        self.jointTorques = pin.rnea(self.model, self.data, qmes, vmes, self.dv)
            
        self.t += self.DT
            
        return(self.jointTorques, Kp, Kd, self.q, self.v)
        



    def sample(self, i):
        self.jointTorques_list[i,:] = self.jointTorques[:]
        self.q_list[i,:] = self.q[:]
        self.v_list[i,:] = self.v[:]
        self.y_list[i,:] = self.y[:3]
        
    def saveAll(self, filename = "data"):
        date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')
        np.savez(filename + date_str + ".npz",
                 q=self.q_list, 
                 v=self.v_list,
                 y=self.y_list,
                 jointTorques=self.jointTorques_list)
   

