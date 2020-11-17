#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 17:26:17 2020

@author: abonnefoy
"""


import pinocchio as pin

import numpy as np
from numpy.linalg import norm as norm
from datetime import datetime as datetime


import os

from controllers.safety_control import Safety_Control


class Sinu_1_Control:
    
    def __init__(self, logSize = None, difficulty = 0):
        
        self.DT = 0.001
        self.PRINT_N = 500                     
        self.DISPLAY_N = 25      
        
        # Robot model 
        self.dof = 8
        
        # Desired motion
        self.amp_hip = difficulty * .5
        self.amp_knee = difficulty * 1.0
        self.freq = 0.8
        
        
        # Safety 
        self.error = False
        self.joint_error_list = []
        self.time_error = False
        self.tau_max = 3.0
        self.safety_controller = Safety_Control()
        
        
        ########## ROBOT MODEL CREATION ##########
    
        # Definition of the path for the urdf and srdf files of the robot
        path = os.path.dirname(__file__)
        path = os.path.join(path, '../.')
        urdf = os.path.join(path, 'solo_description/robots/solo.urdf')
        vector = pin.StdVec_StdString()
        vector.extend(item for item in path)        
        srdf = os.path.join(path, 'solo_description/srdf/solo.srdf')
                  
        # Creation of the robot models
        robot_display = pin.RobotWrapper.BuildFromURDF(urdf, [path, ]) #with pinocchio
        model_display = robot_display.model
        self.frames = model_display.names
        self.frame_names = [ name for i,name in enumerate(self.frames)] 
        
        # Reference configuration
        pin.loadReferenceConfigurations(model_display, srdf)
        self.q_init = model_display.referenceConfigurations["standing"]
        
        
        if logSize is not None:
            self.jointTorques_list = np.zeros((logSize,self.dof))
            self.q_list = np.zeros((logSize,self.dof))
            self.v_list = np.zeros((logSize,self.dof))
        
        
        
    def Init(self, qmes, vmes):
        # Initial configuration
        self.q = qmes.copy()
        self.v = vmes.copy()
        self.dv = np.zeros(self.dof)
        self.jointTorques = np.zeros(self.dof)
        self.t = 0.0 # time
	
        self.safety_controller.Init(qmes, vmes)        
            
        self.offset = np.reshape(self.q_init, 8)
        self.amp                  = np.zeros(8)
        self.amp[1]               = self.amp_knee
        self.two_pi_f             = np.zeros(8)
        self.two_pi_f[1]          = 2*np.pi*self.freq
        self.two_pi_f_amp         = np.multiply(self.two_pi_f,self.amp)
        self.two_pi_f_squared_amp = np.multiply(self.two_pi_f, self.two_pi_f_amp)

        
    def control(self, qmes, vmes, i, Kp, Kd):
        
        self.q = self.offset + np.multiply(self.amp, np.sin(self.two_pi_f*self.t))
        self.v = np.multiply(self.two_pi_f_amp, np.cos(self.two_pi_f*self.t))
        self.dv = np.multiply(self.two_pi_f_squared_amp, -np.sin(self.two_pi_f*self.t))
        
        # Safety controller
        if self.error:
            torque = self.safety_controller.control(qmes, vmes, i, Kp, Kd)
            self.error=False
        # Proportional controller
        else:   
            torque = np.array(Kp * (self.q - qmes) + Kd * (self.v-vmes))
            for index in range(len(qmes)):
                if (qmes[index]<-3.14) or (qmes[index]>3.14) or (vmes[index]<-30) or (vmes[index]>30):
                    torque = self.safety_controller.control(qmes, vmes, i, Kp, Kd)
        self.jointTorques = np.clip(torque, -self.tau_max * np.ones(8), self.tau_max * np.ones(8))

            
        self.t += self.DT            
        return(self.jointTorques)
    
    def sample(self, i):
        self.jointTorques_list[i,:] = self.jointTorques[:]
        self.q_list[i,:] = self.q[:]
        self.v_list[i,:] = self.v[:]
        
    def saveAll(self, filename = "data"):
        date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')
        np.savez(filename + date_str + ".npz",
                 q=self.q_list, 
                 v=self.v_list,
                 jointTorques=self.jointTorques_list)
        

