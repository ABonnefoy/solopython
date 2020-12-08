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


class Freq_FF_Sinu_Control:
    
    def __init__(self, logSize = None, dt = 0.001, freq_hip = 0.0, amp_hip = 0.0, freq_knee = 0.0, amp_knee = 0.0):
        
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
        
        
        # Safety 
        self.error = False
        self.joint_error_list = []
        self.time_error = False
        self.safety_controller = Safety_Control()
        self.tau_max = 3.0
        
        
        ########## ROBOT MODEL CREATION ##########
    
        # Definition of the path for the urdf and srdf files of the robot
        path = os.path.dirname(__file__)
        path = os.path.join(path, '../.')
        urdf = os.path.join(path, 'solo_description/robots/solo.urdf')
        vector = pin.StdVec_StdString()
        vector.extend(item for item in path)        
        srdf = os.path.join(path, 'solo_description/srdf/solo.srdf')
                  
        # Creation of the robot models
        robot = pin.RobotWrapper.BuildFromURDF(urdf, [path, ]) #with pinocchio
        self.model = robot.model
        self.data = robot.data
        self.frames = self.model.names
        self.frame_names = [ name for i,name in enumerate(self.frames)] 
        
        # Reference configuration
        pin.loadReferenceConfigurations(self.model, srdf)
        self.q_init = self.model.referenceConfigurations["standing"]
        
        
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
        self.amp[0]               = self.amp_hip
        self.amp[1]               = self.amp_knee
        self.two_pi_f             = np.zeros(8)
        self.two_pi_f[0]          = 2*np.pi*self.freq_hip
        self.two_pi_f[1]          = 2*np.pi*self.freq_knee
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

        self.q = self.offset + np.multiply(self.amp, np.sin(self.two_pi_f*self.t))
        self.v = np.multiply(self.two_pi_f_amp, np.cos(self.two_pi_f*self.t))
        self.dv = np.multiply(self.two_pi_f_squared_amp, -np.sin(self.two_pi_f*self.t))
  
        #self.jointTorques = self.invdyn.getActuatorForces(self.sol)
        self.jointTorques = pin.rnea(self.model, self.data, qmes, vmes, self.dv)
            
        self.t += self.DT
            
        return(self.jointTorques, Kp, Kd, self.q, self.v)
    
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
        

