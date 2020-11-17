#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 17:26:17 2020

@author: abonnefoy
"""


import pinocchio as pin

import numpy as np
from numpy.linalg import norm as norm

import os



class Safety_Control:
    
    def __init__(self, logSize = None, safety = 0.05):
        
        self.DT = 0.001
        self.PRINT_N = 500                     
        self.DISPLAY_N = 25      
        
        # Robot model 
        self.dof = 8            
        
        # Safety 
        self.error = False
        self.joint_error_list = []
        self.time_error = False
        self.s = safety
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
        self.q = qmes
        self.v = vmes
        self.dv = np.zeros(self.dof)
        self.jointTorques = np.zeros(self.dof)
        self.t = 0.0 # time

        
    def control(self, qmes, vmes, i, Kd, Kp):
        
        torque = - self.s * vmes
        self.jointTorques = np.clip(torque, -self.tau_max * np.ones(8), self.tau_max * np.ones(8))        
        print("Safety controller, iteration = ", i)    
        self.t += self.DT            
        return(self.jointTorques)
    
    def sample(self, i):
        self.jointTorques_list[i,:] = self.jointTorques[:]
        self.q_list[i,:] = self.q[:]
        self.v_list[i,:] = self.v[:]
        
    def saveAll(self, filename = "data.npz"):
        np.savez(filename,  q=self.q_list, 
                            v=self.v_list,
                            jointTorques=self.jointTorques_list)
        

