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

class Direct_TSID_Static_Control:
    
    def __init__(self, logSize = None):
        
        self.DT = 0.001
        self.PRINT_N = 500                     
        self.DISPLAY_N = 25      
        
        # Robot model 
        self.dof = 8

        # Posture Task         
        self.w_posture = 1.0
        self.kp_posture = 10.0
        self.level_posture = 1
        self.q_disp = np.zeros(self.dof)
        self.amp_hip = 0.4
        self.amp_knee = 0.8
        self.freq = 0.5
           
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
        self.robot = tsid.RobotWrapper(urdf, vector, False) # with tsid
        self.model = self.robot.model()
        robot_display = pin.RobotWrapper.BuildFromURDF(urdf, [path, ]) #with pinocchio
        model_display = robot_display.model
        self.frames = model_display.names
        self.frame_names = [ name for i,name in enumerate(self.frames)] 
        
        # Reference configuration
        pin.loadReferenceConfigurations(model_display, srdf)
        self.q_init = model_display.referenceConfigurations["standing"]
        
        
        if logSize is not None:
            self.tau_list = np.zeros((logSize,self.dof))
            self.jointTorques_list = np.zeros((logSize,self.dof))
            self.q_list = np.zeros((logSize,self.dof))
            self.v_list = np.zeros((logSize,self.dof))
        
        
    def Init(self, qmes, vmes):
        # Initial configuration
        self.q = qmes.copy()
        self.v = vmes.copy()
        self.dv = np.zeros(self.dof)
        self.jointTorques = np.zeros(self.dof)
        self.tau = np.zeros(self.dof)
	
        self.safety_controller.Init(qmes, vmes)

	# Dynamics Problem initialization
        self.t = 0.0 # time
        self.invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", self.robot, False)
        self.invdyn.computeProblemData(self.t, self.q, self.v)
        
        # Task definition
        self.postureTask = tsid.TaskJointPosture("task-posture", self.robot)
        self.postureTask.setKp(self.kp_posture * np.ones(self.dof)) 
        self.postureTask.setKd(2.0 * np.sqrt(self.kp_posture) * np.ones(self.dof))
        self.invdyn.addMotionTask(self.postureTask, self.w_posture, self.level_posture, 0.0)
        q_ref = self.q.copy() 
        self.trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref) # Goal = static
        self.samplePosture = self.trajPosture.computeNext()

        # Solver
        self.solver = tsid.SolverHQuadProgFast("qp solver")
        self.solver.resize(self.invdyn.nVar, self.invdyn.nEq, self.invdyn.nIn)
        HQPData = self.invdyn.computeProblemData(self.t, self.q, self.v)
        HQPData.print_all()        
        self.sol = self.solver.solve(HQPData)
        if(self.sol.status!=0):
            print ("QP problem could not be solved! Error code:", self.sol.status)
            self.error = True


        
    def control(self, qmes, vmes, i, Kp, Kd):

        self.q = qmes.copy()
        self.v = vmes.copy()
        
        # TSID computation
        self.samplePosture = self.trajPosture.computeNext()
        self.postureTask.setReference(self.samplePosture)
        
        
        HQPData = self.invdyn.computeProblemData(self.t, self.q, self.v)     
        self.sol = self.solver.solve(HQPData)
        if(self.sol.status!=0):
            print ("QP problem could not be solved! Error code:", self.sol.status)
            self.error = True


        # Safety controller
        if self.error:
            torque = self.safety_controller.control(qmes, vmes, i, Kp, Kd)
            self.error=False
        # Proportional controller
        else:  
            torque = self.invdyn.getActuatorForces(self.sol) 
            for index in range(len(qmes)):
                if (qmes[index]<-3.14) or (qmes[index]>3.14) or (vmes[index]<-30) or (vmes[index]>30):
                    torque = self.safety_controller.control(qmes, vmes, i, Kp, Kd)
        self.jointTorques = np.clip(torque, -self.tau_max * np.ones(8), self.tau_max * np.ones(8))
        

        self.t += self.DT
        self.dv = self.invdyn.getAccelerations(self.sol)
        self.v += self.dv*self.DT
        self.q = pin.integrate(self.model, self.q, self.v*self.DT)       
     
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
        

