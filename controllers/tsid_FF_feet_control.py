#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:30:01 2020

@author: abonnefoy
"""

import pinocchio as pin
import tsid

import numpy as np
from numpy.linalg import norm as norm
from datetime import datetime as datetime


import os
import sys

from controllers.safety_control import Safety_Control


class TSID_FF_Feet_Control:
    
    def __init__(self, logSize = None):
        
        self.DT = 0.001
        self.PRINT_N = 500                     
        self.DISPLAY_N = 25      
        
        # Robot model 
        self.dof = 8

        # Posture Task         
        self.w_posture = 1e-3
        self.kp_posture = 10.0
        self.level_posture = 1

        # SE3 Task         
        self.w_se3 = 1.0
        self.kp_se3 = 30.0
        self.level_se3 = 1
        self.foot = 'FR_FOOT'
           
        # Safety 
        self.error = False
        self.joint_error_list = []
        self.time_error = False
        self.safety_controller = Safety_Control()
        self.tau_max = 3.0
        
        
        ########## ROBOT MODEL CREATION ##########
    
        # Definition of the path for the urdf and srdf files of the robot
        path = "/opt/openrobots/share/example-robot-data/robots/"
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
        
        
        # Reference configuration
        pin.loadReferenceConfigurations(self.model_display, srdf)
        self.q_init = self.model_display.referenceConfigurations["standing"].copy()
        
        
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

        self.offset = np.zeros([6])
        self.offset[:3] = se3_ref.translation
        self.amp                  = np.zeros(6)
        self.amp[2]               = 0.1
        self.two_pi_f             = np.zeros(6)
        self.two_pi_f[2]          = 2*np.pi*0.5
        self.two_pi_f_amp         = np.multiply(self.two_pi_f,self.amp)
        self.two_pi_f_squared_amp = np.multiply(self.two_pi_f, self.two_pi_f_amp)
        
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
     
        self.dv = self.invdyn.getAccelerations(self.sol)
        self.v += self.dv*self.DT
        self.q = pin.integrate(self.model, self.q, self.v*self.DT)
           
        # TSID computation        

        self.sampleSE3.pos(self.offset + np.multiply(self.amp, np.sin(self.two_pi_f*self.t)))
        self.sampleSE3.vel(np.multiply(self.two_pi_f_amp, np.cos(self.two_pi_f*self.t)))
        self.sampleSE3.acc(np.multiply(self.two_pi_f_squared_amp, -np.sin(self.two_pi_f*self.t)))
        self.se3Task.setReference(self.sampleSE3)

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
            torque_FB = np.array(Kp * (self.q - qmes) + Kd * (self.v - vmes))
            torque_FF = self.invdyn.getActuatorForces(self.sol)
            torque = torque_FB + torque_FF
            for index in range(len(qmes)):
                if (qmes[index]<-3.14) or (qmes[index]>3.14 or (vmes[index]<-30) or (vmes[index]>30)): # or (vmes[index]<-30) or (vmes[index]>30)
                    torque = self.safety_controller.control(qmes, vmes, i, Kp, Kd)
        self.jointTorques = np.clip(torque, -self.tau_max * np.ones(8), self.tau_max * np.ones(8))

            
        self.t += self.DT

            
        return(self.jointTorques)

    def low_level_control(self):
     
        self.dv = self.invdyn.getAccelerations(self.sol)
        self.v += self.dv*self.DT
        self.q = pin.integrate(self.model, self.q, self.v*self.DT)
           
        # TSID computation
        self.samplePosture.pos(self.offset + np.multiply(self.amp, np.sin(self.two_pi_f*self.t)))
        self.samplePosture.vel(np.multiply(self.two_pi_f_amp, np.cos(self.two_pi_f*self.t)))
        self.samplePosture.acc(np.multiply(self.two_pi_f_squared_amp, -np.sin(self.two_pi_f*self.t)))
        self.postureTask.setReference(self.samplePosture)
        
        
        HQP = self.invdyn.computeProblemData(self.t, self.q, self.v)     
        self.sol = self.solver.solve(HQPData)
        if(self.sol.status!=0):
            print ("QP problem could not be solved! Error code:", self.sol.status)
            self.error = True
  
        self.jointTorques = self.invdyn.getActuatorForces(self.sol)
        q_des = self.trajPosture.pos()
        v_des = self.trajPosture.vel()
            
        self.t += self.DT
            
        return(self.jointTorques, q_des, v_des)
    
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
        

