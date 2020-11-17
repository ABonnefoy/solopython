"""
Created on Tue Sep 29 15:02:15 2020

@author: abonnefoy
"""

import pybullet as pyb
import pybullet_data
import numpy as np
import os
from datetime import datetime as datetime

class Solo_Simu_Pybullet:
    
    def __init__(self, dt, logSize=None):
        
        self.dt = dt
        self.dof = 8
        
        # Start the client for PyBullet
        physicsClient = pyb.connect(pyb.GUI) # or p.DIRECT for non-graphical version
        
        # Load horizontal plane
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = pyb.loadURDF("plane.urdf")
        
        # Set the gravity
        pyb.setGravity(0,0,-9.81)
        
        
        # Load Quadruped robot
        self.robotStartPos = [0,0,0.5]
        self.robotStartOrientation = pyb.getQuaternionFromEuler([0,0,0])
        pyb.setAdditionalSearchPath("/opt/openrobots/share/example-robot-data/robots/solo_description/robots")
        self.robotId = pyb.loadURDF('solo.urdf',self.robotStartPos, self.robotStartOrientation)
        self.revoluteJointIndices = [0,1, 3,4, 6,7, 9,10]
        self.nb_motors = len(self.revoluteJointIndices)
        pyb.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=(0.0*(180/3.1415)+45), cameraPitch=-39.9,
                                     cameraTargetPosition=self.robotStartPos)
        
        
        # Disable default motor control for revolute joints
        pyb.setJointMotorControlArray(self.robotId, jointIndices = self.revoluteJointIndices, controlMode = pyb.VELOCITY_CONTROL,targetVelocities = [0.0 for m in self.revoluteJointIndices], forces = [0.0 for m in self.revoluteJointIndices])
        
        if logSize is not None:
            self.q_mes_list = np.zeros((logSize,self.dof))
            self.v_mes_list = np.zeros((logSize,self.dof))
        
        
    def Init(self, q_init):
        # Initialize the joint configuration to the position straight_standing
        initial_joint_positions = [ joint for i,joint in enumerate(q_init)]
        for i in range (len(initial_joint_positions)):
            pyb.resetJointState(self.robotId, self.revoluteJointIndices[i], initial_joint_positions[i])
        
        # Enable torque control for revolute joints
        jointTorques = [0.0 for m in self.revoluteJointIndices]
        
        pyb.setJointMotorControlArray(self.robotId, self.revoluteJointIndices, controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)
        
        pyb.createConstraint(self.robotId, -1, -1, -1, pyb.JOINT_FIXED, [0, 0, 0], [0, 0, 0], self.robotStartPos) # Fixed base
        jointStates = pyb.getJointStates(self.robotId, self.revoluteJointIndices)
            
        self.q_mes = np.array([jointStates[i_joint][0] for i_joint in range(len(jointStates))])
        self.v_mes = np.array([jointStates[i_joint][1] for i_joint in range(len(jointStates))])
        
        # Set time step for the simulation
        pyb.setTimeStep(self.dt)
        
        
    def runSimulation(self, jointTorques):
        pyb.setJointMotorControlArray(self.robotId, self.revoluteJointIndices, controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)
        pyb.stepSimulation()
        jointStates = pyb.getJointStates(self.robotId, self.revoluteJointIndices)
        self.q_mes = np.array([jointStates[i_joint][0] for i_joint in range(len(jointStates))])
        self.v_mes = np.array([jointStates[i_joint][1] for i_joint in range(len(jointStates))]) 
        
    def sample(self,i):
        self.q_mes_list[i,:] = self.q_mes[:].flat
        self.v_mes_list[i,:] = self.v_mes[:].flat
        
    def saveAll(self, filename = "data"):
        date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')
        np.savez(filename + date_str + ".npz",  
                 q_mes=self.q_mes_list, 
                 v_mes=self.v_mes_list)
        
