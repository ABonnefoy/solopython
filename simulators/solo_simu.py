"""
Created on Tue Sep 29 15:02:15 2020

@author: abonnefoy
"""

import numpy as np
import os
from datetime import datetime as datetime
from time import clock

import pinocchio as pin
import gepetto.corbaserver

class Solo_Simu:
    
    def __init__(self, dt, logSize=None, noise=False):
        
        self.dt = dt
        self.noise = noise
        self.dof = 8
        self.last = 0
        self.cpt = 0
        self.t = 0

        self.maximumTorque = 3.0

        # Gaussian parameters; Parameters chosen experimentally (Ewen) 
        self.hip_mean = -0.072 # Mean
        self.hip_std = np.sqrt(0.091) # Standard deviation
        self.knee_mean = 0.0031 # Mean
        self.knee_std = np.sqrt(0.00047) # Standard deviation

        # Robot creation
        path = os.path.dirname(__file__)
        urdf = os.path.join(path, 'solo_description/robots/solo.urdf')
        vector = pin.StdVec_StdString()
        srdf = path + '/example-robot-data/robots/solo_description/srdf/solo.srdf'
   
        self.robot = pin.RobotWrapper.BuildFromURDF("/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo.urdf")
        cl = gepetto.corbaserver.Client()
        gui = cl.gui
        self.robot.initViewer(loadModel=True)

        self.model = self.robot.model
        self.data = self.robot.data
        self.foot = 'FR_FOOT'
        self.foot_index = self.model.getFrameId(self.foot)

        self.nb_motors = self.dof
        
        if logSize is not None:
            self.q_mes_list = np.zeros((logSize,self.dof))
            self.v_mes_list = np.zeros((logSize,self.dof))
            self.v_noise_list = np.zeros((logSize,self.dof))
            self.p_mes_list = np.zeros((logSize, 3))
            self.torque_cmd_list = np.zeros((logSize,self.dof))
        
        
    def Init(self, q_init): # Initialize robot configuration
        self.q_mes = q_init
        self.v_mes = np.zeros(self.dof)
        self.v_noise = np.zeros(self.dof)
        self.a_mes = np.zeros(self.dof)
 
        self.robot.displayCollisions(False)
        self.robot.displayVisuals(True)
        self.robot.display(self.q_mes)  
        
    ### HIGH-LEVEL CONTROL    
    def runSimulation(self, torque_cmd):
        self.a_mes = pin.aba(self.model, self.data, self.q_mes, self.v_mes, torque_cmd)
        self.v_mes += self.a_mes * self.dt
        self.q_mes = pin.integrate(self.model, self.q_mes, self.v_mes*self.dt) 

        self.robot.display(self.q_mes)

    ### LOW-LEVEL CONTROL
    def SetDesiredJointTorque(self, torque_FF):
        self.feedforwardTorque = torque_FF
        for i in range(self.nb_motors):
            if self.feedforwardTorque[i] > self.maximumTorque:
                self.feedforwardTorque[i] = self.maximumTorque
            elif self.feedforwardTorque[i] < -self.maximumTorque:
                self.feedforwardTorque[i] = -self.maximumTorque

    def SetDesiredJointPDgains(self, Kp, Kd):
        self.Kp = Kp
        self.Kd = Kd

    def SetDesiredJointPosition(self, q_des):
        self.q_des = q_des

    def SetDesiredJointVelocity(self, v_des):
        self.v_des = v_des

    def SendCommand(self, WaitEndOfCycle=True):  
        torque = np.array(self.Kp * (self.q_des - self.q_mes) + self.Kd * (self.v_des - self.v_mes) + self.feedforwardTorque)
        self.torque_cmd = np.clip(torque, -self.maximumTorque * np.ones(8), self.maximumTorque * np.ones(8))

        self.a_mes = pin.aba(self.model, self.data, self.q_mes, self.v_mes, self.torque_cmd)
        self.v_mes += self.a_mes * self.dt

        if self.noise:
            # Measurement error
            v_hip = np.random.normal(self.hip_mean, self.hip_std, 4) # Adding a Gaussian to simulate noise
            v_knee = np.random.normal(self.knee_mean, self.knee_std, 4)    
            for j in range(4):
                self.v_noise[2*j] = self.v_mes[2*j]+v_hip[j]
                self.v_noise[2*j+1] = self.v_mes[2*j+1]+v_knee[j]
            self.q_mes = pin.integrate(self.model, self.q_mes, self.v_noise*self.dt) 

            pin.forwardKinematics(self.model, self.data, self.q_mes, self.v_noise, self.a_mes)
            pin.updateFramePlacements(self.model, self.data)
            self.p_mes = self.data.oMf[self.foot_index].translation
        else:
            self.q_mes = pin.integrate(self.model, self.q_mes, self.v_mes*self.dt) 

            pin.forwardKinematics(self.model, self.data, self.q_mes, self.v_mes, self.a_mes)
            pin.updateFramePlacements(self.model, self.data)
            self.p_mes = self.data.oMf[self.foot_index].translation

        self.robot.display(self.q_mes)
        if WaitEndOfCycle:
            self.WaitEndOfCycle()

    def WaitEndOfCycle(self):
        '''This Blocking fuction will wait for the end of timestep cycle (dt).'''
        while(1):
            if((clock() - self.last) >= self.dt):
                self.last = clock()
                self.cpt += 1
                self.t += self.dt
                return
      



    ### DATA LOGS AND PLOTS        
    def sample(self,i):
        self.q_mes_list[i,:] = self.q_mes[:].flat
        self.v_mes_list[i,:] = self.v_mes[:].flat
        self.p_mes_list[i,:] = self.p_mes[:].flat
        self.torque_cmd_list[i,:] = self.torque_cmd[:].flat
        
    def saveAll(self, filename = "data"):
        date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')
        np.savez(filename + date_str + ".npz", 
                 p_mes=self.p_mes_list, 
                 q_mes=self.q_mes_list, 
                 v_mes=self.v_mes_list, 
                 torque_cmd=self.torque_cmd_list)
        
