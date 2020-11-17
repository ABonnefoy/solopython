"""
Created on Tue Sep 29 15:02:15 2020

@author: abonnefoy
"""

import numpy as np
import os
from datetime import datetime as datetime

import pinocchio as pin
import gepetto.corbaserver

class Solo_Simu:
    
    def __init__(self, dt, logSize=None):
        
        self.dt = dt
        self.dof = 8


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

        self.nb_motors = self.dof
        
        if logSize is not None:
            self.q_mes_list = np.zeros((logSize,self.dof))
            self.v_mes_list = np.zeros((logSize,self.dof))
        
        
    def Init(self, q_init):
        self.q_mes = q_init
        self.v_mes = np.zeros(self.dof)
        self.a_mes = np.zeros(self.dof)
 
        self.robot.displayCollisions(False)
        self.robot.displayVisuals(True)
        self.robot.display(self.q_mes)  
        
        
    def runSimulation(self, jointTorques):
        self.a_mes = pin.aba(self.model, self.data, self.q_mes, self.v_mes, jointTorques)
        self.v_mes += self.a_mes * self.dt
        self.q_mes = pin.integrate(self.model, self.q_mes, self.v_mes*self.dt) 

        self.robot.display(self.q_mes)
        
    def sample(self,i):
        self.q_mes_list[i,:] = self.q_mes[:].flat
        self.v_mes_list[i,:] = self.v_mes[:].flat
        
    def saveAll(self, filename = "data"):
        date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')
        np.savez(filename + date_str + ".npz",  
                 q_mes=self.q_mes_list, 
                 v_mes=self.v_mes_list)
        
