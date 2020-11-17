import numpy as np
import libmaster_board_sdk_pywrap as mbs
import sys
from time import clock, sleep
import math
'''This file collects a set of usefull functions to maniputate a master_board_sdk object'''
class GotoController():
    def __init__(self,robot_if,nb_motors,dt,T_move=1.0, T_static=5.0,Kp = 2.0,Kd = 0.05,imax=3.0,FinalPosition=None):
        '''
        Control motors to reach a given position 'pos' in T_move seconds, 
        then hold this position for T_static seconds
        '''
        self.nb_motors = nb_motors
        self.robot_if = robot_if
        self.dt = dt  

        if (FinalPosition==None):
            self.FinalPosition = nb_motors * [0.]
        else:
            self.FinalPosition = FinalPosition
        self.InitialPosition = nb_motors * [0.]
        self.dt = dt

        self.READING_INITIAL_POSITION = 0
        self.CONTROLLING = 1
        
        self.state = nb_motors * [self.READING_INITIAL_POSITION]
        self.control = nb_motors * [0.]
        self.Kp = Kp
        self.Kd = Kd
        self.imax = imax

        self.T_move = T_move
        self.T_static = T_static
        
        self.t = 0.0

    def ManageControl(self):
        self.t+=self.dt
        ended = True
        for motor in range(self.nb_motors):
            if self.robot_if.GetMotor(motor).IsEnabled():
                #*** State machine ***
                if (self.state[motor] == self.READING_INITIAL_POSITION):
                # READING INITIAL POSITION
                    self.InitialPosition[motor] = self.robot_if.GetMotor(motor).GetPosition()
                    self.state[motor] = self.CONTROLLING

                elif (self.state[motor] == self.CONTROLLING):
                # POSITION CONTROL
                    if (self.t<self.T_move):
                        traj = self.InitialPosition[motor] + (self.FinalPosition[motor]-self.InitialPosition[motor])*0.5*(1-math.cos(2*math.pi*(0.5/self.T_move)*self.t))
                    else:
                        traj = self.FinalPosition[motor]
                    self.control[motor] = self.Kp*(traj - self.robot_if.GetMotor(motor).GetPosition() - self.Kd*self.robot_if.GetMotor(motor).GetVelocity())
                #*** END OF STATE MACHINE ***
                
                ended = self.t>(self.T_static+self.T_move)
                #disable the controller at the end
                if (ended):
                    self.control[motor] = 0.0     

                self.control[motor] = min(self.imax, max(-self.imax, self.control[motor]))
                self.robot_if.GetMotor(motor).SetCurrentReference(self.control[motor])
        return (ended)

class CalibrationController():
    '''Control motors to find encoders index'''
    POSITIVE = +1
    NEGATIVE = -1
    ALTERNATIVE = 0
    def __init__(self, robot_if, nb_motors, dt, Kp = 2.0,Kd = 0.05,imax=3.0,searchStrategy=None ):
        self.nb_motors = nb_motors
        self.robot_if = robot_if
        self.dt = dt   

        self.control = nb_motors * [0.]
        self.InitialPosition = nb_motors * [0.]

        self.READING_INITIAL_POSITION = 0
        self.SEARCHING_INDEX = 1
        self.CALIBRATION_DONE = 2

        self.state = nb_motors * [self.READING_INITIAL_POSITION]
        if (searchStrategy==None):
            self.searchStrategy = nb_motors * [self.ALTERNATIVE]
        else:
            self.searchStrategy=searchStrategy

        self.Kp = Kp
        self.Kd = Kd
        self.imax = imax
        self.all_motors_calibrated = False
        self.t = 0

    def ManageCalibration(self):
        self.t+=self.dt
        test_calibrated = True
        for motor in range(self.nb_motors):
            if self.robot_if.GetMotor(motor).IsEnabled():
                #*** State machine for calibration ***
                if (self.state[motor] == self.READING_INITIAL_POSITION):
                # READING INITIAL POSITION
                    self.InitialPosition[motor] = self.robot_if.GetMotor(motor).GetPosition()
                    self.state[motor] = self.SEARCHING_INDEX

                elif (self.state[motor] == self.SEARCHING_INDEX):
                # POSITION CONTROL TO FIND INDEX
                    if (self.robot_if.GetMotor(motor).HasIndexBeenDetected()):
                        self.control[motor] = 0
                        self.state[motor] = self.CALIBRATION_DONE
                    else:
                        T=2.0
                        if (self.searchStrategy[motor] == self.ALTERNATIVE):
                            if (self.t<T/2.0):
                                calib_traj = self.InitialPosition[motor]+1.2*math.pi*0.5*(1-math.cos(2*math.pi*(1./T)*self.t))
                            else:
                                calib_traj = self.InitialPosition[motor]+1.2*math.pi*math.cos(2*math.pi*(0.5/T)*(self.t-T/2.0))
                        elif (self.searchStrategy[motor] == self.POSITIVE):
                            calib_traj = self.InitialPosition[motor]+1.2*math.pi*1*(1-math.cos(2*math.pi*(0.5/T)*self.t))
                        elif (self.searchStrategy[motor] == self.NEGATIVE):
                            calib_traj = self.InitialPosition[motor]-1.2*math.pi*1*(1-math.cos(2*math.pi*(0.5/T)*self.t))
                        self.control[motor] = self.Kp*(calib_traj - self.robot_if.GetMotor(motor).GetPosition() - self.Kd*self.robot_if.GetMotor(motor).GetVelocity())
                #*** END OF STATE MACHINE ***

                self.control[motor] = min(self.imax, max(-self.imax, self.control[motor]))
                self.robot_if.GetMotor(motor).SetCurrentReference(self.control[motor])
            if (self.state[motor] != self.CALIBRATION_DONE):
                test_calibrated = False
        self.all_motors_calibrated = test_calibrated

        return self.all_motors_calibrated
