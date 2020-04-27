import sys
import numpy as np
from time import clock
from masterboard_utils import CalibrationController, mbs, GotoController


class RobotHAL():
    ''' This class provide a robot specific access to the solo generic hardware'''
 
    def __init__(self, interfaceName="eth0", dt=0.001):
        self.isInitialized = False
        self.InitRobotSpecificParameters()
        assert len(self.motorToUrdf) == self.nb_motors
        assert len(self.motorSign) == self.nb_motors
        assert self.nb_motors % 2 == 0
        # TODO assert mapping..
        assert self.maximumCurrent >= 0

        # decide how to search for the index, given that we are close to the real zero
        searchStrategy = self.nb_motors * [CalibrationController.ALTERNATIVE]

        # to be in [-2pi;+2pi]
        self.encoderOffsets = np.fmod(self.encoderOffsets, 2*np.pi)
        # to be in [-pi;+pi]
        self.encoderOffsets[self.encoderOffsets > +np.pi] -= 2*np.pi
        self.encoderOffsets[self.encoderOffsets < -np.pi] += 2*np.pi

        for i in range(self.nb_motors):
            if (self.encoderOffsets[i] > (np.pi/2.0)):
                searchStrategy[i] = CalibrationController.POSITIVE
            elif (self.encoderOffsets[i] < - (np.pi/2.0)):
                searchStrategy[i] = CalibrationController.NEGATIVE
        print(searchStrategy)
        print(self.encoderOffsets)
        self.t = 0
        self.cpt = 0
        self.last = 0
        self.dt = dt
        self.nb_motorDrivers = int(self.nb_motors/2)
        self.gearRatioSigned = np.zeros(8)
        self.gearRatioSigned = self.motorSign * self.gearRatio
        self.jointKtSigned = self.motorKt * self.gearRatioSigned  # Nm(joint)/A(motor)

        self.q_mes = np.zeros(self.nb_motors)
        self.v_mes = np.zeros(self.nb_motors)
        self.torquesFromCurrentMeasurment = np.zeros(self.nb_motors)
        self.baseAngularVelocity = np.zeros(3)
        self.baseOrientation = np.array([0., 0., 0., 1.])
        self.baseLinearAcceleration = np.zeros(3)
        self.hardware = mbs.MasterBoardInterface(interfaceName, 0)
        self.calibCtrl = CalibrationController(self.hardware, self.nb_motors, self.dt, Kd=0.01, Kp=3.0 ,searchStrategy=searchStrategy)
        self.gotoCtrl = GotoController(self.hardware, self.nb_motors, self.dt, Kd=0.01, Kp=3.0)

    def InitRobotSpecificParameters(self):
        '''
        This function initialises all robot specific parameters
        This function **must** be overloaded in a child class for different robots
        '''
        raise RuntimeError("This class is an abstract class. Please overload this method.")

    def Init(self,calibrateEncoders=False):
        # Initialization of the interface between the computer and the master board and the master board itself
        self.hardware.Init()
        self.EnableAllMotors()
        self.isInitialized = True

        self.InitMasterBoard() # Initialization of the master board

        if calibrateEncoders:
            for i in range(self.nb_motors):
                self.hardware.GetMotor(i).SetPositionOffset(self.encoderOffsets[i])
                self.hardware.GetMotor(i).enable_index_offset_compensation = True
            print("Running calibration...")
            self.RunHommingRoutine()
            print("End Of Calibration")
    
    @staticmethod
    def EulerToQuaternion(roll_pitch_yaw):
        roll, pitch, yaw = roll_pitch_yaw
        sr = np.sin(roll/2.)
        cr = np.cos(roll/2.)
        sp = np.sin(pitch/2.)
        cp = np.cos(pitch/2.)
        sy = np.sin(yaw/2.)
        cy = np.cos(yaw/2.)
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        return [qx, qy, qz, qw]
        # return [-qy, -qx, qz, -qw]  # return with a rotation of 90dege around Z

    def SetDesiredJointTorque(self, tau):
        for i in range(self.nb_motors):
            cur = tau[self.motorToUrdf[i]] / self.jointKtSigned[i]
            # cur = np.clip(cur,-self.maximumCurrent,self.maximumCurrent) #takes too long !
            if (cur > self.maximumCurrent):
                cur = self.maximumCurrent
            elif (cur < -self.maximumCurrent):
                cur = -self.maximumCurrent
            if self.hardware.GetMotor(i).IsEnabled():
                self.hardware.GetMotor(i).SetCurrentReference(cur)
            else:
                self.hardware.GetMotor(i).SetCurrentReference(0)
        return

    def EnableAllMotors(self):
        for i in range(self.nb_motorDrivers):
            self.hardware.GetDriver(i).motor1.SetCurrentReference(0)
            self.hardware.GetDriver(i).motor2.SetCurrentReference(0)
            self.hardware.GetDriver(i).motor1.Enable()
            self.hardware.GetDriver(i).motor2.Enable()
            self.hardware.GetDriver(i).EnablePositionRolloverError()
            self.hardware.GetDriver(i).SetTimeout(5)
            self.hardware.GetDriver(i).Enable()
        return

    def SendInit(self, WaitEndOfCycle=True):
        '''This (possibly blocking) fuction will send an init packet to the robot'''
        assert self.isInitialized, "The Robot HAL is not initialized. You have to call init() first"
        self.hardware.SendInit()
        if WaitEndOfCycle:
            self.WaitEndOfCycle()

    def InitMasterBoard(self):
        while not self.hardware.IsAckMsgReceived():
            self.SendInit(True)

    def UpdateMeasurment(self):
        '''This function will parse the last sensor packet, and convert position and velocity according to the robot actuation parameters'''
        self.hardware.ParseSensorData()
        for i in range(self.nb_motors):
            # TODO check integrity differently
            if self.hardware.GetMotor(i).IsEnabled():
                self.q_mes[self.motorToUrdf[i]] = self.hardware.GetMotor(i).GetPosition()/self.gearRatioSigned[i]
                self.v_mes[self.motorToUrdf[i]] = self.hardware.GetMotor(i).GetVelocity()/self.gearRatioSigned[i]
                self.torquesFromCurrentMeasurment[self.motorToUrdf[i]] = self.hardware.GetMotor(i).GetCurrent()*self.jointKtSigned[i]

        # Angular velocities of the base from IMU Gyroscope
        self.baseAngularVelocity[:] = self.rotateImuVectors(
            [self.hardware.imu_data_gyroscope(0), self.hardware.imu_data_gyroscope(1), self.hardware.imu_data_gyroscope(2)])

        # Orientation of the base from IMU Estimation Filter  
        self.baseOrientation[:] = self.rotateImuOrientation(self.EulerToQuaternion(
            [self.hardware.imu_data_attitude(0), self.hardware.imu_data_attitude(1), self.hardware.imu_data_attitude(2)]))

        # Linear Acceleration of the base from IMU Estimation Filter, note the rotation !                                                 
        self.baseLinearAcceleration[:] = self.rotateImuVectors(
            [self.hardware.imu_data_linear_acceleration(0), self.hardware.imu_data_linear_acceleration(1), self.hardware.imu_data_linear_acceleration(2)])

        return

    def SendCommand(self, WaitEndOfCycle=True):
        '''This (possibly blocking) fuction will send the command packet to the robot'''
        assert self.isInitialized, "The Robot HAL is not initialized. You have to call init() first"
        self.hardware.SendCommand()
        if WaitEndOfCycle:
            self.WaitEndOfCycle()

    def AreAllDriversConnected(self):
        '''Test if all motor drivers are connected'''
        for i in range(0, self.nb_motors, 2):
            if not (self.hardware.GetDriver(i).is_connected):
                return False
        return True

    def AreAllMotorsReady(self):
        '''Test if all motors are enabled and ready'''
        for i in range(self.nb_motors):
            if not (self.hardware.GetMotor(i).IsEnabled() and self.hardware.GetMotor(i).IsReady()):
                return False
        return True

    def RunHommingRoutine(self):
        '''This Blocking function will perform a homing Routine'''
        state = 0
        while (state != 3):
            if (self.hardware.IsTimeout()):
                return
            self.UpdateMeasurment()
            if (state == 0):
                if (self.AreAllMotorsReady()):
                    state = 1
            elif (state == 1):
                if (self.calibCtrl.ManageCalibration()):
                    state = 2
            elif (state == 2):
                if (self.gotoCtrl.ManageControl()):
                    state = 3
            self.SendCommand(WaitEndOfCycle=True)

        return

    def WaitEndOfCycle(self):
        '''This Blocking fuction will wait for the end of timestep cycle (dt).'''
        while(1):
            if((clock() - self.last) >= self.dt):
                self.last = clock()
                self.cpt += 1
                self.t += self.dt
                return

    def Print(self):
        print(chr(27) + "[2J")
        self.hardware.PrintIMU()
        self.hardware.PrintADC()
        self.hardware.PrintMotors()
        self.hardware.PrintMotorDrivers()
        self.hardware.PrintCmdStats()
        self.hardware.PrintSensorStats()
        print("q_mes = ", self.q_mes)
        print("v_mes = ", self.v_mes)
        sys.stdout.flush()
