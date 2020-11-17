import pinocchio as pin
import numpy as np
from IPython import embed
import time

def cross3(left,right):
	"""Numpy is inefficient for this"""
	return np.array([left[1] * right[2] - left[2] * right[1],
					 left[2] * right[0] - left[0] * right[2],
					 left[0] * right[1] - left[1] * right[0]])
class BaseEstimator():
	def __init__(self,urdf="/opt/openrobots/lib/python3.5/site-packages/../../../share/example-robot-data/robots/solo_description/robots/solo.urdf",modelPath="/opt/openrobots/lib/python3.5/site-packages/../../../share/example-robot-data/robots",contactFrameNames = ['HR_FOOT', 'HL_FOOT', 'FR_FOOT', 'FL_FOOT']):
		self.robot = pin.RobotWrapper.BuildFromURDF(urdf,modelPath)
		self.contactFrameIds = [self.robot.model.getFrameId(x) for x in contactFrameNames]
	def update_sensors(self, q, v, gyr):
		#compute the different velocities from all the contacts
		self.robot.forwardKinematics(q,v)
		self.BaseVelocitiesFromKin = [self.BaseVelocityFromKinAndIMU(q,v,gyr,x) for x in self.contactFrameIds]
	
	def BaseVelocityFromKinAndIMU(self,q,v,gyr, contactFrameId):
		frameVelocity = self.robot.frameVelocity(q,v,contactFrameId,update_kinematics=False)
		framePlacement = self.robot.framePlacement(q,contactFrameId,update_kinematics=False)
		
		# Angular velocity of the base wrt the world in the base frame (Gyroscope)
		_1w01 = gyr
		# Linear velocity of the foot wrt the base in the base frame
		_Fv1F = frameVelocity.linear
		# Level arm between the base and the foot
		_1F =  framePlacement.translation
		# Orientation of the foot wrt the base
		_1RF =  framePlacement.rotation
		# Linear velocity of the base from wrt world in the base frame
		#_1v01 = np.cross(_1F.A1 , _1w01) - (_1RF * _Fv1F).A1
		_1v01 = cross3(_1F.A1 , _1w01) - (_1RF * _Fv1F).A1
		return _1v01
	

def benchmark():
	import timeit
	a="import numpy as np; from solo_estimator import BaseEstimator; be=BaseEstimator(); q=np.zeros(8); v=np.zeros(8); gyr=np.zeros(3)"
	print(timeit.timeit('be.update_sensors(q,v,gyr)', setup=a, number=1000))
