import numpy as np
from IPython import embed
from matplotlib import pyplot as plt
from solo_estimator import BaseEstimator
data = np.load("/tmp/data_main_solo12_estimation.npz")
# data = np.load("data_main_solo12_estimation_with_control.npz")
# data = np.load("data_main_solo12_estimation.npz")
q_mes = data['q_mes']
v_mes = data['v_mes']
baseOrientation = data['baseOrientation']
baseAngularVelocity = data['baseAngularVelocity']
baseLinearAcceleration = data['baseLinearAcceleration']
mocapPosition = data['mocapPosition']
mocapVelocity = data['mocapVelocity']
mocapAngularVelocity = data['mocapAngularVelocity']
mocapOrientationMat9 = data['mocapOrientationMat9']
mocapOrientationQuat = data['mocapOrientationQuat']

dt=0.001
N = len(q_mes)

mocapVelocityRotated = np.zeros([N,3])
for i in range(N):
    mocapVelocityRotated[i] = (np.matrix(mocapOrientationMat9[i])*np.matrix(mocapVelocity[i]).T).A1

mocapAngularVelocityRotated = np.zeros([N,3])
for i in range(N):
    mocapAngularVelocityRotated[i] = (np.matrix(mocapOrientationMat9[i]).T * np.matrix(mocapAngularVelocityRotated[i]).T).A1

velocityImuIntegration = np.zeros([N,3])
for i in range(1,N):
    velocityImuIntegration[i]=velocityImuIntegration[i-1]+baseLinearAcceleration[i]*dt

#This will come from estimator

velocityKinematic =  np.zeros([4,N,3])

be = BaseEstimator(urdf="/opt/openrobots/lib/python3.5/site-packages/../../../share/example-robot-data/robots/solo_description/robots/solo12.urdf",modelPath="/opt/openrobots/lib/python3.5/site-packages/../../../share/example-robot-data/robots",contactFrameNames = ['HR_FOOT', 'HL_FOOT', 'FR_FOOT', 'FL_FOOT'])

#example of use:
for i in range(N):
    be.update_sensors(q_mes[i],v_mes[i],baseAngularVelocity[i])
    velocityKinematic[0][i] = be.BaseVelocitiesFromKin[0]
    velocityKinematic[1][i] = be.BaseVelocitiesFromKin[1]
    velocityKinematic[2][i] = be.BaseVelocitiesFromKin[2]
    velocityKinematic[3][i] = be.BaseVelocitiesFromKin[3]

#complementary_filter
def HP(x,alpha):
    N=len(x)
    y = np.zeros(x.shape)
    y[0]=0
    for i in range(1,N):
        y[i]=alpha*(x[i]-x[i-1] + y[i-1])
    return y

def LP(x,alpha):
    N=len(x)
    y = np.zeros(x.shape)
    y[0]=x[0]
    for i in range(1,N):
        y[i]=alpha*y[i-1] + (1-alpha)*x[i]
    return y






plt.figure("linear velocity: Imu Integration/Mocap Rotated")
for axis in range(3):
    plt.subplot(3,1,axis+1)
    plt.plot(velocityImuIntegration[:,axis])
    plt.plot(mocapVelocityRotated[:,axis])
    #plt.plot(velocityKinematic[0][:,axis])
    #plt.plot(velocityKinematic[1][:,axis])
    #plt.plot(velocityKinematic[2][:,axis])
    #plt.plot(velocityKinematic[3][:,axis])

plt.figure("linear velocity: Estimated/Mocap Rotated")
alpha=0.99
#mocapLP = LP(mocapVelocityRotated,alpha)
imuHP = HP(velocityImuIntegration,alpha)

velocityKinematic0LP = LP(velocityKinematic[0],alpha)
velocityKinematic1LP = LP(velocityKinematic[1],alpha)
velocityKinematic2LP = LP(velocityKinematic[2],alpha)
velocityKinematic3LP = LP(velocityKinematic[3],alpha)

for axis in range(3):
    plt.subplot(3,1,axis+1)
    plt.plot(velocityKinematic0LP[:,axis]+imuHP[:,axis],'g')
    plt.plot(velocityKinematic1LP[:,axis]+imuHP[:,axis],'b')
    plt.plot(velocityKinematic2LP[:,axis]+imuHP[:,axis])
    plt.plot(velocityKinematic3LP[:,axis]+imuHP[:,axis])
    plt.plot(mocapVelocityRotated[:,axis],'r')

plt.figure("Angular velocity: Estimated/Mocap Rotated")
for axis in range(3):
    plt.subplot(3,1,axis+1)
    plt.plot(baseAngularVelocity[:,axis],'g')
    plt.plot(mocapVelocityRotated[:,axis],'r')

plt.show()





