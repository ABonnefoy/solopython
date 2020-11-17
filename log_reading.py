import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def reading(controllerfile, devicefile):

    date_device = devicefile[-20:-4]
    date_controller = controllerfile[-20:-4]

    assert date_device == date_controller

    log_device = np.load(devicefile)
    log_controller = np.load(controllerfile)

    q_mes = log_device['q_mes'].copy()
    v_mes = log_device['v_mes'].copy()
    tau_mes = log_device['torquesFromCurrentMeasurment'].copy()

    q_des = log_controller['q'].copy()
    v_des = log_controller['v'].copy()
    tau_des = log_controller['jointTorques'].copy()

    q_diff = q_mes-q_des
    v_diff = v_mes-v_des
    tau_diff = tau_mes-tau_des

    plotAll(q_diff, v_diff, tau_diff, date_device)

def plotAll(q, v, tau, date):
    path = os.path.dirname(__file__) 
    savefile = os.path.join(path, 'Images/')
    fontsize = 10
    frame_names = ['Hip', 'Knee']

    plt.figure(1)
    plt.suptitle('Torque difference (measured-desired)', fontsize = fontsize+2)
    for k in range(2):
        plt.subplot(2,1,k+1)
        plt.title(frame_names[k],fontsize = fontsize)
        plt.plot(tau[:,k], 'b', linestyle = 'dotted')
        axes = plt.gca()
        axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
        if (k==1): 
            axes.set_ylabel('Torque difference [Nm]',fontsize = fontsize)
            axes.set_xlabel('Time [ms]', fontsize = fontsize)
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'expe_TORQUES_%s.svg' %(date))
    plt.show() 

    plt.figure(1)
    plt.suptitle('Position difference (measured-desired)', fontsize = fontsize+2)
    for k in range(2):
        plt.subplot(2,1,k+1)
        plt.title(frame_names[k],fontsize = fontsize)
        plt.plot(q[:,k], 'b', linestyle = 'dotted')
        axes = plt.gca()
        axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
        if (k==1): 
            axes.set_ylabel('Position difference [rad]',fontsize = fontsize)
            axes.set_xlabel('Time [ms]', fontsize = fontsize)
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'expe_Q_%s.svg' %(date))
    plt.show() 

    plt.figure(1)
    plt.suptitle('Velocitie difference (measured-desired)', fontsize = fontsize+2)
    for k in range(2):
        plt.subplot(2,1,k+1)
        plt.title(frame_names[k],fontsize = fontsize)
        plt.plot(v[:,k], 'b', linestyle = 'dotted')
        axes = plt.gca()
        axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
        if (k==1): 
            axes.set_ylabel('Velocity difference [rad/s]',fontsize = fontsize)
            axes.set_xlabel('Time [ms]', fontsize = fontsize)
    plt.tight_layout(pad = 0.1, rect=[0, 0, 1, .92])
    plt.savefig(savefile + 'expe_V_%s.svg' %(date))
    plt.show() 

def main():
    parser = argparse.ArgumentParser(description='Example masterboard use in python.')
    parser.add_argument('-cf',
                        '--controllerfile',
                        type=str,
                        required=True,
                        help='Choose controller log file to read')
    parser.add_argument('-df',
                        '--devicefile',
                        type=str,
                        required=True,
                        help='Choose device log file to read')
    

    reading(parser.parse_args().controllerfile, parser.parse_args().devicefile)


if __name__ == "__main__":
    main()
