'''

Simple python script to sample the MiM slider box data via USB
Thomas FLAYOLS - LAAS CNRS

Use:
To display data, run "python slider_box.py"
'''
import serial
import time
from copy import deepcopy
from multiprocessing import Process
from multiprocessing.sharedctypes import Value
from ctypes import c_double, c_bool


class SliderBox():

    def __init__(self):
        self.running = Value(c_bool, lock=True)
        self.e_stop = Value(c_bool, lock=True)
        self.slider_a = Value(c_double, lock=True)
        self.slider_b = Value(c_double, lock=True)
        self.slider_c = Value(c_double, lock=True)
        self.slider_d = Value(c_double, lock=True)

        self.e_stop.value = True
        self.slider_a.value = 0.0
        self.slider_b.value = 0.0
        self.slider_c.value = 0.0
        self.slider_d.value = 0.0

        args = (self.running, self.e_stop, self.slider_a,
                self.slider_b, self.slider_c, self.slider_d)
        self.process = Process(target=self.run, args=args)
        self.process.start()
        time.sleep(0.2)

    def run(self, running, e_stop, slider_a, slider_b, slider_c, slider_d):
        # open serial port, Change if needed
        serial_connexion = serial.Serial(
            '/dev/ttyACM0', 115200, timeout=None)
        # Read the slider box
        running.value = True
        while(running.value):
            data_str = serial_connexion.readline()
            data_str = serial_connexion.readline()
            data_str_list = data_str.split()
            if len(data_str_list) == 5:
                e_stop.value = not bool(int(data_str_list[0]))
                slider_a.value = float(data_str_list[1]) / 1024.0
                slider_b.value = float(data_str_list[2]) / 1024.0
                slider_c.value = float(data_str_list[3]) / 1024.0
                slider_d.value = float(data_str_list[4]) / 1024.0
                serial_connexion.flushInput()
        # Close the file
        serial_connexion.close()

    def stop(self):
        self.running.value = False
        self.process.terminate()
        self.process.join()

    def get_e_stop(self):
        return self.e_stop.value

    def get_slider_a(self):
        return self.slider_a.value

    def get_slider_b(self):
        return self.slider_b.value

    def get_slider_c(self):
        return self.slider_c.value

    def get_slider_d(self):
        return self.slider_d.value


if __name__ == "__main__":
    slider_box = SliderBox()
    for i in range(1000):
        print("E-stop = ", slider_box.get_e_stop(), end=" ; ")
        print("Slider A = ", slider_box.get_slider_a(), end=" ; ")
        print("Slider B = ", slider_box.get_slider_b(), end=" ; ")
        print("Slider C = ", slider_box.get_slider_c(), end=" ; ")
        print("Slider D = ", slider_box.get_slider_d())
        time.sleep(0.1)

    slider_box.stop()
