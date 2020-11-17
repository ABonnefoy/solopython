# solopython
Python scripts to control Solo robot.


## Main simulation code: 
*main_simu_solo8_tsid_control.py*


## Main experimental code: 
*main_solo8_tsid_control.py*


## How to run the code ? 
```python3 *main_code.py* -exp *controller number*```


## Different controllers:

    0 - Safety controller

### Impedance control/ No FeedForward term/ Simple trajectories: 

    1 - Static

    2 - Small sinusoidal knee motion

    3 - Small sinusoidal hip & knee motions

    4 - Big sinusoidal hip & knee motions



### Impedance control/ FeedForward term/ Simple trajectories:

    5 - Static

    6 - Small sinusoidal knee motion

    7 - Small sinusoidal hip & knee motions

    8 - Big sinusoidal hip & knee motions


### Impedance control/ No FeedForward term/ TSID generated trajectories:

    9 - Static

    10 - Small sinusoidal knee motion

    11 - Small sinusoidal hip & knee motions

    12 - Big sinusoidal hip & knee motions


### Impedance control/ FeedForward term/ TSID generated trajectories:

    13 - Static

    14 - Small sinusoidal knee motion

    15 - Small sinusoidal hip & knee motions

    16 - Big sinusoidal hip & knee motions


### Direct Drive control/ TSID generated trajectories:

    17 - Static

    18 - Small sinusoidal knee motion

    19 - Small sinusoidal hip & knee motions

    20 - Big sinusoidal hip & knee motions

