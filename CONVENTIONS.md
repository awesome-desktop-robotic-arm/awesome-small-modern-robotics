# ASMR CONVENTIONS

All rotations follow right hand rules.

## Symbols

q = [joint angles] in radians  
qd = [joint velocities] in radians/s   
qdd = [joint accelerations] in radians/s^2

x = [ee position] in meters  
xd = [ee velocities] in meters/s   
xdd = [ee accelerations] in meters/s^2  

p = [translations in 3-vector] in meters   
R = [Rotations in SO(3)]       
T = [Transformations in SE(3)]    

F = [wrench in 6-vector] -> [Force 3-vector; Torque 3-vector]
tau = [joint torques]

J = [Jacobian in world frame] -> [J_linear; J_angular]   

> [!Note]
> All jacobians default to world frame space jacobians.


## Implementations

All functions that have returns should return `np.ndarray`.

All public functions will have type hints.

All functions should have logging.logger for debugging.
