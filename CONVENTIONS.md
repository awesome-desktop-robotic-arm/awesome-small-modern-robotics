# ASMR CONVENTIONS

## Symbols

q = [joint angles] in radians
qd = [joint velocities]
qdd = [joint accelerations]

x = [ee position] in meters
xd = [ee velocities]
xdd = [ee accelerations]

p = [translations in 3-vector]
R = [Rotations in SO(3)]
T = [Transformations in SE(3)]

F = [wrench in 6-vector]
tau = [joint torques]

## 

## Implementations

All functions that have returns should return `np.ndarray`.

All public functions will have type hints.

All functions should have logging.logger for debugging.
