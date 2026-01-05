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


## Transform Notation

### Standard Convention: T_A_B (or X_A_B)

A homogeneous transform `T_A_B` (4×4 matrix) maps coordinates from frame B to frame A:

```
p_A = T_A_B @ p_B
```

Where:
- `p_A` is a point expressed in frame A coordinates
- `p_B` is the same point expressed in frame B coordinates
- `T_A_B` transforms from B → A

### Structure of a Homogeneous Transform

```
T_A_B = [R_A_B | t_A_B]
        [  0   |   1  ]
```

Where:
- `R_A_B` (3×3): Rotation matrix that maps vectors from frame B to frame A
  - `v_A = R_A_B @ v_B`
- `t_A_B` (3×1): Translation vector representing the origin of frame B expressed in frame A coordinates

### Inverse Transform

```
T_B_A = (T_A_B)^(-1)
```

Maps coordinates from frame A to frame B:
- `p_B = T_B_A @ p_A`
- `R_B_A = R_A_B^T` (rotation matrices are orthogonal)

## MJCF/Robot Model Convention

### Link Origins

In MJCF files like:
```xml
<body name="fr3_link1" pos="0 0 0.333">
```

The `pos` attribute represents the **child origin expressed in the parent frame**.

### Storage Convention in Robot Class

For a link in the robot tree:
- `link.T_origin` = **T_parent_child** (maps child → parent)
  - `link.T_origin[:3, :3]` = `R_parent_child` (rotation: child → parent)
  - `link.T_origin[:3, 3]` = translation from parent origin to child origin, expressed in parent frame
  
- `link.T_origin_inv` = **T_child_parent** (maps parent → child)
  - `link.T_origin_inv[:3, :3]` = `R_child_parent` (rotation: parent → child)
  - This is the inverse of `link.T_origin`
