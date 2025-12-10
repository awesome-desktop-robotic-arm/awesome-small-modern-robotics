# Lib ASMR - Awesome Small Modern Robotics Library
A utility collection for robotics development

>[!NOTE]
> This is a library under development - hang on for more updates!

## Modules

- Utils
    - [Robot dataclass definition](utils/robot_class.py)
    - [Robot parser](utils/model_loader.py)
    - [Robot visualizer](utils/visualizer.py) 
    - [Geometric helpers](utils/geometry.py)
- Kinematics:    
    - [x] [Kinematics module](asmr/kinematics.py)
        Stateless kinematics functions that relies on `Robot` class
        - [x] FK
        - [x] IK
        - [x] Analytical jacobian

    - [ ] [Dynamics module](asmr/dynamics.py)
        Stateless dynamics functions that relies on `Robot` class
        - [x] ID
        - [ ] FD
            - [ ] FD with Featherstone's ADA