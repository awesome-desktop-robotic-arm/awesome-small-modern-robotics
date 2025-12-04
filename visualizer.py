"""
Mujoco visualizer
"""

import mujoco
import mujoco.viewer
from pathlib import Path

MODEL_DIR = Path("mujoco_menagerie/agilex_piper/scene.xml")

if __name__ =="__main__":
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_DIR}")
    
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        while viewer.is_running():
        
            mujoco.mj_step(model, data)

            viewer.sync()


