# -*- coding: utf-8 -*-
# Minimal offscreen render test inside the server image
import os, sys
os.environ.setdefault('MUJOCO_GL', 'glfw')
try:
    import mujoco
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="box" size="0.1 0.1 0.1" pos="0 0 0.1"/></worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)
    print('mujoco step OK, sim time =', round(d.time, 3))
    print('mj_version =', mujoco.__version__)
    # offscreen render test (EGL fallback)
    try:
        renderer = mujoco.Renderer(m, 64, 64)
        img = renderer.render()
        print('renderer OK, img shape =', img.shape)
        renderer.close()
    except Exception as e:
        print('renderer FAILED:', e)
except Exception as e:
    print('mujoco FAILED:', e)
    sys.exit(1)
