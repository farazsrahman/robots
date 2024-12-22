print("Hello, World!")
print('Importing Genesis...')
from genesis.ext import pyrender, trimesh
from genesis.ext.pyrender.jit_render import JITRenderer
import genesis as gs
import argparse
import numpy as np

print('\nGenesis imported successfully')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -10.0),
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file  = 'xml/franka_emika_panda/panda.xml',
            pos   = (0.0, 0.0, 0.0),
            euler = (0, 0, 0),
        ),
    )

    jnt_names = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7',
        'finger_joint1',
        'finger_joint2',
    ]
    dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]


    ########################## build ##########################
    scene.build()

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis, franka, dofs_idx))
    if args.vis:
        scene.viewer.start()


def run_sim(scene, enable_vis, franka, dofs_idx):
    from time import time

    t_prev = time()
    i = 0
    while True:
        i += 1

        if i == 0:
            franka.control_dofs_position(
                np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
                dofs_idx,
            )
        elif i == 250:
            franka.control_dofs_position(
                np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
                dofs_idx,
            )
        elif i == 500:
            franka.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                dofs_idx,
            )
        elif i == 750:
            # control first dof with velocity, and the rest with position
            franka.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
                dofs_idx[1:],
            )
            franka.control_dofs_velocity(
                np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
                dofs_idx[:1],
            )
        elif i == 1000:
            franka.control_dofs_force(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                dofs_idx,
            )
        # This is the control force computed based on the given control command
        # If using force control, it's the same as the given control command
        print('control force:', franka.get_dofs_control_force(dofs_idx))

        # This is the actual force experienced by the dof
        print('internal force:', franka.get_dofs_force(dofs_idx))
        scene.step()

        t_now = time()
        print(1 / (t_now - t_prev), "FPS")
        t_prev = t_now

        if i > 1000:
            break

    if enable_vis:
        scene.viewer.stop()


if __name__ == "__main__":
    main()