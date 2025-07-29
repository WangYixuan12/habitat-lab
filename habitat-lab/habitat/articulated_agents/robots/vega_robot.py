# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np

from habitat.articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams,
    MobileManipulator,
    MobileManipulatorParams,
)


class VegaRobot(MobileManipulator):
    def _get_vega_params(self):
        return MobileManipulatorParams(
            arm_joints=list(range(19, 26)) + list(range(27, 34)),
            gripper_joints=[36, 37],
            leg_joints=[1],
            arm_init_params=np.array([np.pi*2.0/3.0, 0.0, 0.0, -np.pi*2.0/3.0, 0.0, 0.0, 0.0, -np.pi*2.0/3.0, 0.0, 0.0, -np.pi*2.0/3.0, 0.0, 0.0, 0.0]), # [L_j1, L_j2, L_j3, L_j4, L_j5, L_j6, L_j7, R_j1, R_j2, R_j3, R_j4, R_j5, R_j6, R_j7]
            gripper_init_params=np.zeros(2),
            leg_init_params=np.zeros(1),
            ee_offset=[mn.Vector3(0.0, 0.0, 0.0)],
            ee_links=[34, 26],
            ee_constraint=np.array(
                [[[-0.08, 0.29], [-0.84, -0.27], [0.01, 1.12]]]
            ),
            cameras={
                "head": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.0, 0.0, 0.1),
                    cam_orientation=mn.Vector3(np.pi / 2.0, 0.0, 3.0 * np.pi / 2.0),
                    attached_link_id=37,
                ),
                "third": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(-1.0, -1.0, 1.0),
                    cam_orientation=mn.Vector3(np.pi / 4.0, 0.0, 3.0 * np.pi / 2.0),
                    attached_link_id=37,
                ),
            },
            gripper_closed_state=np.array([0.0], dtype=np.float32),
            gripper_open_state=np.array([-1.56], dtype=np.float32),
            gripper_state_eps=0.01,
            arm_mtr_pos_gain=0.3,
            arm_mtr_vel_gain=0.3,
            arm_mtr_max_impulse=10.0,
            leg_mtr_pos_gain=2.0,
            leg_mtr_vel_gain=1.3,
            leg_mtr_max_impulse=100.0,
            base_offset=mn.Vector3(0.0, 0.0, 0.0),
            # base_rot_offset=mn.Quaternion.rotation(mn.Rad(3.0 * np.pi / 2.0), mn.Vector3(1, 0, 0)),
            base_rot_offset=mn.Quaternion.from_matrix(
                np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                ])
            ),
            base_link_names={
                "base_link",
            },
            navmesh_offsets=[[0.0, 0.0], [0.25, 0.0], [-0.25, 0.0]],
        )

    @property
    def base_transformation(self):
        add_rot = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0)
        )
        return self.sim_obj.transformation @ add_rot

    def __init__(
        self, agent_cfg, sim, limit_robo_joints=True, fixed_base=True
    ):
        super().__init__(
            self._get_vega_params(),
            agent_cfg,
            sim,
            limit_robo_joints,
            fixed_base,
            base_type="leg",
        )
