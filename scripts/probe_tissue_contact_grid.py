#!/usr/bin/env python3
"""Fast MuJoCo search for a physically valid L2 tissue grasp.

This probe intentionally uses the official scene, robot collision meshes,
actuator gains and free tissue body.  It is a dynamics/contact check rather
than an IK-only reachability check.  Candidate poses approach from above,
close the real tendon-driven fingers, then lift and pull.  Results are ranked
by dual-finger contact, retained object motion and shelf clearance.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUPERMARKET = ROOT / "examples" / "supermarket_sorting"
sys.path.insert(0, str(SUPERMARKET))

from manipulation.cartesian_path import interpolate_se3, plan_continuous_ik_path  # noqa: E402
from mmk2_kdl import MMK2Kdl  # noqa: E402


TOP_ROTATION = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float
)
OBJECT = "slot_D_L2_C1_zhijin"
FINGERS = ("rgt_finger_left_link", "rgt_finger_right_link")
SHELF_PREFIXES = ("shelf_", "rack_")


def pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def yawed_top_rotation(angle: float, local_roll: float = 0.0) -> np.ndarray:
    """Keep the shelf-frame grasp fixed while the chassis yaws by ``angle``."""
    c, s = math.cos(angle), math.sin(angle)
    inverse_chassis_yaw = np.array(
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
    )
    cr, sr = math.cos(local_roll), math.sin(local_roll)
    roll_y = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
    calibration = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    endpoint_local_roll = calibration @ roll_y @ calibration.T
    return inverse_chassis_yaw @ TOP_ROTATION @ endpoint_local_roll


class ContactProbe:
    def __init__(self) -> None:
        xml_path = SUPERMARKET / "mjcf" / "retail_competition.xml"
        xml = xml_path.read_text(encoding="utf-8").replace(
            "__REPO_ROOT__", str(SUPERMARKET)
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.kdl = MMK2Kdl()

        self.object_body = self.model.body(OBJECT).id
        self.object_joint = self.model.joint(f"{OBJECT}_freejoint")
        self.object_qadr = int(self.object_joint.qposadr[0])
        self.finger_bodies = tuple(self.model.body(name).id for name in FINGERS)
        self.slide_qadr = int(self.model.joint("slide_joint").qposadr[0])
        self.arm_qadrs = np.array(
            [int(self.model.joint(f"rgt_arm_joint{i}").qposadr[0]) for i in range(1, 7)]
        )
        self.arm_dadrs = np.array(
            [int(self.model.joint(f"rgt_arm_joint{i}").dofadr[0]) for i in range(1, 7)]
        )
        self.finger_qadrs = np.array(
            [
                int(self.model.joint("rgt_finger_left_joint").qposadr[0]),
                int(self.model.joint("rgt_finger_right_joint").qposadr[0]),
            ]
        )
        self.slide_act = self.model.actuator("lift").id
        self.arm_acts = np.array(
            [self.model.actuator(f"rgt_joint{i}").id for i in range(1, 7)]
        )
        self.gripper_act = self.model.actuator("rgt_gripper").id
        self.robot_joint = self.model.joint(self.model.body("mmk2").jntadr[0])
        self.robot_qadr = int(self.robot_joint.qposadr[0])
        self.robot_dadr = int(self.robot_joint.dofadr[0])
        self.robot_pose = np.array(
            [
                0.700,
                2.678,
                0.0,
                math.cos(math.pi / 4.0),
                0.0,
                0.0,
                math.sin(math.pi / 4.0),
            ]
        )
        self.base_lateral = 0.0
        self.base_yaw = 0.0

        shelf_ids = set()
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if name.startswith(SHELF_PREFIXES):
                shelf_ids.add(body_id)
        # The scene uses shelf bodies named retail_shelf_* as well.
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if "shelf" in name or "rack" in name:
                shelf_ids.add(int(self.model.geom_bodyid[geom_id]))
        self.shelf_bodies = shelf_ids
        self.pin_arm_target: np.ndarray | None = None
        self.reference_object_xy: np.ndarray | None = None
        self.max_dual_shift = 0.0
        self.blockers_seen: set[str] = set()

    def solve_path(
        self,
        points: list[np.ndarray],
        rotation: np.ndarray,
        slide: float,
        seed: np.ndarray,
        step: float = 0.012,
    ) -> tuple[np.ndarray, ...] | None:
        poses = []
        for start, goal in zip(points[:-1], points[1:]):
            segment = interpolate_se3(
                pose(start, rotation),
                pose(goal, rotation),
                translation_step=step,
                rotation_step_rad=0.08,
            )
            poses.extend(segment)

        def solve(candidate: np.ndarray, previous: np.ndarray):
            solutions = self.kdl.inverse_kinematics(
                T_right=candidate,
                ref_pos=np.concatenate(([slide], previous)),
                target_height=slide,
            )
            return [np.asarray(solution)[1:7] for solution in (solutions or ())]

        result = plan_continuous_ik_path(
            poses, solve, np.asarray(seed), max_joint_step=0.45
        )
        return result.joint_path if result.complete else None

    def joint_edge_free(
        self, start: np.ndarray, goal: np.ndarray, *, max_step: float = 0.045
    ) -> bool:
        segments = max(1, int(math.ceil(float(np.max(np.abs(goal - start))) / max_step)))
        for fraction in np.linspace(0.0, 1.0, segments + 1)[1:]:
            joints = (1.0 - fraction) * start + fraction * goal
            self.data.qpos[self.arm_qadrs] = joints
            self.data.qvel[self.arm_dadrs] = 0.0
            mujoco.mj_forward(self.model, self.data)
            if self.arm_environment_contacts():
                return False
        return True

    def plan_joint_rrt(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        iterations: int = 12000,
        seed: int = 97,
    ) -> tuple[np.ndarray, ...] | None:
        """Plan a collision-checked joint detour around adjacent products."""
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        self.data.qpos[self.arm_qadrs] = start
        mujoco.mj_forward(self.model, self.data)
        if self.arm_environment_contacts():
            return None
        self.data.qpos[self.arm_qadrs] = goal
        mujoco.mj_forward(self.model, self.data)
        if self.arm_environment_contacts():
            return None
        if self.joint_edge_free(start, goal):
            return (start.copy(), goal.copy())

        lower = np.array(
            [self.model.jnt_range[self.model.joint(f"rgt_arm_joint{i}").id, 0] for i in range(1, 7)]
        )
        upper = np.array(
            [self.model.jnt_range[self.model.joint(f"rgt_arm_joint{i}").id, 1] for i in range(1, 7)]
        )
        rng = np.random.default_rng(seed)
        nodes = [start.copy()]
        parents = [-1]
        step_size = 0.24
        for iteration in range(iterations):
            if iteration % 5 == 0:
                sample = goal
            else:
                sample = rng.uniform(lower, upper)
            matrix = np.asarray(nodes)
            nearest_index = int(np.argmin(np.linalg.norm(matrix - sample, axis=1)))
            nearest = nodes[nearest_index]
            delta = sample - nearest
            distance = float(np.linalg.norm(delta))
            candidate = sample.copy() if distance <= step_size else nearest + delta * (step_size / distance)
            if not self.joint_edge_free(nearest, candidate):
                continue
            nodes.append(candidate.copy())
            parents.append(nearest_index)
            new_index = len(nodes) - 1
            if float(np.linalg.norm(candidate - goal)) <= 0.34 and self.joint_edge_free(candidate, goal):
                nodes.append(goal.copy())
                parents.append(new_index)
                path = []
                cursor = len(nodes) - 1
                while cursor >= 0:
                    path.append(nodes[cursor])
                    cursor = parents[cursor]
                path.reverse()
                # Random collision-preserving shortcutting removes the
                # sampling zig-zags before actuator replay.
                for _ in range(160):
                    if len(path) <= 2:
                        break
                    left = int(rng.integers(0, len(path) - 2))
                    right = int(rng.integers(left + 2, len(path)))
                    if self.joint_edge_free(path[left], path[right]):
                        path = path[: left + 1] + path[right:]
                dense = [path[0].copy()]
                for edge_start, edge_goal in zip(path[:-1], path[1:]):
                    segments = max(
                        1,
                        int(
                            math.ceil(
                                float(np.max(np.abs(edge_goal - edge_start))) / 0.045
                            )
                        ),
                    )
                    dense.extend(
                        (1.0 - fraction) * edge_start + fraction * edge_goal
                        for fraction in np.linspace(1.0 / segments, 1.0, segments)
                    )
                return tuple(np.asarray(joints).copy() for joints in dense)
        return None

    def contacts(self) -> tuple[set[int], bool]:
        finger_hits: set[int] = set()
        shelf_hit = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            bodies = {
                int(self.model.geom_bodyid[contact.geom1]),
                int(self.model.geom_bodyid[contact.geom2]),
            }
            if self.object_body in bodies:
                for finger_index, body_id in enumerate(self.finger_bodies):
                    if body_id in bodies:
                        finger_hits.add(finger_index)
            if bodies & set(self.finger_bodies) and bodies & self.shelf_bodies:
                shelf_hit = True
            # Any arm link/shelf contact invalidates a nominally good grasp.
            if bodies & self.shelf_bodies:
                other = bodies - self.shelf_bodies
                for body_id in other:
                    name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                    ) or ""
                    if name.startswith("rgt_arm_"):
                        shelf_hit = True
        return finger_hits, shelf_hit

    def arm_environment_contacts(self) -> set[str]:
        """Report shelf or non-target product bodies touching the right arm."""
        blockers: set[str] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body_ids = (
                int(self.model.geom_bodyid[contact.geom1]),
                int(self.model.geom_bodyid[contact.geom2]),
            )
            names = tuple(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                for body_id in body_ids
            )
            arm_sides = [
                side
                for side, name in enumerate(names)
                if name.startswith("rgt_arm_") or name.startswith("rgt_finger_")
            ]
            for side in arm_sides:
                other = names[1 - side]
                if other == OBJECT:
                    continue
                if other.startswith("slot_") or "shelf" in other or "rack" in other:
                    blockers.add(other)
        return blockers

    def body_bounds(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a world AABB of collision geoms attached to one body."""
        points = []
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) != body_id:
                continue
            geom_type = int(self.model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh_id = int(self.model.geom_dataid[geom_id])
                first = int(self.model.mesh_vertadr[mesh_id])
                count = int(self.model.mesh_vertnum[mesh_id])
                local = np.asarray(self.model.mesh_vert[first : first + count])
                rotation = np.asarray(self.data.geom_xmat[geom_id]).reshape(3, 3)
                world = local @ rotation.T + np.asarray(self.data.geom_xpos[geom_id])
                points.extend(world)
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                size = np.asarray(self.model.geom_size[geom_id])
                corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) * size
                rotation = np.asarray(self.data.geom_xmat[geom_id]).reshape(3, 3)
                points.extend(corners @ rotation.T + np.asarray(self.data.geom_xpos[geom_id]))
        if not points:
            position = np.asarray(self.data.body(body_id).xpos).copy()
            return position, position
        array = np.asarray(points)
        return np.min(array, axis=0), np.max(array, axis=0)

    def step_for(self, seconds: float) -> tuple[set[int], bool]:
        finger_hits: set[int] = set()
        shelf_hit = False
        count = max(1, int(round(seconds / self.model.opt.timestep)))
        for _ in range(count):
            # The ROS server closes a chassis velocity loop while manipulating.
            # Raw MuJoCo has no such loop, so reaction torques otherwise let
            # the unbraked wheels rotate the base by tens of degrees and make
            # the contact sweep meaningless.  Hold only the free base pose;
            # arm, fingers, object and shelf remain fully dynamic.
            self.data.qpos[self.robot_qadr : self.robot_qadr + 7] = self.robot_pose
            self.data.qvel[self.robot_dadr : self.robot_dadr + 6] = 0.0
            if self.pin_arm_target is not None:
                self.data.qpos[self.arm_qadrs] = self.pin_arm_target
                self.data.qvel[self.arm_dadrs] = 0.0
            mujoco.mj_step(self.model, self.data)
            hits, collision = self.contacts()
            self.blockers_seen |= self.arm_environment_contacts()
            if len(hits) == 2 and self.reference_object_xy is not None:
                object_xy = np.asarray(self.data.body(self.object_body).xpos[:2])
                self.max_dual_shift = max(
                    self.max_dual_shift,
                    float(np.linalg.norm(object_xy - self.reference_object_xy)),
                )
            finger_hits |= hits
            shelf_hit |= collision
        return finger_hits, shelf_hit

    def command_path(
        self, joint_path: tuple[np.ndarray, ...], dwell: float = 0.30
    ) -> tuple[set[int], bool]:
        finger_hits: set[int] = set()
        shelf_hit = False
        for joints in joint_path:
            self.data.ctrl[self.arm_acts] = joints
            if os.getenv("PROBE_PIN_ARM", "1") == "1":
                self.pin_arm_target = np.asarray(joints).copy()
                self.data.qpos[self.arm_qadrs] = joints
                self.data.qvel[self.arm_dadrs] = 0.0
                mujoco.mj_forward(self.model, self.data)
                waypoint_dwell = min(
                    dwell, float(os.getenv("PROBE_PINNED_WAYPOINT_DWELL", "0.25"))
                )
                hits, collision = self.step_for(waypoint_dwell)
            else:
                # Mirror the client waypoint gate: do not stream a new target
                # while the physical arm still trails the current one.
                hits, collision = set(), False
                waited = 0.0
                while waited < 3.0:
                    step_hits, step_collision = self.step_for(0.05)
                    hits |= step_hits
                    collision |= step_collision
                    waited += 0.05
                    error = float(
                        np.max(np.abs(self.data.qpos[self.arm_qadrs] - joints))
                    )
                    if waited >= dwell and error < 0.025:
                        break
            finger_hits |= hits
            shelf_hit |= collision
        return finger_hits, shelf_hit

    def reset(self, slide: float, initial_joints: np.ndarray) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        # Put the chassis directly in front of the nominal D-L2-C1 object.
        self.robot_pose[0] = 0.700 + self.base_lateral
        yaw = math.pi / 2.0 + self.base_yaw
        self.robot_pose[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
        self.data.qpos[self.robot_qadr : self.robot_qadr + 7] = self.robot_pose
        self.data.qpos[self.slide_qadr] = slide
        self.data.qpos[self.arm_qadrs] = initial_joints
        self.data.qpos[self.finger_qadrs] = [-0.04, 0.04]
        self.data.ctrl[self.slide_act] = slide
        self.data.ctrl[self.arm_acts] = initial_joints
        self.data.ctrl[self.gripper_act] = 1.0
        self.pin_arm_target = initial_joints.copy() if os.getenv("PROBE_PIN_ARM", "1") == "1" else None
        mujoco.mj_forward(self.model, self.data)
        initial_object = np.asarray(self.data.body(self.object_body).xpos).copy()
        self.reference_object_xy = initial_object[:2].copy()
        self.max_dual_shift = 0.0
        self.blockers_seen = set()
        self.step_for(0.25)
        return initial_object

    def run_top(
        self,
        *,
        slide: float,
        endpoint_z: float,
        depth_bias: float,
        lateral_bias: float,
        yaw_deg: float,
        base_lateral: float = 0.0,
    ) -> dict[str, object] | None:
        chassis_yaw = math.radians(yaw_deg)
        local_roll = math.radians(float(os.getenv("PROBE_LOCAL_ROLL", "0.0")))
        rotation = yawed_top_rotation(chassis_yaw, local_roll)
        self.base_lateral = base_lateral
        self.base_yaw = chassis_yaw
        c, s = math.cos(chassis_yaw), math.sin(chassis_yaw)
        # Transform the fixed world shelf-depth and lateral axes into the
        # yawed footprint.  The world object remains at [0.700, 3.243].
        centre = np.array(
            [
                c * (0.565 + depth_bias) - s * lateral_bias,
                -s * (0.565 + depth_bias) - c * lateral_bias + base_lateral,
                0.895,
            ]
        )
        approach_clearance = float(os.getenv("PROBE_APPROACH_CLEARANCE", "0.10"))
        retract_vector = np.array([-0.16 * c, 0.16 * s, 0.0])
        pre = centre + retract_vector + np.array(
            [0.0, 0.0, endpoint_z + approach_clearance]
        )
        above = centre + np.array([0.0, 0.0, endpoint_z + approach_clearance])
        grasp = centre + np.array([0.0, 0.0, endpoint_z])
        if os.getenv("PROBE_LOW_INSERT", "0") == "1":
            pre = centre + retract_vector + np.array([0.0, 0.0, endpoint_z])
            route_points = [grasp, pre]
        else:
            route_points = [grasp, above, pre]
        stow = np.array([0.0, -0.166, 0.032, 0.0, -1.571, -2.223])
        if os.getenv("PROBE_STABLE_BRANCH", "1") == "1" and abs(slide - 0.5) < 1e-6:
            stable_reference = np.array(
                [0.509, -1.238, 1.496, 1.182, -1.140, 0.545]
            )
            references = [stable_reference, stow, -stable_reference]
            rng = np.random.default_rng(73)
            references.extend(rng.uniform(-2.6, 2.6, (20, 6)))
            grasp_candidates = []
            for reference in references:
                solutions = self.kdl.inverse_kinematics(
                    T_right=pose(grasp, rotation),
                    ref_pos=np.concatenate(([slide], reference)),
                    target_height=slide,
                )
                for solution in solutions or ():
                    joints = np.asarray(solution)[1:7]
                    if not any(
                        np.max(np.abs(joints - known)) < 0.03
                        for known in grasp_candidates
                    ):
                        grasp_candidates.append(joints)
            selected = None
            for grasp_joints in sorted(
                grasp_candidates,
                key=lambda joints: float(np.linalg.norm(joints - stable_reference)),
            ):
                reverse_path = self.solve_path(
                    route_points, rotation, slide, grasp_joints
                )
                if not reverse_path:
                    continue
                candidate = tuple(reversed(reverse_path)) + (grasp_joints,)
                # Collision-check the whole branch in the exact official
                # scene.  IK continuity alone cannot see a link4 sweep through
                # the neighbouring product in the next 220-mm shelf column.
                self.reset(slide, candidate[0])
                blockers: set[str] = set()
                for joints in candidate:
                    self.data.qpos[self.arm_qadrs] = joints
                    self.data.qvel[self.arm_dadrs] = 0.0
                    mujoco.mj_forward(self.model, self.data)
                    blockers |= self.arm_environment_contacts()
                    if blockers:
                        break
                if not blockers:
                    selected = candidate
                    break
                # Both endpoint configurations are safe but this Cartesian
                # branch sweeps link4 through the next shelf column.  Search a
                # bounded configuration-space detour with the same official
                # MuJoCo collision model.
                detour = self.plan_joint_rrt(
                    candidate[0], grasp_joints, seed=97 + len(grasp_candidates)
                )
                if os.getenv("PROBE_DEBUG", "0") == "1":
                    self.data.qpos[self.arm_qadrs] = candidate[0]
                    mujoco.mj_forward(self.model, self.data)
                    start_blockers = sorted(self.arm_environment_contacts())
                    self.data.qpos[self.arm_qadrs] = grasp_joints
                    mujoco.mj_forward(self.model, self.data)
                    goal_blockers = sorted(self.arm_environment_contacts())
                    print(
                        "DEBUG RRT "
                        f"start={np.round(candidate[0], 3)} blockers={start_blockers} "
                        f"goal={np.round(grasp_joints, 3)} blockers={goal_blockers} "
                        f"found={detour is not None}",
                        flush=True,
                    )
                if detour is not None:
                    selected = detour
                    break
            if selected is None:
                return None
            approach = selected
            initial = approach[0]
        else:
            initial_solutions = self.kdl.inverse_kinematics(
                T_right=pose(pre, rotation),
                ref_pos=np.concatenate(([slide], stow)),
                target_height=slide,
            )
            if not initial_solutions:
                return None
            initial = np.asarray(initial_solutions[0])[1:7]
            forward_points = list(reversed(route_points))
            approach = self.solve_path(
                forward_points, rotation, slide, initial
            )
            if not approach:
                return None
        # The official score is based on horizontal displacement.  Keep the
        # box supported by L2 during the initial extraction; attempting to
        # lift first asks the 1-N gripper actuator to suspend the whole box by
        # friction and releases an otherwise stable dual contact.
        pulled = grasp + np.array([-0.35 * c, 0.35 * s, 0.0])
        extraction = self.solve_path(
            [grasp, pulled], rotation, slide, approach[-1]
        )
        if not extraction:
            return None

        initial_object = self.reset(slide, initial)
        approach_hits, shelf_hit = self.command_path(approach)
        preclose_object = np.asarray(self.data.body(self.object_body).xpos).copy()
        if os.getenv("PROBE_DEBUG", "0") == "1":
            print(
                "DEBUG pose "
                f"params={(slide, endpoint_z, depth_bias, lateral_bias, yaw_deg, base_lateral)} "
                f"base_site={np.round(self.data.site('base_link').xpos, 4)} "
                f"base_xmat={np.round(np.asarray(self.data.site('base_link').xmat).reshape(3, 3), 3)} "
                f"link={np.round(self.data.body('rgt_arm_link6').xpos, 4)} "
                f"object={np.round(preclose_object, 4)} "
                f"q_start={np.round(approach[0], 4)} "
                f"q_target={np.round(approach[-1], 4)} "
                f"q_actual={np.round(self.data.qpos[self.arm_qadrs], 4)}",
                flush=True,
            )
            for name, body_id in zip(FINGERS, self.finger_bodies):
                lower, upper = self.body_bounds(body_id)
                print(
                    f"DEBUG {name} body={np.round(self.data.body(body_id).xpos, 4)} "
                    f"bounds={np.round(lower, 4)}..{np.round(upper, 4)}",
                    flush=True,
                )
            pairs = []
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                geom_names = []
                for geom_id in (contact.geom1, contact.geom2):
                    geom_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    body_name = mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(self.model.geom_bodyid[geom_id]),
                    )
                    geom_names.append(geom_name or body_name or str(geom_id))
                if any("rgt_" in value for value in geom_names):
                    pairs.append(tuple(geom_names))
            print(f"DEBUG contacts={pairs[:20]}", flush=True)
        self.data.ctrl[self.gripper_act] = 0.0
        close_hits, close_shelf = self.step_for(4.5)
        close_end_hits, _ = self.contacts()
        closed_gripper = float(self.data.qpos[self.finger_qadrs[1]] * 25.0)
        closed_object = np.asarray(self.data.body(self.object_body).xpos).copy()
        extraction_hits, extract_shelf = self.command_path(extraction)
        final_hits, _ = self.contacts()
        final_object = np.asarray(self.data.body(self.object_body).xpos).copy()
        all_hits = approach_hits | close_hits | extraction_hits
        moved_before_grip = float(np.linalg.norm(preclose_object[:2] - initial_object[:2]))
        extracted_xy = float(np.linalg.norm(final_object[:2] - initial_object[:2]))
        lifted = float(final_object[2] - initial_object[2])
        return {
            "dual": len(close_end_hits) == 2,
            "retained": len(final_hits) == 2,
            "ever_dual": len(all_hits) == 2,
            "shelf": shelf_hit or close_shelf or extract_shelf,
            "blockers": sorted(self.blockers_seen),
            "pre_shift": moved_before_grip,
            "extract_xy": extracted_xy,
            "max_dual_shift": self.max_dual_shift,
            "lifted": lifted,
            "closed_gripper": closed_gripper,
            "gripper": float(self.data.qpos[self.finger_qadrs[1]] * 25.0),
            "final": final_object,
        }

    def equilibrium_error(
        self, *, slide: float, endpoint_z: float, depth_bias: float
    ) -> dict[str, object] | None:
        """Measure gravity sag from a statically exact top-pinch IK pose."""
        rotation = yawed_top_rotation(0.0)
        target = np.array([0.565 + depth_bias, 0.0, 0.895 + endpoint_z])
        references = [
            np.array([0.0, -0.166, 0.032, 0.0, -1.571, -2.223]),
            np.array([0.0, -1.0, 1.0, 0.0, -1.0, 0.0]),
            np.array([0.0, 0.5, -1.0, 0.0, 1.0, 0.0]),
        ]
        rng = np.random.default_rng(41)
        references.extend(rng.uniform(-2.6, 2.6, (24, 6)))
        candidates = []
        for reference in references:
            solutions = self.kdl.inverse_kinematics(
                T_right=pose(target, rotation),
                ref_pos=np.concatenate(([slide], reference)),
                target_height=slide,
            )
            for solution in solutions or ():
                joints = np.asarray(solution)[1:7]
                if not any(np.max(np.abs(joints - known)) < 0.03 for known in candidates):
                    candidates.append(joints)
        if not candidates:
            return None
        best = None
        for joints in candidates:
            self.pin_arm_target = None
            self.reset(slide, joints)
            self.pin_arm_target = None
            self.data.ctrl[self.arm_acts] = joints
            static_blockers = self.arm_environment_contacts()
            if static_blockers:
                continue
            _, shelf_hit = self.step_for(8.0)
            dynamic_blockers = self.arm_environment_contacts()
            base_p = np.asarray(self.data.site("base_link").xpos)
            base_R = np.asarray(self.data.site("base_link").xmat).reshape(3, 3)
            actual = base_R.T @ (
                np.asarray(self.data.site("rgt_endpoint").xpos) - base_p
            )
            q_error = float(np.max(np.abs(self.data.qpos[self.arm_qadrs] - joints)))
            endpoint_error = float(np.linalg.norm(actual - target))
            result = {
                "endpoint_error": endpoint_error,
                "q_error": q_error,
                "shelf": shelf_hit,
                "blockers": sorted(dynamic_blockers),
                "actual": actual.copy(),
                "joints": joints.copy(),
            }
            if dynamic_blockers:
                continue
            if best is None or endpoint_error < float(best["endpoint_error"]):
                best = result
        if best is not None:
            best["branches"] = len(candidates)
        return best


def main() -> None:
    probe = ContactProbe()
    if os.getenv("PROBE_EQUILIBRIUM_ONLY", "0") == "1":
        slides = tuple(
            float(value)
            for value in os.getenv("PROBE_SLIDES", "0.10,0.20,0.30,0.40,0.50,0.60").split(",")
        )
        endpoint_z = float(os.getenv("PROBE_HEIGHTS", "0.060").split(",")[0])
        depth_bias = float(os.getenv("PROBE_DEPTHS", "-0.004").split(",")[0])
        for slide in slides:
            result = probe.equilibrium_error(
                slide=slide, endpoint_z=endpoint_z, depth_bias=depth_bias
            )
            print(f"equilibrium slide={slide:.3f} result={result}", flush=True)
        return
    candidates = []
    total = 0
    if os.getenv("PROBE_FULL_GRID", "0") == "1":
        grid = itertools.product(
            (0.30, 0.40, 0.50),
            (0.055, 0.065, 0.075, 0.085, 0.095),
            (-0.010, 0.0, 0.010),
            (-0.006, 0.0, 0.006),
            (-6.0, -3.0, 0.0, 3.0, 6.0),
        )
    else:
        # Establish the physically useful vertical window first.  The larger
        # lateral/depth/cant sweep is only valuable around a contact-positive
        # height and is enabled with PROBE_FULL_GRID=1.
        coarse_heights = tuple(
            float(value)
            for value in os.getenv(
                "PROBE_HEIGHTS", "0.045,0.055,0.065,0.075,0.085,0.095,0.105"
            ).split(",")
        )
        coarse_slides = tuple(
            float(value) for value in os.getenv("PROBE_SLIDES", "0.50").split(",")
        )
        coarse_depths = tuple(
            float(value) for value in os.getenv("PROBE_DEPTHS", "0.0").split(",")
        )
        coarse_laterals = tuple(
            float(value) for value in os.getenv("PROBE_LATERALS", "0.0").split(",")
        )
        coarse_yaws = tuple(
            float(value) for value in os.getenv("PROBE_YAWS", "0.0").split(",")
        )
        coarse_base_laterals = tuple(
            float(value)
            for value in os.getenv("PROBE_BASE_LATERALS", "0.0").split(",")
        )
        grid = itertools.product(
            coarse_slides,
            coarse_heights,
            coarse_depths,
            coarse_laterals,
            coarse_yaws,
            coarse_base_laterals,
        )
    for slide, endpoint_z, depth_bias, lateral_bias, yaw_deg, base_lateral in grid:
        total += 1
        result = probe.run_top(
            slide=slide,
            endpoint_z=endpoint_z,
            depth_bias=depth_bias,
            lateral_bias=lateral_bias,
            yaw_deg=yaw_deg,
            base_lateral=base_lateral,
        )
        if result is None:
            continue
        score = (
            120.0 * float(result["retained"])
            + 50.0 * float(result["dual"])
            + 20.0 * float(result["ever_dual"])
            - 100.0 * float(result["shelf"])
            - 500.0 * max(0.0, float(result["pre_shift"]) - 0.045)
            + 40.0 * float(result["extract_xy"])
            + 40.0 * max(0.0, float(result["lifted"]))
        )
        candidates.append(
            (
                score,
                slide,
                endpoint_z,
                depth_bias,
                lateral_bias,
                yaw_deg,
                base_lateral,
                result,
            )
        )
        if result["retained"] and not result["shelf"]:
            print(
                "VALID "
                f"slide={slide:.2f} z={endpoint_z:.3f} depth={depth_bias:+.3f} "
                f"lat={lateral_bias:+.3f} yaw={yaw_deg:+.1f} result={result}",
                flush=True,
            )

    print(f"evaluated={total} dynamically_simulated={len(candidates)}")
    for item in sorted(candidates, key=lambda value: value[0], reverse=True)[:30]:
        score, slide, endpoint_z, depth_bias, lateral_bias, yaw_deg, base_lateral, result = item
        print(
            f"score={score:7.2f} slide={slide:.2f} z={endpoint_z:.3f} "
            f"depth={depth_bias:+.3f} lat={lateral_bias:+.3f} yaw={yaw_deg:+.1f} "
            f"base_lat={base_lateral:+.3f} "
            f"result={result}"
        )


if __name__ == "__main__":
    main()
