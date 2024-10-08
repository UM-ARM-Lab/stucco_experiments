import copy
import logging
import os.path
import typing

import pybullet as p
import pybullet_data
import torch
import rospy
import ros_numpy
import time
import math
from threading import Lock

import numpy as np

import stucco_experiments.baselines.hard_tracking
from pytorch_kinematics import transforms as tf
from base_experiments.util import move_figure

from mmint_camera_utils.camera_utils.camera_utils import bilinear_interpolate, project_depth_points
from stucco import tracking
from stucco_experiments.detection_impl import PybulletResidualPlanarContactSensor
from stucco.detection import ContactSensor
from base_experiments import cfg
from stucco.movable_sdf import PlanarMovableSDF
from base_experiments.env.env import InfoKeys, TrajectoryLoader, Env, handle_data_format_for_state_diff, EnvDataSource

from stucco.detection import ContactDetector
from geometry_msgs.msg import Pose

from base_experiments.env.pybullet_env import closest_point_on_surface, ContactInfo, DebugDrawer, \
    state_action_color_pairs
from base_experiments.env.real_env import CombinedVisualizer

from victor_hardware_interface_msgs.msg import ControlMode, MotionStatus
from tf2_geometry_msgs import WrenchStamped
from arm_robots.victor import Victor
from bubble_utils.bubble_med.bubble_med import BubbleMed
from bubble_utils.bubble_parsers.bubble_parser import BubbleParser
from bubble_tools.bubble_tools.bubble_img_tools import process_bubble_img
from wsg_50_utils.wsg_50_gripper import WSG50Gripper

# runner imports
from arm_pytorch_utilities import tensor_utils
import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

DIR = "arm_real"


class ArmRealLoader(TrajectoryLoader):
    @staticmethod
    def _info_names():
        return []

    def _process_file_raw_data(self, d):
        x = d['X']
        # extract the states

        if self.config.predict_difference:
            y = RealArmEnv.state_difference(x[1:], x[:-1])
        else:
            raise RuntimeError("Too hard to predict discontinuous normalized angles; use predict difference")

        xu, y, cc = self._apply_masks(d, x, y)

        return xu, y, cc


def pose_msg_to_pos_quaternion(pm: Pose):
    pos = [pm.position.x, pm.position.y, pm.position.z]
    orientation = [pm.orientation.x, pm.orientation.y, pm.orientation.z, pm.orientation.w]
    return pos, orientation


def xy_pose_distance(a: Pose, b: Pose):
    pos_distance = np.linalg.norm(ros_numpy.numpify(a.position)[:2] - ros_numpy.numpify(b.position)[:2])
    return pos_distance


class RealArmEnv(Env):
    """Interaction with robot via our ROS node; manages what dimensions of the returned observation
    is necessary for dynamics (passed back as state) and which are extra (passed back as info)"""
    nu = 2
    nx = 2

    MAX_PUSH_DIST = 0.02
    RESET_RAISE_BY = 0.025

    # REST_POS = [0.7841804139585614, -0.34821761121288775, 0.9786928519851419]
    REST_POS = [0.7841804139585614, -0.34821761121288775, 0.973]
    REST_ORIENTATION = [-np.pi / 2, np.pi - np.pi / 4, 0]
    # REST_ORIENTATION_QUAT = [-0.7068252, 0, 0, 0.7073883]
    BASE_POSE = ([-0.02, -0.1384885, 1.248],
                 [0.6532814824398555, 0.27059805007378895, 0.270598050072408, 0.6532814824365213])
    # REST_JOINTS = [1.4741407366569612, -0.37367112858393897, 1.420287584732473, 1.3502012009206756, -0.104535997082166,
    #                -0.9739424384610837, -1.177102973890961]
    REST_JOINTS = [1.4769502583855774, -0.3829121045031008, 1.4203813612731826, 1.3409767633051735, -0.1265017390739062,
                   -0.9736777669158343, -1.1727687591440739 + np.pi]
    # REST_JOINTS = [1.0269783861054675, -0.400866330952976, 1.8832600640176749, 1.176414060692023, -0.12416865012501638,
    #                -1.330497996717088, -1.1644768139687487]

    EE_LINK_NAME = "victor_right_arm_link_7"
    WORLD_FRAME = "victor_root"
    ACTIVE_MOVING_ARM = 1  # right

    @staticmethod
    def state_names():
        return ['x ee (m)', 'y ee (m)']

    @staticmethod
    def get_ee_pos(state):
        return state[:2]

    @staticmethod
    @tensor_utils.ensure_2d_input
    def get_ee_pos_states(states):
        return states[:, :2]

    @classmethod
    @handle_data_format_for_state_diff
    def state_difference(cls, state, other_state):
        """Get state - other_state in state space"""
        dpos = state[:, :2] - other_state[:, :2]
        return dpos,

    @classmethod
    def state_cost(cls):
        return np.diag([1, 1])

    @classmethod
    def state_distance(cls, state_difference):
        return state_difference[:, :2].norm(dim=1)

    @staticmethod
    def control_names():
        return ['d$x_r$', 'd$y_r$']

    @staticmethod
    def get_control_bounds():
        u_min = np.array([-1, -1])
        u_max = np.array([1, 1])
        return u_min, u_max

    @classmethod
    @handle_data_format_for_state_diff
    def control_similarity(cls, u1, u2):
        return torch.cosine_similarity(u1, u2, dim=-1).clamp(0, 1)

    @classmethod
    def control_cost(cls):
        return np.diag([1 for _ in range(cls.nu)])

    def __init__(self, environment_level=0, dist_for_done=0.015, obs_time=1, stub=False, residual_threshold=10.,
                 residual_precision=None, vel=0.2):
        self.level = environment_level
        self.dist_for_done = dist_for_done
        self.static_wrench = None
        self.obs_time = obs_time
        self._contact_detector = None
        self._vis = CombinedVisualizer()
        self._contact_debug_names = []
        # for Victor, have to force orientation to be planar in several places
        self._need_to_force_planar = True
        self.motion_frame_transformer = None

        # additional information accumulated in a single step
        self._single_step_contact_info = {}
        self.last_ee_pos = None
        self.vel = vel

        if not stub:
            self._setup_robot_ros(residual_threshold=residual_threshold, residual_precision=residual_precision)

    def _clear_state_before_step(self):
        self.contact_detector.clear_sensors()

    def _setup_robot_ros(self, residual_threshold=10., residual_precision=None):
        victor = Victor(force_trigger=5.0)
        self.robot = victor
        # adjust timeout according to velocity (at vel = 0.1 we expect 400s per 1m)
        self.robot.cartesian._timeout_per_m = 40 / self.vel
        victor.connect()
        self.vis.init_ros()

        self.motion_status_input_lock = Lock()
        self._temp_wrenches = []
        # subscribe to status messages
        self.right_arm_contact_listener = rospy.Subscriber(victor.ns("right_arm/motion_status"), MotionStatus,
                                                           self.contact_listener)
        self.cleaned_wrench_publisher = rospy.Publisher(victor.ns("right_gripper/cleaned_wrench"), WrenchStamped,
                                                        queue_size=10)
        self.large_wrench_publisher = rospy.Publisher(victor.ns("right_gripper/large_wrench"), WrenchStamped,
                                                      queue_size=10)
        self.large_wrench_world_publisher = rospy.Publisher(victor.ns("right_gripper/large_wrench_world"),
                                                            WrenchStamped, queue_size=10)

        # to reset the rest pose, manually jog it there then read the values
        # rest_pose = pose_msg_to_pos_quaternion(victor.get_link_pose(self.EE_LINK_NAME))

        # reset to rest position
        self.return_to_rest(self.robot.right_arm_group)
        self.robot.close_right_gripper()

        base_pose = pose_msg_to_pos_quaternion(victor.get_link_pose('victor_right_arm_mount'))
        status = victor.get_right_arm_status()
        canonical_joints = [status.measured_joint_position.joint_1, status.measured_joint_position.joint_2,
                            status.measured_joint_position.joint_3, status.measured_joint_position.joint_4,
                            status.measured_joint_position.joint_5, status.measured_joint_position.joint_6,
                            status.measured_joint_position.joint_7]

        self.last_ee_pos = self._observe_ee(return_z=True)
        self.state = self._obs()

        if residual_precision is None:
            residual_precision = np.diag([1, 1, 0, 1, 1, 1])
        # parallel visualizer for ROS and pybullet
        self._contact_detector = ContactDetector(residual_precision, window_size=50)
        self._contact_detector.register_contact_sensor(KukaResidualContactSensor("victor", residual_threshold,
                                                                                 base_pose=base_pose,
                                                                                 default_joint_config=canonical_joints,
                                                                                 canonical_pos=self.REST_POS,
                                                                                 canonical_orientation=self.REST_ORIENTATION,
                                                                                 visualizer=self.vis))
        self.vis.sim = None

    @property
    def vis(self) -> CombinedVisualizer:
        return self._vis

    def enter_cartesian_mode(self):
        self.robot.set_control_mode(control_mode=ControlMode.CARTESIAN_IMPEDANCE, vel=self.vel)

    def return_to_rest(self, group_name):
        self.robot.set_control_mode(control_mode=ControlMode.JOINT_POSITION, vel=self.vel)
        # self.robot.plan_to_pose(group_name, self.EE_LINK_NAME, self.REST_POS + self.REST_ORIENTATION)
        self.robot.plan_to_joint_config(group_name, self.REST_JOINTS)
        self.enter_cartesian_mode()

    def recalibrate_static_wrench(self):
        start = time.time()
        self.static_wrench = None
        self._temp_wrenches = []
        # collect them in the frame we detect them
        while len(self._temp_wrenches) < 10:
            rospy.sleep(0.1)

        self.static_wrench = np.mean(self._temp_wrenches, axis=0)
        wrench_var = np.var(self._temp_wrenches, axis=0)
        print(f"calibrate static wrench elapsed {time.time() - start} {self.static_wrench} {wrench_var}")
        self._contact_detector.clear()

    @property
    def contact_detector(self) -> ContactDetector:
        return self._contact_detector

    def set_task_config(self, goal=None, init=None):
        """Change task configuration; assumes only goal position is specified"""
        if goal is not None:
            if len(goal) != 2:
                raise RuntimeError("Expected hole to be (x, y), instead got {}".format(goal))
            self.goal = goal
        if init is not None:
            if len(init) != 2:
                raise RuntimeError("Expected peg to be (x, y), instead got {}".format(init))
            self.init = init

    # ros methods
    def contact_listener(self, status: MotionStatus):
        if self.contact_detector is None:
            return
        with self.motion_status_input_lock:
            w = status.estimated_external_wrench
            # convert wrench to world frame
            wr = WrenchStamped()
            wr.header.frame_id = self.EE_LINK_NAME
            wr.wrench.force.x = w.x
            wr.wrench.force.y = w.y
            wr.wrench.force.z = w.z
            wr.wrench.torque.x = w.a
            wr.wrench.torque.y = w.b
            wr.wrench.torque.z = w.c

        if self.motion_frame_transformer is not None:
            wr = self.motion_frame_transformer.motion_status_to_ee_wrench(wr)
            wr.header.frame_id = self.EE_LINK_NAME
        wr_world = self.robot.tf_wrapper.transform_to_frame(wr, self.WORLD_FRAME)
        wr = wr_world.wrench

        # clean with static wrench
        wr_np = np.array([wr.force.x, wr.force.y, wr.force.z, wr.torque.x, wr.torque.y, wr.torque.z])
        if self.static_wrench is None:
            self._temp_wrenches.append(wr_np)
            return
        wr_np -= self.static_wrench

        # wr_np = self._fix_torque_to_planar(wr_np)

        # visualization
        wr = WrenchStamped()
        wr.header.frame_id = self.WORLD_FRAME
        wr.wrench.force.x, wr.wrench.force.y, wr.wrench.force.z, wr.wrench.torque.x, wr.wrench.torque.y, wr.wrench.torque.z = wr_np
        # self.cleaned_wrench_publisher.publish(wr)

        # print residual
        residual = wr_np.T @ self.contact_detector.residual_precision @ wr_np
        self.vis.ros.draw_text("residualmag", f"{np.round(residual, 2)}", [0.2, 0.5, 0.5], absolute_pos=True)

        # observe and save contact info
        info = {}
        pose = pose_msg_to_pos_quaternion(self.robot.get_link_pose(self.EE_LINK_NAME))
        # TODO consider if everything needs to be inside this input lock
        self._observe_contact(pose, wr_np, info)

        info[InfoKeys.HIGH_FREQ_EE_POSE] = np.r_[pose[0], pose[1]]

        # save reaction force
        info[InfoKeys.HIGH_FREQ_REACTION_F] = wr_np[:3]
        info[InfoKeys.HIGH_FREQ_REACTION_T] = wr_np[3:]
        info[InfoKeys.HIGH_FREQ_WRENCH] = wr_np

        self.digest_info(info)

    def digest_info(self, info):
        for key, value in info.items():
            if key not in self._single_step_contact_info:
                self._single_step_contact_info[key] = []
            self._single_step_contact_info[key].append(value)

    def _observe_contact(self, pose, wrench_np, info):
        pos = pose[0]
        orientation = pose[1]
        if self._need_to_force_planar:
            # manually make observed point planar
            orientation = list(p.getEulerFromQuaternion(pose[1]))
            orientation[1] = 0
            orientation[2] = 0
        if self.contact_detector.observe_residual(wrench_np, (pos, orientation)):
            dx = np.subtract(pos, self.last_ee_pos)
            self.contact_detector.observe_dx(dx[:2])
            info[InfoKeys.DEE_IN_CONTACT] = dx
        self.last_ee_pos = pos

    def _fix_torque_to_planar(self, wr_np, fix_threshold=0.065):
        # make force more rectilinear - tilting along z is too powerful
        round_ratio_threshold = 0.5
        if np.abs(wr_np[0]) < np.abs(wr_np[1]) * round_ratio_threshold:
            wr_np[0] *= 0.2
        elif np.abs(wr_np[1]) < np.abs(wr_np[0]) * round_ratio_threshold:
            wr_np[1] *= 0.2
        # no yaw, must mean pushing at the EE, which in this configuration means pushing straight forward
        if np.abs(wr_np[-1]) < 0.1:
            wr_np[1] = 0

        torque_mag = np.linalg.norm(wr_np[3:])
        # since we are planar pushing, shouldn't experience torque along world x and y
        if wr_np[-1] > fix_threshold or wr_np[-1] < -fix_threshold:
            wr_np[3:5] = 0
            wr_np[-1] = torque_mag if wr_np[0] < 0 else -torque_mag
        # magnitude also seems to be off
        wr_np[-1] *= 2.5
        return wr_np

    def setup_experiment(self):
        pass

    # --- observing state
    def _obs(self):
        """Observe current state from ros"""
        state = self._observe_ee(return_z=False)
        return state

    def _observe_ee(self, return_z=False):
        pose = self.robot.get_link_pose(self.EE_LINK_NAME)
        if return_z:
            return np.array([pose.position.x, pose.position.y, pose.position.z])
        else:
            return np.array([pose.position.x, pose.position.y])

    # --- control helpers (rarely overridden)
    def evaluate_cost(self, state, action=None):
        # stub
        return 0, False

    # --- control (commonly overridden)
    def _unpack_action(self, action):
        dx = action[0] * self.MAX_PUSH_DIST
        dy = action[1] * self.MAX_PUSH_DIST
        return dx, dy

    def step(self, action, dz=0., orientation=None):
        self._clear_state_before_step()
        action = np.clip(action, *self.get_control_bounds())
        # self.static_wrench = self.DIR_TO_WRENCH_OFFSET[tuple(action)]

        # normalize action such that the input can be within a fixed range
        dx, dy = self._unpack_action(action)

        self.last_ee_pos = self._observe_ee(return_z=True)
        self._single_step_contact_info = {}

        # set target orientation as rest orientation
        if orientation is None:
            orientation = copy.deepcopy(self.REST_ORIENTATION)
            if self._need_to_force_planar:
                orientation[1] += np.pi / 4
        self.robot.move_delta_cartesian_impedance(self.ACTIVE_MOVING_ARM, dx=dx, dy=dy,
                                                  target_z=self.REST_POS[2] + dz,
                                                  # target_z=0.1,
                                                  blocking=True, step_size=0.025, target_orientation=orientation)
        self.state = self._obs()
        info = self.aggregate_info()

        cost, done = self.evaluate_cost(self.state, action)

        return np.copy(self.state), -cost, done, info

    def aggregate_info(self):
        with self.motion_status_input_lock:
            info = {key: np.stack(value, axis=0) for key, value in self._single_step_contact_info.items() if len(value)}
            # don't need to aggregate external wrench with new contact detector
            info['reaction'], info['torque'] = self._observe_reaction_force_torque(info)
        name = InfoKeys.DEE_IN_CONTACT
        if name in info:
            info[name] = info[name].sum(axis=0)
        else:
            info[name] = np.zeros(3)
        return info

    def _observe_reaction_force_torque(self, info):
        """Return representative reaction force for simulation steps up to current one since last control step"""
        fs = info.get(InfoKeys.HIGH_FREQ_REACTION_F, None)
        if fs is None:
            return np.zeros(3), np.zeros(3)
        mag = np.linalg.norm(fs, axis=1)
        median_mini_step = np.argsort(mag)[len(mag) // 2]
        return fs[median_mini_step], info[InfoKeys.HIGH_FREQ_REACTION_T][median_mini_step]

    def reset(self):
        self.return_to_rest(self.robot.right_arm_group)
        self.state = self._obs()
        self.contact_detector.clear()
        return np.copy(self.state), None

    def close(self):
        pass

    def visualize_contact_set(self, contact_set: tracking.ContactSet):
        if isinstance(contact_set, stucco_experiments.baselines.hard_tracking.ContactSetHard):
            # clear all previous markers because we don't know which one was removed
            if len(self._contact_debug_names) > len(contact_set):
                for name in set.union(*self._contact_debug_names):
                    self.vis.ros.clear_markers(name)
                self._contact_debug_names = []
            for i, c in enumerate(contact_set):
                color, u_color = state_action_color_pairs[i % len(state_action_color_pairs)]
                if i >= len(self._contact_debug_names):
                    self._contact_debug_names.append(set())
                # represent the uncertainty of the center point
                name = 'cp.{}'.format(i)
                eigval, eigvec = torch.eig(c.cov[0], eigenvectors=True)
                yx_ratio = eigval[1, 0] / eigval[0, 0]
                rot = math.atan2(eigvec[1, 0], eigvec[0, 0])
                l = eigval[0, 0] * 100
                w = c.weight
                self.vis.draw_point(name, self.get_ee_pos(c.mu[0]), length=l.item(), length_ratio=yx_ratio, rot=rot,
                                    color=color)
                self._contact_debug_names[i].add(name)

                base_name = str(i)
                names = self.visualize_state_actions(base_name, c.points, c.actions, color, u_color, 0.1 * w)
                for name in names:
                    self._contact_debug_names[i].add(name)
        elif isinstance(contact_set, tracking.ContactSetSoft):
            pts = contact_set.get_posterior_points()
            if pts is None:
                return
            groups = contact_set.get_hard_assignment(contact_set.p.hard_assignment_threshold)
            # clear all previous markers because we don't know which one was removed
            if len(self._contact_debug_names) > len(groups):
                for name in set.union(*self._contact_debug_names):
                    self.vis.ros.clear_markers(name)
                self._contact_debug_names = []
            for i, indices in enumerate(groups):
                color, u_color = state_action_color_pairs[i % len(state_action_color_pairs)]
                if i >= len(self._contact_debug_names):
                    self._contact_debug_names.append(set())
                # represent the uncertainty of the center point
                base_name = str(i)
                names = self.visualize_state_actions(base_name, pts[indices], contact_set.acts[indices], color, u_color,
                                                     0.1)
                for name in names:
                    self._contact_debug_names[i].add(name)

    def visualize_state_actions(self, base_name, states, actions, state_c, action_c, action_scale):
        if torch.is_tensor(states):
            states = states.cpu()
            if actions is not None:
                actions = actions.cpu()
        debug_names_created = []
        j = -1
        for j in range(len(states)):
            p = self.get_ee_pos(states[j])
            p = [p[0], p[1], self.REST_POS[2]]
            name = '{}.{}'.format(base_name, j)
            debug_names_created.append(name)
            self.vis.draw_point(name, p, color=state_c)
            if actions is not None:
                # draw action
                name = '{}a.{}'.format(base_name, j)
                self.vis.draw_2d_line(name, p, actions[j], color=action_c, scale=action_scale)
                debug_names_created.append(name)
        self.vis.clear_visualization_after(base_name, j + 1)
        self.vis.clear_visualization_after('{}a'.format(base_name), j + 1)
        return debug_names_created

    @classmethod
    def _create_sim_robot_base(cls, base_pose=None, canonical_joint=None, canonical_pos=None,
                               canonical_orientation=None,
                               visualizer: typing.Optional[CombinedVisualizer] = None):
        if base_pose is None:
            base_pose = cls.BASE_POSE
        if canonical_pos is None:
            canonical_pos = cls.REST_POS
        if canonical_joint is None:
            canonical_joint = cls.REST_JOINTS
        if canonical_orientation is None:
            canonical_orientation = cls.REST_ORIENTATION

        # create pybullet environment and load kuka arm
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        if visualizer is not None:
            if visualizer.sim is None:
                visualizer.init_sim(0, 1.8)
                assert isinstance(visualizer.sim, DebugDrawer)
                visualizer.sim.set_camera_position(canonical_pos, yaw=90)
                visualizer.sim.toggle_3d(True)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        # get base pose relative to the world frame
        robot_id = p.loadURDF("kuka_iiwa/model.urdf", base_pose[0], base_pose[1], useFixedBase=True)
        # base pose actually doesn't matter as long as our EE is sufficiently close to canonical
        visual_data = p.getVisualShapeData(robot_id)
        for link in visual_data:
            link_id = link[1]
            rgba = list(link[7])
            rgba[3] = 0.5
            p.changeVisualShape(robot_id, link_id, rgbaColor=rgba)

        # move robot EE to desired base pose
        ee_index = 6
        for i, v in enumerate(canonical_joint):
            p.resetJointState(robot_id, i, v)

        data = p.getLinkState(robot_id, ee_index)
        # compare against desired pose
        pos, orientation = data[4], data[5]

        # confirm simmed position and orientation sufficiently close
        d_pos = np.linalg.norm(np.subtract(pos, canonical_pos))
        qd = p.getDifferenceQuaternion(orientation, p.getQuaternionFromEuler(canonical_orientation))
        d_orientation = 2 * math.atan2(np.linalg.norm(qd[:3]), qd[-1])
        if d_pos > 0.05 or d_orientation > 0.04:
            raise RuntimeError(f"sim EE can't arrive at desired EE d pos {d_pos} d orientation {d_orientation}")
        return robot_id, pos, orientation

    @classmethod
    def create_sim_robot_and_gripper(cls, visualizer: typing.Optional[CombinedVisualizer] = None, **kwargs):
        robot_id, pos, orientation = cls._create_sim_robot_base(visualizer=visualizer, **kwargs)
        # create shape that matches the gripper
        col_ids = []
        vis_ids = []
        link_positions = []
        link_orientations = []
        r = 0.03
        l = 0.14
        base_col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=l)
        base_vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=l, rgbaColor=(0.0, 0.8, 0, 0.5))
        # palm
        r = 0.063
        col_ids.append(p.createCollisionShape(p.GEOM_SPHERE, radius=r))
        vis_ids.append(p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=(0, 0.7, 0, 0.5)))
        link_positions.append([0, 0, 0.11])
        link_orientations.append([0, 0, 0, 1])
        # fingers
        half_extents = [0.05, 0.01, 0.06]
        col_ids.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents))
        vis_ids.append(p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=(0, 0.6, 0.0, 0.5)))
        l = 0.024
        ll = 0.17
        open_angle = np.pi / 5.4
        link_positions.append([-l, -l, ll])
        link_orientations.append(p.getQuaternionFromEuler([open_angle, 0, np.pi / 2 + np.pi / 4]))
        # other finger, mirrored
        col_ids.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents))
        vis_ids.append(p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=(0, 0.6, 0.0, 0.5)))
        link_positions.append([l, l, ll])
        link_orientations.append(p.getQuaternionFromEuler([-open_angle, 0, np.pi / 2 + np.pi / 4]))
        # fill space in between
        r = 0.03
        col_ids.append(p.createCollisionShape(p.GEOM_SPHERE, radius=r))
        vis_ids.append(p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=(0, 0.5, 0, 0.5)))
        link_positions.append([0, 0, ll])
        link_orientations.append([0, 0, 0, 1])
        # center of fingers
        half_extents = [0.05, 0.005, 0.01]
        col_ids.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents))
        vis_ids.append(p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=(0, 0.4, 0.0, 0.5)))
        ll = 0.22
        link_positions.append([0, 0, ll])
        link_orientations.append(p.getQuaternionFromEuler([0, 0, np.pi / 2 + np.pi / 4]))

        gripper_id = p.createMultiBody(0, base_col_id, base_vis_id, basePosition=pos, baseOrientation=orientation,
                                       linkCollisionShapeIndices=col_ids, linkVisualShapeIndices=vis_ids,
                                       linkMasses=[0 for _ in col_ids],
                                       linkPositions=link_positions,
                                       linkOrientations=link_orientations,
                                       linkInertialFramePositions=[[0, 0, 0] for _ in col_ids],
                                       linkInertialFrameOrientations=[[0, 0, 0, 1] for _ in col_ids],
                                       linkParentIndices=[0 for _ in col_ids],
                                       linkJointTypes=[p.JOINT_FIXED for _ in col_ids],
                                       linkJointAxis=[[0, 0, 1] for _ in col_ids])

        return robot_id, gripper_id, pos, orientation, []


class RealArmEnvMedusa(RealArmEnv):
    """Same as inherited class, but on medusa rather than victor and with a bubble gripper"""
    MAX_PUSH_DIST = 0.02
    RESET_RAISE_BY = 0.025

    REST_POS = [0.565179880383697, -0.029918686192412114, 0.09889075218881908 - 0.015]
    REST_ORIENTATION = [0.75 * np.pi, 0.5 * np.pi, - 0.75 * np.pi]
    # REST_ORIENTATION_QUAT = [-0.7068252, 0, 0, 0.7073883]
    BASE_POSE = ([0, 0, 0], [0, 0, 0, 1])
    REST_JOINTS = [-0.6247649084794565, 1.49610584350157, 0.6107584653902799, -1.3317384239240129, -2.3319198194954844,
                   -1.2559155511588722, 0.6192390422605022]

    EE_LINK_NAME = "med_kuka_link_7"
    WORLD_FRAME = "med_base"
    ACTIVE_MOVING_ARM = 0  # only 1 arm

    def _setup_robot_ros(self, residual_threshold=10., residual_precision=None):
        self._need_to_force_planar = False
        # adjust timeout according to velocity (at vel = 0.1 we expect 400s per 1m)
        med = BubbleMed(display_goals=False,
                        base_kwargs={'cartesian_impedance_controller_kwargs': {'pose_distance_fn': xy_pose_distance,
                                                                               'timeout_per_m': 40 / self.vel,
                                                                               'position_close_enough': 0.003}})
        self.robot = med
        self.robot.connect()
        self.vis.init_ros(world_frame="world")
        self.gripper = WSG50Gripper()

        self.motion_status_input_lock = Lock()
        self._temp_wrenches = []
        # subscribe to status messages
        self.contact_listener = rospy.Subscriber(self.robot.ns("motion_status"), MotionStatus,
                                                 self.contact_listener)
        self.cleaned_wrench_publisher = rospy.Publisher(self.robot.ns("cleaned_wrench"), WrenchStamped, queue_size=10)
        self.large_wrench_publisher = rospy.Publisher(self.robot.ns("large_wrench"), WrenchStamped, queue_size=10)
        self.large_wrench_world_publisher = rospy.Publisher(self.robot.ns("large_wrench_world"), WrenchStamped,
                                                            queue_size=10)

        # to reset the rest pose, manually jog it there then read the values
        rest_pose = pose_msg_to_pos_quaternion(self.robot.get_link_pose(self.EE_LINK_NAME))

        # reset to rest position
        self.return_to_rest(self.robot.arm_group)
        self.gripper.move(0)

        base_pose = pose_msg_to_pos_quaternion(self.robot.get_link_pose('med_base'))
        status = self.robot.get_arm_status()
        canonical_joints = [status.measured_joint_position.joint_1, status.measured_joint_position.joint_2,
                            status.measured_joint_position.joint_3, status.measured_joint_position.joint_4,
                            status.measured_joint_position.joint_5, status.measured_joint_position.joint_6,
                            status.measured_joint_position.joint_7]

        self.last_ee_pos = self._observe_ee(return_z=True)
        self.REST_POS[2] = self.last_ee_pos[-1]
        self.state = self._obs()

        if residual_precision is None:
            residual_precision = np.diag([1, 1, 0, 1, 1, 1])

        use_cameras = True
        camera_parser_right = BubbleParser(camera_name='pico_flexx_right') if use_cameras else None
        camera_parser_left = BubbleParser(camera_name='pico_flexx_left') if use_cameras else None

        self._contact_detector = ContactDetectorPlanarRealArmBubble("medusa", residual_precision, residual_threshold,
                                                                    camera_parser_left, camera_parser_right,
                                                                    self.EE_LINK_NAME,
                                                                    visualizer=self.vis,
                                                                    canonical_pos=self.REST_POS,
                                                                    canonical_orientation=self.REST_ORIENTATION)
        # parallel visualizer for ROS and pybullet
        self.vis.sim = None

    # --- observing state
    def aggregate_info(self):
        with self.motion_status_input_lock:
            info = {key: np.stack(value, axis=0) for key, value in self._single_step_contact_info.items() if len(value)}
            # don't need to aggregate external wrench with new contact detector
            info['reaction'], info['torque'] = self._observe_reaction_force_torque(info)
        name = InfoKeys.DEE_IN_CONTACT
        if name in info:
            info[name] = info[name].sum(axis=0)
        else:
            info[name] = np.zeros(3)
        return info

    @classmethod
    def create_sim_robot_and_gripper(cls, visualizer: typing.Optional[CombinedVisualizer] = None, **kwargs):
        robot_id, pos, orientation = cls._create_sim_robot_base(visualizer=visualizer, base_pose=cls.BASE_POSE,
                                                                canonical_joint=cls.REST_JOINTS,
                                                                canonical_pos=cls.REST_POS,
                                                                canonical_orientation=cls.REST_ORIENTATION)
        # need to offset by z in orientation to get ee_link frame
        offset = p.rotateVector(orientation, [0, 0, 0.045])

        # load gripper including inflated bubbles
        gripper_id = p.loadURDF(os.path.join(cfg.URDF_DIR, "wsg50_flipped_inflated.urdf"),
                                basePosition=np.add(pos, offset), baseOrientation=orientation)

        # grippers are not connected at the middle, so SDF at the center will be close to 0
        # because we're going to rummage with it closed, we should treat it as a closed object, so add filler body
        filler_body_ids = []
        r = 0.08
        base_col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
        base_vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=(0.0, 0.8, 0, 0.5))

        col_ids = []
        vis_ids = []
        link_positions = []
        link_orientations = []
        # fingers
        half_extents = [0.01, 0.005, 0.071]
        col_ids.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents))
        vis_ids.append(p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=(0, 0.6, 0.0, 0.5)))
        l = 0.024
        ll = -0.02
        open_angle = 1.8 * np.pi / 3
        link_positions.append([0, -l, ll])
        link_orientations.append(p.getQuaternionFromEuler([open_angle, 0, 0]))
        # other finger, mirrored
        col_ids.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents))
        vis_ids.append(p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=(0, 0.6, 0.0, 0.5)))
        link_positions.append([0, l, ll])
        link_orientations.append(p.getQuaternionFromEuler([-open_angle, 0, 0]))

        offset = p.rotateVector(orientation, [0, 0, 0.21])
        # filler_body_ids.append(p.createMultiBody(0, base_col_id, base_vis_id, basePosition=np.add(pos, offset),
        #                                          baseOrientation=orientation))
        filler_body_ids.append(p.createMultiBody(0, base_col_id, base_vis_id, basePosition=np.add(pos, offset),
                                                 baseOrientation=orientation,
                                                 linkCollisionShapeIndices=col_ids, linkVisualShapeIndices=vis_ids,
                                                 linkMasses=[0 for _ in col_ids],
                                                 linkPositions=link_positions,
                                                 linkOrientations=link_orientations,
                                                 linkInertialFramePositions=[[0, 0, 0] for _ in col_ids],
                                                 linkInertialFrameOrientations=[[0, 0, 0, 1] for _ in col_ids],
                                                 linkParentIndices=[0 for _ in col_ids],
                                                 linkJointTypes=[p.JOINT_FIXED for _ in col_ids],
                                                 linkJointAxis=[[0, 0, 1] for _ in col_ids]))

        return robot_id, gripper_id, pos, orientation, filler_body_ids


class ArmRealDataSource(EnvDataSource):
    loader_map = {RealArmEnv: ArmRealLoader, RealArmEnvMedusa: ArmRealLoader}

    @staticmethod
    def _default_data_dir():
        return DIR

    @staticmethod
    def _loader_map(env_type):
        return ArmRealDataSource.loader_map.get(env_type, None)


class KukaResidualContactSensor(PybulletResidualPlanarContactSensor):
    """Residual contact sensor for real robot, with init points loaded from pybullet"""

    def _init_sample_surface_points_in_canonical_pose(self, visualizer: typing.Optional[CombinedVisualizer] = None):
        self.robot_id, gripper_id, pos, orientation, _ = RealArmEnv.create_sim_robot_and_gripper(visualizer=visualizer)
        z = pos[2]
        # instead of actually using the link frame, we'll use a rotated version of it so all points lie on the same z
        # this is because the actual frame is not axis aligned wrt the world
        fixed_orientation = list(p.getEulerFromQuaternion(orientation))
        fixed_orientation[1] = 0
        fixed_orientation[2] = 0

        r = 0.2
        # sample evenly in terms of angles, but leave out the section in between the fingers
        leave_out = 1.0
        start_angle = -np.pi / 2
        angles = np.linspace(start_angle + leave_out, np.pi * 2 - leave_out + start_angle, self.num_sample_points)

        offset_y = 0.2
        sample_pts = []
        for angle in angles:
            pt = [np.cos(angle) * r + pos[0], np.sin(angle) * r + pos[1] + offset_y, z]
            sample_pts.append(pt)

        cached_points, cached_normals = self._project_sample_points_to_surface([self.robot_id, gripper_id], pos,
                                                                               fixed_orientation, sample_pts,
                                                                               visualizer=visualizer)

        if visualizer is not None:
            x = tf.Translate(*self._canonical_pos, device=self.device, dtype=self.dtype)
            # actual orientation is rotated wrt world frame so not all points are on same z level
            orientation = copy.deepcopy(self._canonical_orientation)
            orientation[1] = 0
            orientation[2] = 0
            r = tf.Rotate(orientation, device=self.device, dtype=self.dtype)
            trans = x.compose(r)
            ros_pts = trans.transform_points(cached_points)

            for i, min_pt_at_z in enumerate(ros_pts):
                t = i / len(cached_points)
                visualizer.ros.draw_point(f'c.{i}', min_pt_at_z, color=(t, t, 1 - t))

        p.disconnect()
        return cached_points, cached_normals


class BubbleResidualContactSensor(PybulletResidualPlanarContactSensor):
    """Residual contact sensor for real robot with a bubble gripper, with init points loaded from pybullet"""

    def _init_sample_surface_points_in_canonical_pose(self, visualizer: typing.Optional[CombinedVisualizer] = None):
        robot_id, gripper_id, pos, orientation, filler_ids = RealArmEnvMedusa.create_sim_robot_and_gripper(
            visualizer=visualizer)

        # single point at the front
        points = []
        offset = p.rotateVector(orientation, [0, 0, 0.4])
        points.append(np.add(pos, offset))
        # points on the side
        points.append(np.add(pos, p.rotateVector(orientation, [0, 0.15, 0.15])))
        points.append(np.add(pos, p.rotateVector(orientation, [0, -0.15, 0.15])))

        cached_points, cached_normals = self._project_sample_points_to_surface([robot_id, gripper_id] + filler_ids, pos,
                                                                               orientation, points,
                                                                               visualizer=visualizer)

        p.disconnect()
        return cached_points, cached_normals


class BubbleCameraContactSensor(ContactSensor):
    def __init__(self, camera: BubbleParser, link_frame, imprint_threshold=0.004,
                 deform_number_threshold=20, ref_img=None, dtype=torch.float, device='cpu'):
        self.camera = camera
        self.link_frame = link_frame
        self.ref = ref_img
        self.cache = None
        self.imprint_threshold = imprint_threshold
        self.deform_number_threshold = deform_number_threshold

        self.dtype = dtype
        self.device = device

        if self.camera is not None:
            self.input_lock = Lock()
            self.ref = self.camera.get_image_depth()
            info = self.camera.get_camera_info_depth()
            self.K = info['K']

            self.camera.register_callback('depth', self._depth_img_callback)
        else:
            rospy.logwarn("Creating contact detector without camera")

        super().__init__()

    def clear(self):
        super().clear()
        self.cache = None

    def observe_residual(self, residual):
        with self.input_lock:
            self.in_contact = self.cache is not None and self.cache['contact']

    def isolate_contact(self, ee_force_torque, pose, q=None, visualizer=None):
        return self.get_link_frame_deform_points()

    def get_last_obs_time(self):
        if self.cache is not None:
            return self.cache['time']
        return None

    def get_link_frame_deform_points(self):
        if self.cache is not None and self.cache['contact']:
            cache = self.get_cache()
            return self.process_link_frame_deform_points(cache, self.camera)
        return None

    def get_cached_obs(self, key):
        with self.input_lock:
            if self.cache is not None:
                return self.cache[key]
        return None

    def get_cache(self):
        # return a copy of the current cache
        if self.cache is not None:
            with self.input_lock:
                return copy.deepcopy(self.cache)
        return None

    def _depth_img_callback(self, img):
        with self.input_lock:
            self.cache = self._process_depth_img(img, self.ref)

    def _process_depth_img(self, img, ref_img):
        unfiltered_imprint = ref_img - img
        imprint = process_bubble_img(unfiltered_imprint)
        # check if any are above threshold
        deform = imprint > self.imprint_threshold
        deform_num = deform.sum()
        deform_significant = deform_num > self.deform_number_threshold
        # stamp current time it was processed
        obs_time = rospy.Time.now()
        return {'ref_img': ref_img, 'depth_img': img,
                'imprint': imprint, 'unfiltered_imprint': unfiltered_imprint,
                'mask': deform, 'mask_num': deform_num,
                'contact': deform_significant,
                'time': obs_time,
                }

    def process_link_frame_deform_points(self, cache, camera):
        # return the undeformed point to prevent penetration with the model
        depth_im = cache['ref_img']
        mask = cache['mask']
        # these are in optical/camera frame
        # average in image space, then project the average to point cloud
        vs, us, _ = np.nonzero(mask)
        v, u = vs.mean(), us.mean()
        # interpolation on the depth image to get depth value
        d = bilinear_interpolate(depth_im, u, v)

        pt = project_depth_points(u, v, d, self.K)
        pt_l = camera.transform_pc(np.array(pt).reshape(1, -1), camera.optical_frame['depth'], self.link_frame)
        cache['contact_avg_pixel'] = (v, u)
        cache['contact_pt_link_frame'] = pt_l
        return torch.tensor(pt_l[0], dtype=self.dtype, device=self.device)


class ContactDetectorPlanarRealArmBubble(ContactDetector):
    """Contact detector using the bubble gripper"""

    def __init__(self, name, residual_precision, residual_threshold,
                 camera_l: BubbleParser, camera_r: BubbleParser,
                 ee_link_frame: str, imprint_threshold=0.004, deform_number_threshold=20,
                 window_size=5, dtype=torch.float, device='cpu', **kwargs):
        super().__init__(residual_precision, window_size=window_size, dtype=dtype, device=device)
        self.register_contact_sensor(BubbleResidualContactSensor(name, residual_threshold, **kwargs))
        if camera_l is not None:
            self.register_contact_sensor(BubbleCameraContactSensor(camera_l, ee_link_frame,
                                                                   imprint_threshold=imprint_threshold,
                                                                   deform_number_threshold=deform_number_threshold,
                                                                   dtype=dtype, device=device))
            self.register_contact_sensor(BubbleCameraContactSensor(camera_r, ee_link_frame,
                                                                   imprint_threshold=imprint_threshold,
                                                                   deform_number_threshold=deform_number_threshold,
                                                                   dtype=dtype, device=device))

            plt.ion()
            self.imprint_norm = matplotlib.colors.Normalize(vmin=-imprint_threshold, vmax=imprint_threshold)
            self.imprint_fig, [self.imprint_ax_l, self.imprint_ax_r] = plt.subplots(ncols=2, sharex=True, sharey=True,
                                                                                    figsize=(9, 3))
            self.imprint_fig.colorbar(matplotlib.cm.ScalarMappable(norm=self.imprint_norm),
                                      ax=[self.imprint_ax_l, self.imprint_ax_r])

            move_figure(self.imprint_fig, 0, 0)
            self.im_handle = {}
            self.removable_artists = []
            plt.show()
        else:
            rospy.logwarn("Creating contact detector without camera")

    def _draw_deformation(self, ax, **cache_content):
        imprint = cache_content['imprint']
        # debug visualization of depth images and imprints
        if ax not in self.im_handle:
            self.im_handle[ax] = ax.imshow(imprint.squeeze(), norm=self.imprint_norm)
        else:
            self.im_handle[ax].set_data(imprint.squeeze())
        vu = cache_content.get('contact_avg_pixel', None)
        if vu is not None:
            self.removable_artists.append(ax.annotate('contact pt', xy=(vu[1], vu[0]), xycoords='data', xytext=(0, 25),
                                                      textcoords='offset points',
                                                      arrowprops=dict(facecolor='black', shrink=0.05),
                                                      horizontalalignment='center',
                                                      verticalalignment='top'))
        plt.pause(0.0001)

    def draw_deformation(self):
        # always draw deformations even if we don't have an isolated contact (why moved outside isolate_contact)
        cache_l = self.sensors[1].cache
        cache_r = self.sensors[2].cache
        for artist in self.removable_artists:
            artist.remove()
        self.removable_artists = []
        if cache_l is not None:
            self._draw_deformation(self.imprint_ax_l, **cache_l)
        if cache_r is not None:
            self._draw_deformation(self.imprint_ax_r, **cache_r)

    def get_last_contact_location(self, pose=None, **kwargs):
        # call this first to process the caches
        ret = super(ContactDetectorPlanarRealArmBubble, self).get_last_contact_location(pose=pose, **kwargs)
        self.draw_deformation()
        return ret

    def isolate_contact(self, ee_force_torque, pose, q=None, visualizer=None):
        pt = []
        dx = []

        for s in range(1, len(self.sensors)):
            pt_candidate = self.sensors[s].isolate_contact(ee_force_torque, pose, q=q,
                                                           visualizer=visualizer)
            if pt_candidate is not None:
                pt.append(pt_candidate)
                dx.append(self.sensors[s].get_dx())
        if len(pt) == 0:
            # can pass points for no bubble deformation into normal pipeline, instead of always selecting it
            rospy.loginfo("Non-bubble contact")
            pt_candidate = self.sensors[0].isolate_contact(ee_force_torque, pose, q=q, visualizer=None)
            pt.append(pt_candidate)
            dx.append(self.sensors[0].get_dx())

        if pt[0] is None:
            pt = None
        else:
            pt = torch.stack(pt)
        dx = torch.stack(dx)

        if visualizer is not None and pt is not None:
            xr = tf.Transform3d(device=self.device, dtype=self.dtype, pos=pose[0], rot=tf.xyzw_to_wxyz(pose[1]))
            pts = xr.transform_points(pt)
            visualizer.draw_point(f'most likely contact', pts[0], color=(0, 1, 0), scale=2)
            visualizer.draw_2d_line('reaction', pts[0], ee_force_torque.mean(dim=0)[:3], color=(0, 0.2, 1.0),
                                    scale=0.2)

        return pt, dx


class RealArmMovableSDF(PlanarMovableSDF):
    def __init__(self, env: RealArmEnv):
        # try loading cache
        # assume that each class has its own configuration, so only need 1 cache per class
        fullname = os.path.join(cfg.DATA_DIR, f'{env.__class__.__name__}_point_to_config.pkl')
        if os.path.exists(fullname):
            super(RealArmMovableSDF, self).__init__(*torch.load(fullname))
        else:
            robot_id, gripper_id, pos, orientation, filler_body_ids = env.create_sim_robot_and_gripper(
                visualizer=env.vis)
            mins = []
            maxs = []
            for i in range(-1, p.getNumJoints(gripper_id)):
                aabb_min, aabb_max = p.getAABB(gripper_id, i)
                mins.append(aabb_min)
                maxs.append(aabb_max)
            # total AABB
            aabb_min = np.min(mins, axis=0)
            aabb_max = np.max(maxs, axis=0)

            extents = np.subtract(aabb_max, aabb_min)
            aabb_vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=extents / 2, rgbaColor=(1, 0, 0, 0.3))
            aabb_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=aabb_vis_id,
                                        basePosition=np.mean([aabb_min, aabb_max], axis=0))
            p.removeBody(aabb_id)

            # cache points inside bounding box of robot to accelerate lookup
            min_x, min_y = aabb_min[:2]
            max_x, max_y = aabb_max[:2]
            cache_resolution = 0.001
            # create mesh grid
            x = np.arange(min_x, max_x + cache_resolution, cache_resolution)
            y = np.arange(min_y, max_y + cache_resolution, cache_resolution)
            cache_y_len = len(y)

            d = np.zeros((len(x), len(y)))
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    pt = [xi, yj, pos[2]]
                    closest_arm = closest_point_on_surface(robot_id, pt)
                    closest_gripper = closest_point_on_surface(gripper_id, pt)
                    dists = [closest_arm[ContactInfo.DISTANCE], closest_gripper[ContactInfo.DISTANCE]]
                    for body_id in filler_body_ids:
                        dists.append(closest_point_on_surface(body_id, pt)[ContactInfo.DISTANCE])
                    d[i, j] = min(dists)
            d_cache = d.reshape(-1)
            # save things in (rotated) link frame, so subtract the REST PSO
            min_x -= pos[0]
            max_x -= pos[0]
            min_y -= pos[1]
            max_y -= pos[1]
            data = [d_cache, min_x, min_y, max_x, max_y, cache_resolution, cache_y_len]
            torch.save(data, fullname)
            super(RealArmMovableSDF, self).__init__(*data)
