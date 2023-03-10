import argparse
import copy
import time
import torch
import pybullet as p
import numpy as np
import logging
import os
from datetime import datetime

from sklearn.cluster import Birch, DBSCAN, KMeans

from stucco_experiments.baselines.cluster import OnlineAgglomorativeClustering, OnlineSklearnFixedClusters
from base_experiments.defines import NO_CONTACT_ID
from stucco_experiments.evaluation import compute_contact_error, clustering_metrics, object_robot_penetration_score
from base_experiments.env.env import InfoKeys

from arm_pytorch_utilities import rand, tensor_utils, math_utils

from base_experiments import cfg
from stucco import tracking
from stucco_experiments.env import arm
from stucco_experiments.env.arm import Levels
from stucco_experiments.env_getters.arm import RetrievalGetter

from stucco_experiments.retrieval_controller import rot_2d_mat_to_angle, \
    sample_model_points, pose_error, TrackingMethod, OurSoftTrackingMethod, \
    SklearnTrackingMethod, KeyboardController, PHDFilterTrackingMethod
from base_experiments.util import MakedirsFileHandler
from stucco_experiments import registration

ch = logging.StreamHandler()
fh = MakedirsFileHandler(os.path.join(cfg.LOG_DIR, "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)


def run_retrieval(env, method: TrackingMethod, seed=0, ctrl_noise_max=0.005):
    dtype = torch.float32

    predetermined_control = {}

    ctrl = [[0.7, -1]] * 5
    ctrl += [[0.4, 0.4], [.5, -1]] * 6
    ctrl += [[-0.2, 1]] * 4
    ctrl += [[0.3, -0.3], [0.4, 1]] * 4
    ctrl += [[1., -1]] * 3
    ctrl += [[1., 0.6], [-0.7, 0.5]] * 4
    ctrl += [[0., 1]] * 5
    ctrl += [[1., 0]] * 4
    ctrl += [[0.4, -1.], [0.4, 0.5]] * 4
    rand.seed(0)
    noise = (np.random.rand(len(ctrl), 2) - 0.5) * 0.5
    ctrl = np.add(ctrl, noise)
    predetermined_control[Levels.SIMPLE_CLUTTER] = ctrl

    ctrl = [[0.9, -0.3]] * 2
    ctrl += [[1.0, 0.], [-.2, -0.6]] * 6
    ctrl += [[0.1, -0.9], [0.8, -0.6]] * 1
    ctrl += [[0.1, 0.8], [0.1, 0.9], [0.1, -1.0], [0.1, -1.0], [0.2, -1.0]]
    ctrl += [[0.1, 0.8], [0.1, 0.9], [0.3, 0.8], [0.4, 0.6]]
    ctrl += [[0.1, -0.8], [0.1, -0.9], [0.3, -0.8]]
    ctrl += [[-0.2, 0.8], [0.1, 0.9], [0.3, 0.8]]
    ctrl += [[-0., -0.8], [0.1, -0.7]]
    ctrl += [[0.4, -0.5], [0.2, -1.0]]
    ctrl += [[-0.2, -0.5], [0.2, -0.6]]
    ctrl += [[0.2, 0.4], [0.2, -1.0]]
    ctrl += [[0.4, 0.4], [0.4, -0.4]] * 3
    ctrl += [[-0.5, 1.0]] * 3
    ctrl += [[0.5, 0.], [0, 0.4]] * 3
    predetermined_control[Levels.FLAT_BOX] = ctrl

    # created from keyboard controller
    predetermined_control[Levels.BEHIND_CAN] = [(0, 1), (0, -1), (0, -1), (1, -1), (0, -1), (-1, 0), (1, -1), (-1, 0),
                                                (1, -1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
                                                (0, -1), (0, -1), (1, -1), (1, 0), (0, 1), (1, -1), (0, 1), (0, 1),
                                                (-1, 0), (0, 1), (-1, 0), (1, 1), (-1, 0), (1, 1), (0, -1), (1, 1),
                                                (0, -1), (1, 0), (1, 1), (-1, -1), (0, 1), (0, 1), (-1, 0), (1, 1),
                                                (-1, -1), (0, -1), (-1, 0), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1),
                                                (0, -1), (1, -1), (0, -1), (0, -1), (-1, 0), (1, -1), (-1, 0), (1, -1),
                                                (-1, 0), (1, -1), (0, 1), (0, 1), (1, -1), (1, -1), (0, -1), (0, 1),
                                                (0, 1), (1, 0), (1, 0), (1, -1), (0, 1), (1, 1), (1, -1), (0, 1),
                                                (1, -1), (-1, 0), (0, -1), (0, 1), (1, 1), (0, 1), (-1, -1), (0, 1),
                                                (-1, -1), (0, 1), (-1, -1), (-1, -1), (1, 0), (0, 1), (0, 1), (-1, -1),
                                                (0, 1), (-1, -1), (-1, 0), (-1, 0), (0, 1), (-1, 1), (-1, 1), (0, 1),
                                                (0, 1), (1, 0), (1, 0), (-1, 0), (0, 1), (1, 0)]

    ctrl = [(0, 1), (1, -1), (0, 1), (0, -1), (0, 1), (0, 1), (1, 1), (0, -1), (0, -1), (0, 1), (1, -1), (1, 1),
            (1, -1), (-1, 0), (0, -1), (0, 1), (1, 1), (0, -1), (1, 0), (1, 1), (1, -1), (0, 1), (-1, 0), (1, 0),
            (0, -1), (0, -1), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, -1), (0, -1), (0, -1), (0, -1),
            (0, -1), (1, 0), (0, -1), (1, 1)]
    ctrl += [(-1, 0), (0, 1), (1, 1), (0, -1)]

    predetermined_control[Levels.IN_BETWEEN] = ctrl

    ctrl = [(1, 0), (0, -1), (-1, 0), (0, -1), (1, 1), (-1, 0), (-1, 0), (1, 0),
            (0, -1), (1, -1), (-1, 0), (0, -1), (-1, 0), (0, -1), (1, 0), (-1, 0),
            (0, 1), (1, 1), (1, 1), (1, 1), (-1, 0), (0, 1), (1, 1), (-1, 0),
            (0, 1), (1, 0), (1, 0), (-1, 0), (0, -1), (0, -1), (0, -1), (0, -1),
            (0, 1), (0, -1), (1, -1), (0, 1), (1, -1), (1, -1), (0, 1), (0, -1), ]
    # ctrl += [(1., -1)]
    # ctrl += [(0., 0.7)] * 2
    # ctrl += [(-0.6, 0.1), (0.8, 0.4)] * 3
    predetermined_control[Levels.TOMATO_CAN] = ctrl

    for k, v in predetermined_control.items():
        rand.seed(seed)
        predetermined_control[k] = np.add(v, (np.random.rand(len(v), 2) - 0.5) * ctrl_noise_max)

    ctrl = method.create_controller(predetermined_control[env.level])

    obs = env.reset()
    z = env._observe_ee(return_z=True)[-1]

    model_name = "tomato_can" if env.level in [Levels.TOMATO_CAN] else "cheezit"
    sample_in_order = env.level in [Levels.TOMATO_CAN]
    model_points, model_normals, bb = sample_model_points(env.target_object_id, num_points=50, force_z=z, seed=0,
                                                          name=model_name, clean_cache=True,
                                                          sample_in_order=sample_in_order)
    mph = model_points.clone().to(dtype=dtype)
    bb = bb.to(dtype=dtype)
    # make homogeneous [x, y, 1]
    mph[:, -1] = 1

    ctrl.set_goal(env.goal[:2])
    info = None
    simTime = 0

    B = 30
    best_tsf_guess = None
    guess_pose = None
    pose_error_per_step = {}

    pt_to_config = arm.ArmMovableSDF(env)

    contact_id = []

    rand.seed(seed)
    while not ctrl.done():
        best_distance = None
        simTime += 1
        env.draw_user_text("{}".format(simTime), xy=(0.5, 0.7, -1))

        action = ctrl.command(obs, info)
        method.visualize_contact_points(env)
        if env.contact_detector.in_contact():
            contact_id.append(info[InfoKeys.CONTACT_ID])
            all_configs = torch.tensor(np.array(ctrl.x_history), dtype=dtype, device=mph.device).view(-1, env.nx)
            dist_per_est_obj = []
            transforms_per_object = []
            best_segment_idx = 0
            for k, this_pts in enumerate(method):
                this_pts = tensor_utils.ensure_tensor(model_points.device, dtype, this_pts)
                T, distances, _ = registration.icp_3(this_pts.view(-1, 2), model_points[:, :2],
                                                     given_init_pose=best_tsf_guess, batch=B)
                transforms_per_object.append(T)
                T = T.inverse()
                penetration = [object_robot_penetration_score(pt_to_config, all_configs, T[b], mph) for b in
                               range(T.shape[0])]
                penetration_score = np.abs(penetration)
                best_tsf_index = np.argmin(penetration_score)

                # pick object with lowest variance in its translation estimate
                translations = T[:, :2, 2]
                best_tsf_distances = (translations.var(dim=0).sum()).item()

                dist_per_est_obj.append(best_tsf_distances)
                if best_distance is None or best_tsf_distances < best_distance:
                    best_distance = best_tsf_distances
                    best_tsf_guess = T[best_tsf_index].inverse()
                    best_segment_idx = k

                # for j in range(len(T)):
                #     tf_bb = bb @ T[j].transpose(-1, -2)
                #     for i in range(len(tf_bb)):
                #         pt = [tf_bb[i][0], tf_bb[i][1], z]
                #         next_pt = [tf_bb[(i + 1) % len(tf_bb)][0], tf_bb[(i + 1) % len(tf_bb)][1], z]
                #         env.vis.draw_2d_line(f"tmptbestline{k}-{j}.{i}", pt, np.subtract(next_pt, pt), color=(0, 1, 0),
                #                              size=0.5, scale=1)

            method.register_transforms(transforms_per_object[best_segment_idx], best_tsf_guess)
            logger.debug(f"err each obj {np.round(dist_per_est_obj, 4)}")
            best_T = best_tsf_guess.inverse()

            target_pose = p.getBasePositionAndOrientation(env.target_object_id)
            yaw = p.getEulerFromQuaternion(target_pose[1])[-1]
            target_pose = [target_pose[0][0], target_pose[0][1], yaw]

            guess_pose = [best_T[0, 2].item(), best_T[1, 2].item(), rot_2d_mat_to_angle(best_T.view(1, 3, 3)).item()]
            pos_err, yaw_err = pose_error(target_pose, guess_pose)

            pose_error_per_step[simTime] = pos_err + 0.3 * yaw_err
            logger.debug(f"pose error {simTime}: {pos_err} {yaw_err} {pose_error_per_step[simTime]}")

            # plot current best estimate of object pose (first block samples points, next one is the full perimeter)
            tf_bb = bb @ best_T.transpose(-1, -2)
            for i in range(len(tf_bb)):
                pt = [tf_bb[i][0], tf_bb[i][1], z]
                next_pt = [tf_bb[(i + 1) % len(tf_bb)][0], tf_bb[(i + 1) % len(tf_bb)][1], z]
                env.vis.draw_2d_line(f"tmptbestline.{i}", pt, np.subtract(next_pt, pt), color=(0, 0, 1), size=2,
                                     scale=1)
        else:
            contact_id.append(NO_CONTACT_ID)

        if torch.is_tensor(action):
            action = action.cpu()

        action = np.array(action).flatten()
        obs, rew, done, info = env.step(action)

    # evaluate FMI and contact error here
    labels, moved_points = method.get_labelled_moved_points(np.ones(len(contact_id)) * NO_CONTACT_ID)
    contact_id = np.array(contact_id)

    in_label_contact = contact_id != NO_CONTACT_ID

    m = clustering_metrics(contact_id[in_label_contact], labels[in_label_contact])
    contact_error = compute_contact_error(None, moved_points, None, env=env, visualize=False)
    cme = np.mean(np.abs(contact_error))

    grasp_at_pose(env, guess_pose)

    return m, cme


def grasp_at_pose(env, pose):
    # object is symmetric so pose can be off by 180
    yaw = pose[2]
    if env.level == Levels.FLAT_BOX:
        grasp_offset = [0., -0.25]
        if yaw > np.pi / 2:
            yaw -= np.pi
        elif yaw < -np.pi / 2:
            yaw += np.pi
    elif env.level == Levels.BEHIND_CAN or env.level == Levels.IN_BETWEEN:
        grasp_offset = [0., -0.25]
        if yaw > 0:
            yaw -= np.pi
        elif yaw < -np.pi:
            yaw += np.pi
    elif env.level == Levels.TOMATO_CAN:
        grasp_offset = [0, -0.2]
        # cylinder so doesn't matter what yaw we come in at
        yaw = -np.pi / 2
    else:
        raise RuntimeError(f"No data for level {env.level}")

    grasp_offset = math_utils.rotate_wrt_origin(grasp_offset, yaw)
    target_pos = [pose[0] + grasp_offset[0], pose[1] + grasp_offset[1]]
    z = env._observe_ee(return_z=True)[-1]
    env.vis.draw_point("pre_grasp", [target_pos[0], target_pos[1], z], color=(1, 0, 0))
    # get to target pos
    obs = env._obs()
    diff = np.subtract(target_pos, obs)
    start = time.time()
    while np.linalg.norm(diff) > 0.01 and time.time() - start < 5:
        obs, _, _, _ = env.step(diff / env.MAX_PUSH_DIST)
        diff = np.subtract(target_pos, obs)
    # rotate in place
    prev_ee_orientation = copy.deepcopy(env.endEffectorOrientation)
    env.endEffectorOrientation = p.getQuaternionFromEuler([0, np.pi / 2, yaw + np.pi / 2])
    env.sim_step_wait = 0.01
    env.step([0, 0])
    env.open_gripper()
    env.step([0, 0])
    env.sim_step_wait = None
    # go for the grasp

    move_times = 4
    move_dir = -np.array(grasp_offset)
    while move_times > 0:
        act_mag = move_times if move_times <= 1 else 1
        move_times -= 1
        u = move_dir / np.linalg.norm(move_dir) * act_mag
        obs, _, _, _ = env.step(u)
    env.sim_step_wait = 0.01
    env.close_gripper()
    env.step([0, 0])
    env.sim_step_wait = None

    env.endEffectorOrientation = prev_ee_orientation


def main(env, method_name, seed=0):
    methods_to_run = {
        'ours': OurSoftTrackingMethod(env, RetrievalGetter.contact_parameters(env), arm.ArmMovableSDF(env)),
        'online-birch': SklearnTrackingMethod(env, OnlineAgglomorativeClustering, Birch, n_clusters=None,
                                              inertia_ratio=0.2,
                                              threshold=0.08),
        'online-dbscan': SklearnTrackingMethod(env, OnlineAgglomorativeClustering, DBSCAN, eps=0.05, min_samples=1),
        'online-kmeans': SklearnTrackingMethod(env, OnlineSklearnFixedClusters, KMeans, inertia_ratio=0.2, n_clusters=1,
                                               random_state=0),
        'gmphd': PHDFilterTrackingMethod(env, fp_fn_bias=4, q_mag=0.00005, r_mag=0.00005, birth=0.001, detection=0.3)
    }
    env.draw_user_text(f"{method_name} seed {seed}", xy=[-0.1, 0.28, -0.5])
    return run_retrieval(env, methods_to_run[method_name], seed=seed)


def keyboard_control(env):
    print("waiting for arrow keys to be pressed to command a movement")
    contact_params = RetrievalGetter.contact_parameters(env)
    pt_to_config = arm.ArmMovableSDF(env)
    contact_set = tracking.ContactSetSoft(pt_to_config, contact_params)
    ctrl = KeyboardController(env.contact_detector, contact_set, nu=2)

    obs = env._obs()
    info = None
    while not ctrl.done():
        try:
            env.visualize_contact_set(contact_set)
            u = ctrl.command(obs, info)
            obs, _, done, info = env.step(u)
        except:
            pass
        time.sleep(0.05)
    print(ctrl.u_history)
    cleaned_u = [u for u in ctrl.u_history if u != (0, 0)]
    print(cleaned_u)


parser = argparse.ArgumentParser(description='Downstream task of blind object retrieval')
parser.add_argument('method',
                    choices=['ours', 'ours-rummage', 'online-birch', 'online-dbscan', 'online-kmeans', 'gmphd'],
                    help='which method to run')
parser.add_argument('--seed', metavar='N', type=int, nargs='+',
                    default=[0],
                    help='random seed(s) to run')
parser.add_argument('--no_gui', action='store_true', help='force no GUI')
# run parameters
task_map = {"FB": Levels.FLAT_BOX, "BC": Levels.BEHIND_CAN, "IB": Levels.IN_BETWEEN, "SC": Levels.SIMPLE_CLUTTER,
            "TC": Levels.TOMATO_CAN}
parser.add_argument('--task', default="IB", choices=task_map.keys(), help='what task to run')

args = parser.parse_args()

if __name__ == "__main__":
    level = task_map[args.task]
    method_name = args.method

    env = RetrievalGetter.env(level=level, mode=p.DIRECT if args.no_gui else p.GUI)

    # manual keyboard control to generate rummaging policy
    # keyboard_control(env)
    # exit(0)

    fmis = []
    cmes = []
    # backup video logging in case ffmpeg and nvidia driver are not compatible
    # with WindowRecorder(window_names=("Bullet Physics ExampleBrowser using OpenGL3+ [btgl] Release build",),
    #                     name_suffix="sim", frame_rate=30.0, save_dir=cfg.VIDEO_DIR):
    for seed in args.seed:
        m, cme = main(env, method_name, seed=seed)
        fmi = m[0]
        fmis.append(fmi)
        cmes.append(cme)
        logger.info(f"{method_name} fmi {fmi} cme {cme}")
        env.vis.clear_visualizations()
        env.reset()

    logger.info(
        f"{method_name} mean fmi {np.mean(fmis)} median fmi {np.median(fmis)} std fmi {np.std(fmis)} {fmis}\n"
        f"mean cme {np.mean(cmes)} median cme {np.median(cmes)} std cme {np.std(cmes)} {cmes}")
    env.close()
