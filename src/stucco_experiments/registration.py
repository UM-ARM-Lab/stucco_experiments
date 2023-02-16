import math

import torch

from pytorch_kinematics import transforms as tf


def nearest_neighbor_torch(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: bxMxm array of points
        dst: bxNxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    dist = torch.cdist(src, dst)
    knn = dist.topk(1, largest=False)
    return knn


def best_fit_transform_torch(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: bxNxm numpy array of corresponding points
      B: bxNxm numpy array of corresponding points
    Returns:
      T: bx(m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: bxmxm rotation matrix
      t: bxmx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[-1]
    b = A.shape[0]

    # translate points to their centroids
    centroid_A = torch.mean(A, dim=-2, keepdim=True)
    centroid_B = torch.mean(B, dim=-2, keepdim=True)
    AA = A - centroid_A
    BB = B - centroid_B

    # Orthogonal Procrustes Problem
    # minimize E(R,t) = sum_{i,j} ||bb_i - Raa_j - t||^2
    # equivalent to minimizing ||BB - R AA||^2
    # rotation matrix
    H = AA.transpose(-1, -2) @ BB
    U, S, Vt = torch.svd(H)
    # assume H is full rank, then the minimizing R and t are unique
    R = Vt.transpose(-1, -2) @ U.transpose(-1, -2)

    # special reflection case
    reflected = torch.det(R) < 0
    Vt[reflected, m - 1, :] *= -1
    R[reflected] = Vt[reflected].transpose(-1, -2) @ U[reflected].transpose(-1, -2)

    # translation
    t = centroid_B.transpose(-1, -2) - (R @ centroid_A.transpose(-1, -2))

    # homogeneous transformation
    T = torch.eye(m + 1, dtype=A.dtype, device=A.device).repeat(b, 1, 1)
    T[:, :m, :m] = R
    T[:, :m, m] = t.view(b, -1)

    return T, R, t


def init_random_transform_with_given_init(m, batch, dtype, device, given_init_pose=None):
    # apply some random initial poses
    if m > 2:
        R = tf.random_rotations(batch, dtype=dtype, device=device)
    else:
        theta = torch.rand(batch, dtype=dtype, device=device) * math.pi * 2
        Rtop = torch.cat([torch.cos(theta).view(-1, 1), -torch.sin(theta).view(-1, 1)], dim=1)
        Rbot = torch.cat([torch.sin(theta).view(-1, 1), torch.cos(theta).view(-1, 1)], dim=1)
        R = torch.cat((Rtop.unsqueeze(-1), Rbot.unsqueeze(-1)), dim=-1)

    init_pose = torch.eye(m + 1, dtype=dtype, device=device).repeat(batch, 1, 1)
    init_pose[:, :m, :m] = R[:, :m, :m]
    if given_init_pose is not None:
        # check if it's given as a batch
        if len(given_init_pose.shape) == 3:
            init_pose = given_init_pose.clone()
        else:
            init_pose[0] = given_init_pose
    return init_pose


def icp_3(A, B, A_normals=None, B_normals=None, normal_scale=0.1, given_init_pose=None, max_iterations=20,
          tolerance=1e-4, batch=5, vis=None):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Mxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        A_normals: Mxm numpy array of source mD surface normal vectors
        B_normals: Nxm numpy array of destination mD surface normal vectors
        given_init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = torch.ones((m + 1, A.shape[0]), dtype=A.dtype, device=A.device)
    dst = torch.ones((B.shape[0], m + 1), dtype=A.dtype, device=A.device)
    src[:m, :] = torch.clone(A.transpose(0, 1))
    dst[:, :m] = torch.clone(B)
    src = src.repeat(batch, 1, 1)
    dst = dst.repeat(batch, 1, 1)

    # apply some random initial poses
    init_pose = init_random_transform_with_given_init(m, batch, A.dtype, A.device, given_init_pose=given_init_pose)

    # apply the initial pose estimation
    src = init_pose @ src
    src_normals = A_normals if normal_scale > 0 else None
    dst_normals = B_normals if normal_scale > 0 else None
    if src_normals is not None and dst_normals is not None:
        # NOTE normally we need to multiply by the transpose of the inverse to transform normals, but since we are sure
        # the transform does not include scale, we can just use the matrix itself
        # NOTE normals have to be transformed in the opposite direction as points!
        src_normals = src_normals.repeat(batch, 1, 1) @ init_pose[:, :m, :m].transpose(-1, -2)
        dst_normals = dst_normals.repeat(batch, 1, 1)

    prev_error = 0
    err_list = []

    if vis is not None:
        for j in range(A.shape[0]):
            pt = src[0, :m, j]
            vis.draw_point(f"impt.{j}", pt, color=(0, 1, 0), length=0.003)
            if src_normals is not None:
                vis.draw_2d_line(f"imn.{j}", pt, -src_normals[0, j], color=(0, 0.4, 0), size=2., scale=0.03)

    i = 0
    distances = None
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        # if given normals, scale and append them to find nearest neighbours
        p = src[:, :m, :].transpose(-2, -1)
        q = dst[:, :, :m]
        if src_normals is not None:
            p = torch.cat((p, src_normals * normal_scale), dim=-1)
            q = torch.cat((q, dst_normals * normal_scale), dim=-1)
        distances, indices = nearest_neighbor_torch(p, q)
        # currently only have a single batch so flatten
        distances = distances.view(batch, -1)
        indices = indices.view(batch, -1)

        fit_from = src[:, :m, :].transpose(-2, -1)
        to_fit = []
        for b in range(batch):
            to_fit.append(dst[b, indices[b], :m])
        to_fit = torch.stack(to_fit)
        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform_torch(fit_from, to_fit)

        # update the current source
        src = T @ src
        if src_normals is not None and dst_normals is not None:
            src_normals = src_normals @ T[:, :m, :m].transpose(-1, -2)

        if vis is not None:
            for j in range(A.shape[0]):
                pt = src[0, :m, j]
                vis.draw_point(f"impt.{j}", pt, color=(0, 1, 0), length=0.003)
                if src_normals is not None:
                    vis.draw_2d_line(f"imn.{j}", pt, -src_normals[0, j], color=(0, 0.4, 0), size=2., scale=0.03)

        # check error
        mean_error = torch.mean(distances)
        err_list.append(mean_error.item())
        if torch.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform_torch(A.repeat(batch, 1, 1), src[:, :m, :].transpose(-2, -1))

    if vis is not None:
        # final evaluation
        src = torch.ones((m + 1, A.shape[0]), dtype=A.dtype, device=A.device)
        src[:m, :] = torch.clone(A.transpose(0, 1))
        src = src.repeat(batch, 1, 1)
        src = T @ src
        p = src[:, :m, :].transpose(-2, -1)
        q = dst[:, :, :m]
        distances, indices = nearest_neighbor_torch(p, q)
        # currently only have a single batch so flatten
        distances = distances.view(batch, -1)
        mean_error = torch.mean(distances)
        err_list.append(mean_error.item())
        if src_normals is not None and dst_normals is not None:
            # NOTE normally we need to multiply by the transpose of the inverse to transform normals, but since we are sure
            # the transform does not include scale, we can just use the matrix itself
            src_normals = A_normals.repeat(batch, 1, 1) @ T[:, :m, :m].transpose(-1, -2)

        for j in range(A.shape[0]):
            pt = src[0, :m, j]
            vis.draw_point(f"impt.{j}", pt, color=(0, 1, 0), length=0.003)
            if src_normals is not None:
                vis.draw_2d_line(f"imn.{j}", pt, -src_normals[0, j], color=(0, 0.4, 0), size=2., scale=0.03)
        for dist in err_list:
            print(dist)

    # convert to RMSE
    if distances is not None:
        distances = torch.sqrt(distances.square().sum(dim=1))
    return T, distances, i


class ICPPoseScore:
    """Score an ICP result on domain-specific plausibility; lower score is better.
    This function is designed for objects that manipulator robots will usually interact with"""

    def __init__(self, dim=3, upright_bias=.3, physics_bias=1., reject_proportion_on_score=.3):
        self.dim = dim
        self.upright_bias = upright_bias
        self.physics_bias = physics_bias
        self.reject_proportion_on_score = reject_proportion_on_score

    def __call__(self, T, all_points, icp_rmse):
        # T is matrix taking world frame to link frame
        # score T on rotation's distance away from prior of being upright
        if self.dim == 3:
            rot_axis = tf.matrix_to_axis_angle(T[..., :self.dim, :self.dim])
            rot_axis /= rot_axis.norm(dim=-1, keepdim=True)
            # project onto the z axis (take z component)
            # should be positive; upside down will be penalized and so will sideways
            upright_score = (1 - rot_axis[..., -1])
        else:
            upright_score = 0

        # inherent local minima quality from ICP
        fit_quality_score = icp_rmse if icp_rmse is not None else 0
        # should not float off the ground
        physics_score = all_points[:, :, 2].min(dim=1).values.abs()

        score = fit_quality_score + upright_score * self.upright_bias + physics_score * self.physics_bias

        # reject a portion of the input (assign them inf score) based on their quantile in terms of fit quality
        score_threshold = torch.quantile(score, 1 - self.reject_proportion_on_score)
        score[score > score_threshold] = float('inf')

        return score
