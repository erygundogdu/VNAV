from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from get_frame import DEFAULT_CAMERA_DIR, get_frame
from helper import orbmatch, ransac_estimate_essential, recover_pose, triangulate_points, solve_pnp
from bundle_adjustment import local_bundle_adjustment


USE_BUNDLE_ADJUSTMENT = True


def reprojection_errors(points3d, points2d, K, dist_coeffs, R, t):
    projected, _ = cv2.projectPoints(
        np.asarray(points3d, dtype=np.float64), cv2.Rodrigues(R)[0], t, K, dist_coeffs
    )
    return np.linalg.norm(projected.reshape(-1, 2) - points2d, axis=1)


def triangulate_new_points(matches, kp1, kp2, K, dist_coeffs, R, t):
    points, matches = triangulate_points(matches, kp1, kp2, K, R, t, dist_coeffs)
    if not matches:
        return points, []

    pixels1 = np.float64([kp1[m.queryIdx].pt for m in matches])
    pixels2 = np.float64([kp2[m.trainIdx].pt for m in matches])
    error1 = reprojection_errors(points, pixels1, K, dist_coeffs, np.eye(3), np.zeros((3, 1)))
    error2 = reprojection_errors(points, pixels2, K, dist_coeffs, R, t)
    rays1 = points / np.linalg.norm(points, axis=1, keepdims=True)
    rays2 = points - (-R.T @ t).ravel()
    rays2 /= np.linalg.norm(rays2, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(np.sum(rays1 * rays2, axis=1), -1, 1)))
    valid = (error1 < 2.0) & (error2 < 2.0) & (angles > 1.0)
    return points[valid], [m for m, keep in zip(matches, valid) if keep]


def estimate_trajectory(
    camera_dir=DEFAULT_CAMERA_DIR, start_index=0, end_index=None,
    use_bundle_adjustment=False, ba_window_size=5, ba_interval=10,
    return_poses=False,
):
    if use_bundle_adjustment and (ba_window_size < 3 or ba_interval < 1):
        raise ValueError('Use ba_window_size >= 3 and ba_interval >= 1')
    cv2.setRNGSeed(0)
    camera_dir = Path(camera_dir)
    with (camera_dir / 'sensor.yaml').open() as file:
        calibration = yaml.safe_load(file)
    fx, fy, cx, cy = calibration['intrinsics']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array(calibration['distortion_coefficients'], dtype=np.float64)
    with (camera_dir / 'data.csv').open() as file:
        frame_count = sum(1 for line in file if line.strip() and not line.startswith('#'))
    end_index = frame_count if end_index is None else min(end_index, frame_count)
    if start_index < 0 or end_index - start_index < 2:
        raise ValueError('Select at least two frames within the sequence')

    first_frame = get_frame(start_index, camera_dir)
    trajectory = [(0.0, 0.0, 0.0)]
    map_points = []
    orb = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    for second_index in range(start_index + 1, end_index):
        second_frame = get_frame(second_index, camera_dir)
        kp1, kp2, matches = orbmatch(first_frame, second_frame)
        matches = [m for m in matches if m.distance < 50]
        if len(matches) < 50:
            continue
        displacement = np.float64([
            np.array(kp2[m.trainIdx].pt) - kp1[m.queryIdx].pt for m in matches
        ])
        if np.median(np.linalg.norm(displacement, axis=1)) < 20.0:
            continue
        try:
            E = ransac_estimate_essential(matches, kp1, kp2, K, dist_coeffs)
            if E.shape != (3, 3):
                continue
            R_previous, t_previous = recover_pose(E, matches, kp1, kp2, K, dist_coeffs)
            points, valid_matches = triangulate_new_points(
                matches, kp1, kp2, K, dist_coeffs, R_previous, t_previous
            )
        except (ValueError, cv2.error):
            continue
        if len(points) < 50:
            continue

        map_points = [tuple(point) for point in points]
        _, descriptors = orb.compute(second_frame, kp2)
        map_descriptors = descriptors[[m.trainIdx for m in valid_matches]].copy()
        previous_observations = {
            kp2[m.trainIdx].pt: index for index, m in enumerate(valid_matches)
        }
        poses = {
            start_index: (np.eye(3), np.zeros((3, 1))),
            second_index: (R_previous.copy(), t_previous.copy()),
        }
        observations = {
            start_index: {i: kp1[m.queryIdx].pt for i, m in enumerate(valid_matches)},
            second_index: {i: kp2[m.trainIdx].pt for i, m in enumerate(valid_matches)},
        }
        previous_frame_index = second_index
        trajectory.append(tuple((-R_previous.T @ t_previous).ravel()))
        previous_frame = second_frame
        print(f'Initialized frames {start_index}, {second_index}: {len(map_points)} points')
        break
    else:
        raise RuntimeError('No frame pair with enough reliable triangulated points was found')

    for frame_index in range(second_index + 1, end_index):
        current_frame = get_frame(frame_index, camera_dir)
        kp_previous, kp_current, matches = orbmatch(previous_frame, current_frame)
        matches = [m for m in matches if m.distance < 50]
        _, current_descriptors = orb.compute(current_frame, kp_current)
        if current_descriptors is None or len(current_descriptors) < 2:
            print(f'Tracking lost at frame {frame_index}: too few ORB descriptors')
            break
        candidates = matcher.knnMatch(map_descriptors, current_descriptors, k=2)
        candidates = [
            pair[0] for pair in candidates
            if len(pair) == 2 and pair[0].distance < 50
            and pair[0].distance < 0.75 * pair[1].distance
        ]
        mapped_matches = []
        used_features = set()
        for match in sorted(candidates, key=lambda m: m.distance):
            if match.trainIdx not in used_features:
                mapped_matches.append(match)
                used_features.add(match.trainIdx)
        if len(mapped_matches) < 15:
            print(f'Tracking lost at frame {frame_index}: too few mapped matches')
            break

        point_ids = [m.queryIdx for m in mapped_matches]
        points3d = np.float64([map_points[index] for index in point_ids])
        points2d = np.float64([kp_current[m.trainIdx].pt for m in mapped_matches])
        try:
            R, t = solve_pnp(points3d, points2d, K, dist_coeffs)
        except (ValueError, cv2.error) as error:
            print(f'Tracking lost at frame {frame_index}: {error}')
            break

        errors = reprojection_errors(points3d, points2d, K, dist_coeffs, R, t)
        depths = (R @ points3d.T + t)[2]
        inliers = (errors < 3.0) & (depths > 0)
        if np.count_nonzero(inliers) < 15:
            print(f'Tracking lost at frame {frame_index}: too few PnP inliers')
            break

        success, rvec, refined_t = cv2.solvePnP(
            points3d[inliers], points2d[inliers], K, dist_coeffs,
            cv2.Rodrigues(R)[0], t.copy(), True, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            R, t = cv2.Rodrigues(rvec)[0], refined_t
        errors = reprojection_errors(points3d, points2d, K, dist_coeffs, R, t)
        inliers = (errors < 3.0) & ((R @ points3d.T + t)[2] > 0)

        current_observations = {
            kp_current[m.trainIdx].pt: point_id
            for m, point_id, keep in zip(mapped_matches, point_ids, inliers) if keep
        }
        new_matches = [
            m for m in matches
            if kp_previous[m.queryIdx].pt not in previous_observations
            and kp_current[m.trainIdx].pt not in current_observations
        ]
        R_relative = R @ R_previous.T
        t_relative = t - R_relative @ t_previous
        local_points, valid_matches = triangulate_new_points(
            new_matches, kp_previous, kp_current, K, dist_coeffs, R_relative, t_relative
        )
        for point, match in zip(local_points, valid_matches):
            world_point = R_previous.T @ (point.reshape(3, 1) - t_previous)
            point_id = len(map_points)
            current_observations[kp_current[match.trainIdx].pt] = point_id
            observations[previous_frame_index][point_id] = kp_previous[match.queryIdx].pt
            map_points.append(tuple(world_point.ravel()))
        if valid_matches:
            map_descriptors = np.vstack((
                map_descriptors, current_descriptors[[m.trainIdx for m in valid_matches]]
            ))

        poses[frame_index] = (R.copy(), t.copy())
        observations[frame_index] = {
            point_id: pixel for pixel, point_id in current_observations.items()
        }
        if use_bundle_adjustment and (frame_index - second_index) % ba_interval == 0:
            poses, map_points = local_bundle_adjustment(
                poses, map_points, observations, K, dist_coeffs,
                window_size=ba_window_size,
            )
            R, t = poses[frame_index]
            trajectory = [tuple((-pose_R.T @ pose_t).ravel()) for pose_R, pose_t in poses.values()]
        else:
            trajectory.append(tuple((-R.T @ t).ravel()))
        previous_frame = current_frame
        previous_frame_index = frame_index
        previous_observations = current_observations
        R_previous, t_previous = R, t
        if frame_index % 100 == 0:
            print(f'Frame {frame_index}: {len(trajectory)} poses, {len(map_points)} points')

    if return_poses:
        return trajectory, map_points, poses
    return trajectory, map_points


def visualize(trajectory, points3d):
    trajectory = np.asarray(trajectory)
    points3d = np.asarray(points3d)
    trajectory_axes = plt.figure().add_subplot(111, projection='3d')
    trajectory_axes.plot(*trajectory.T, color='red', label='Camera trajectory')
    trajectory_axes.scatter(*trajectory[0], color='green', label='Start')
    trajectory_axes.set_title('Camera trajectory (arbitrary scale)')

    points_axes = plt.figure().add_subplot(111, projection='3d')
    if points3d.size:
        points_axes.scatter(*points3d.T, s=1, alpha=0.3, label='3D points')
    points_axes.set_title('3D points (arbitrary scale)')

    for axes in (trajectory_axes, points_axes):
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        limits = np.array([axes.get_xlim3d(), axes.get_ylim3d(), axes.get_zlim3d()])
        centers = limits.mean(axis=1)
        radius = max(np.ptp(limits, axis=1).max() / 2, 1e-6)
        axes.set_xlim(centers[0] - radius, centers[0] + radius)
        axes.set_ylim(centers[1] - radius, centers[1] + radius)
        axes.set_zlim(centers[2] - radius, centers[2] + radius)
        axes.set_box_aspect((1, 1, 1))
        if axes.get_legend_handles_labels()[0]:
            axes.legend()
    plt.show()


if __name__ == '__main__':
    trajectory, points3d = estimate_trajectory(use_bundle_adjustment=USE_BUNDLE_ADJUSTMENT)
    visualize(trajectory, points3d)
