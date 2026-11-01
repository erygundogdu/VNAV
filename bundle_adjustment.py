from collections import Counter

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def local_bundle_adjustment(
    poses, points3d, observations, K, dist_coeffs=None,
    window_size=5, max_points=500, max_nfev=20,
):
    if window_size < 3 or max_points < 1 or max_nfev < 1:
        raise ValueError('Use window_size >= 3 and positive optimization limits')

    updated_poses = {
        frame: (np.asarray(R, dtype=np.float64).copy(),
                np.asarray(t, dtype=np.float64).reshape(3, 1).copy())
        for frame, (R, t) in poses.items()
    }
    updated_points = [tuple(point) for point in points3d]
    frames = sorted(poses)[-window_size:]
    if len(frames) < 3:
        return updated_poses, updated_points

    counts = Counter(
        point_id for frame in frames for point_id in observations.get(frame, {})
    )
    point_ids = [point for point, count in counts.most_common() if count >= 2][:max_points]
    if not point_ids:
        return updated_poses, updated_points

    point_lookup = {point: index for index, point in enumerate(point_ids)}
    variable_frames = frames[2:]
    frame_lookup = {frame: index for index, frame in enumerate(variable_frames)}
    measurements = {
        frame: [(point_lookup[p], pixel) for p, pixel in observations.get(frame, {}).items()
                if p in point_lookup]
        for frame in frames
    }
    boundary_counts = Counter()
    for frame in reversed(sorted(f for f in poses if f < frames[0])):
        boundary = [
            (point_lookup[p], pixel) for p, pixel in observations.get(frame, {}).items()
            if p in point_lookup and boundary_counts[p] < 2
        ]
        if boundary:
            measurements[frame] = boundary
            boundary_counts.update(point_ids[index] for index, _ in boundary)

    camera_parameters = np.array([
        np.r_[cv2.Rodrigues(updated_poses[frame][0])[0].ravel(), updated_poses[frame][1].ravel()]
        for frame in variable_frames
    ])
    point_parameters = np.float64([points3d[p] for p in point_ids])
    camera_size = camera_parameters.size
    initial = np.r_[camera_parameters.ravel(), point_parameters.ravel()]
    K = np.asarray(K, dtype=np.float64)
    if dist_coeffs is not None:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    groups = []
    for frame, entries in measurements.items():
        if entries:
            indices = np.array([index for index, _ in entries])
            pixels = np.float64([pixel for _, pixel in entries])
            groups.append((frame, indices, pixels))

    sparsity = lil_matrix((sum(2 * len(indices) for _, indices, _ in groups), len(initial)))
    offset = 0
    for frame, indices, _ in groups:
        for index in indices:
            if frame in frame_lookup:
                column = 6 * frame_lookup[frame]
                sparsity[offset:offset + 2, column:column + 6] = 1
            column = camera_size + 3 * index
            sparsity[offset:offset + 2, column:column + 3] = 1
            offset += 2

    def residuals(parameters):
        cameras = parameters[:camera_size].reshape(-1, 6)
        points = parameters[camera_size:].reshape(-1, 3)
        errors = []
        for frame, indices, pixels in groups:
            if frame in frame_lookup:
                camera = cameras[frame_lookup[frame]]
                rvec, t = camera[:3], camera[3:]
            else:
                R, t = updated_poses[frame]
                rvec = cv2.Rodrigues(R)[0]
            projected, _ = cv2.projectPoints(points[indices], rvec, t, K, dist_coeffs)
            errors.append((projected.reshape(-1, 2) - pixels).ravel())
        return np.concatenate(errors)

    initial_errors = residuals(initial)
    result = least_squares(
        residuals, initial, jac_sparsity=sparsity.tocsr(), method='trf',
        x_scale='jac', loss='soft_l1', f_scale=2.0,
        max_nfev=max_nfev, ftol=1e-4,
    )
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
        return updated_poses, updated_points
    if np.sum(result.fun ** 2) >= np.sum(initial_errors ** 2):
        return updated_poses, updated_points

    cameras = result.x[:camera_size].reshape(-1, 6)
    points = result.x[camera_size:].reshape(-1, 3)
    candidate_poses = updated_poses.copy()
    for frame, camera in zip(variable_frames, cameras):
        candidate_poses[frame] = (cv2.Rodrigues(camera[:3])[0], camera[3:].reshape(3, 1).copy())
    for frame, indices, _ in groups:
        R, t = candidate_poses[frame]
        if np.any((R @ points[indices].T + t)[2] <= 0):
            return updated_poses, updated_points

    for point_id, point in zip(point_ids, points):
        updated_points[point_id] = tuple(point)
    before = np.sqrt(np.mean(initial_errors ** 2))
    after = np.sqrt(np.mean(result.fun ** 2))
    print(f'Local BA: reprojection RMSE {before:.3f} -> {after:.3f} px')
    return candidate_poses, updated_points
