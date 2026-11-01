import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

from get_frame import DEFAULT_CAMERA_DIR
from main import USE_BUNDLE_ADJUSTMENT, estimate_trajectory


def load_ground_truth(camera_dir, frame_ids, max_gap_seconds=0.02):
    camera_dir = Path(camera_dir)
    with (camera_dir / 'data.csv').open() as file:
        camera_timestamps = np.array([
            int(row[0]) for row in csv.reader(file) if row and not row[0].startswith('#')
        ], dtype=np.int64)
    path = camera_dir.parent / 'state_groundtruth_estimate0' / 'data.csv'
    with path.open() as file:
        rows = [row for row in csv.reader(file) if row and not row[0].startswith('#')]
    gt_timestamps = np.array([int(row[0]) for row in rows], dtype=np.int64)
    values = np.array([row[1:8] for row in rows], dtype=np.float64)
    if np.any(np.diff(gt_timestamps) <= 0):
        raise ValueError('Ground-truth timestamps must be strictly increasing')

    timestamps = camera_timestamps[frame_ids]
    valid = (timestamps >= gt_timestamps[0]) & (timestamps <= gt_timestamps[-1])
    right = np.clip(np.searchsorted(gt_timestamps, timestamps, side='left'), 1, len(gt_timestamps) - 1)
    exact = timestamps == gt_timestamps[right]
    valid &= exact | ((gt_timestamps[right] - gt_timestamps[right - 1]) <= max_gap_seconds * 1e9)
    timestamps = timestamps[valid]
    if len(timestamps) < 3:
        raise ValueError('Fewer than 3 estimated poses overlap valid ground truth; process more frames')

    origin = gt_timestamps[0]
    gt_seconds = (gt_timestamps - origin).astype(np.float64) * 1e-9
    seconds = (timestamps - origin).astype(np.float64) * 1e-9
    body_positions = np.column_stack([
        np.interp(seconds, gt_seconds, values[:, axis]) for axis in range(3)
    ])
    body_rotations = Slerp(
        gt_seconds, Rotation.from_quat(values[:, [4, 5, 6, 3]])
    )(seconds).as_matrix()
    with (camera_dir / 'sensor.yaml').open() as file:
        T_BC = np.array(yaml.safe_load(file)['T_BS']['data']).reshape(4, 4)
    camera_positions = body_positions + np.einsum('nij,j->ni', body_rotations, T_BC[:3, 3])
    camera_rotations = body_rotations @ T_BC[:3, :3]
    return valid, timestamps, camera_positions, camera_rotations


def align_sim3(estimated, reference):
    estimated = np.asarray(estimated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    estimated_mean = estimated.mean(axis=0)
    reference_mean = reference.mean(axis=0)
    centered_estimated = estimated - estimated_mean
    centered_reference = reference - reference_mean
    variance = np.mean(np.sum(centered_estimated ** 2, axis=1))
    if variance < 1e-12:
        raise ValueError('Estimated trajectory has insufficient motion for scale alignment')
    U, singular_values, Vt = np.linalg.svd(centered_reference.T @ centered_estimated / len(estimated))
    correction = np.ones(3)
    correction[-1] = 1 if np.linalg.det(U @ Vt) > 0 else -1
    R = U @ np.diag(correction) @ Vt
    scale = np.dot(singular_values, correction) / variance
    if scale <= 0:
        raise ValueError('Cannot obtain a positive alignment scale')
    t = reference_mean - scale * R @ estimated_mean
    aligned = scale * (estimated @ R.T) + t
    return aligned, scale, R, t


def compute_metrics(estimated, reference, estimated_rotations, reference_rotations, timestamps):
    errors = np.linalg.norm(estimated - reference, axis=1)
    orientation_errors = Rotation.from_matrix(
        np.swapaxes(reference_rotations, 1, 2) @ estimated_rotations
    ).magnitude() * 180 / np.pi
    relative_seconds = (timestamps - timestamps[0]).astype(np.float64) * 1e-9
    gt_distance = np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()
    estimated_distance = np.linalg.norm(np.diff(estimated, axis=0), axis=1).sum()
    metrics = {
        'evaluated_poses': len(errors),
        'evaluated_duration_s': float(relative_seconds[-1]),
        'gt_path_length_m': float(gt_distance),
        'aligned_estimated_path_length_m': float(estimated_distance),
        'ate_rmse_m': float(np.sqrt(np.mean(errors ** 2))),
        'ate_mean_m': float(errors.mean()),
        'ate_median_m': float(np.median(errors)),
        'ate_std_m': float(errors.std()),
        'ate_p95_m': float(np.percentile(errors, 95)),
        'ate_max_m': float(errors.max()),
        'aligned_endpoint_error_m': float(errors[-1]),
        'orientation_rmse_deg': float(np.sqrt(np.mean(orientation_errors ** 2))),
    }
    translation_errors, rotation_errors = [], []
    for i, time in enumerate(relative_seconds):
        j = np.searchsorted(relative_seconds, time + 1.0)
        candidates = [k for k in [j - 1, j] if i < k < len(relative_seconds)]
        if not candidates:
            continue
        j = min(candidates, key=lambda k: abs(relative_seconds[k] - time - 1.0))
        if abs(relative_seconds[j] - time - 1.0) > 0.025:
            continue
        R_est = estimated_rotations[i].T @ estimated_rotations[j]
        R_gt = reference_rotations[i].T @ reference_rotations[j]
        t_est = estimated_rotations[i].T @ (estimated[j] - estimated[i])
        t_gt = reference_rotations[i].T @ (reference[j] - reference[i])
        translation_errors.append(np.linalg.norm(t_est - t_gt))
        rotation_errors.append(Rotation.from_matrix(R_gt.T @ R_est).magnitude() * 180 / np.pi)
    metrics['rpe_interval_s'] = 1.0
    metrics['rpe_pairs'] = len(translation_errors)
    metrics['rpe_translation_rmse_m'] = (
        float(np.sqrt(np.mean(np.square(translation_errors)))) if translation_errors else None
    )
    metrics['rpe_rotation_rmse_deg'] = (
        float(np.sqrt(np.mean(np.square(rotation_errors)))) if rotation_errors else None
    )
    return metrics, errors


def plot_comparison(estimated, reference, points3d, timestamps, errors, output_dir, label):
    figure = plt.figure(figsize=(9, 7))
    axes = figure.add_subplot(111, projection='3d')
    axes.plot(*reference.T, color='black', label='Ground truth cam0')
    axes.plot(*estimated.T, color='red', label='Estimated, Sim(3) aligned')
    axes.scatter(*reference[0], color='green', label='GT evaluation start')
    axes.set_title(f'Camera trajectory: {label}')
    axes.legend()
    trajectory_figure = figure

    figure = plt.figure(figsize=(9, 7))
    points_axes = figure.add_subplot(111, projection='3d')
    if len(points3d):
        points_axes.scatter(*points3d.T, s=1, alpha=0.3)
    points_axes.set_title(f'Reconstructed points in GT coordinates: {label}')
    for axis in (axes, points_axes):
        axis.set_xlabel('X (m)')
        axis.set_ylabel('Y (m)')
        axis.set_zlabel('Z (m)')
        limits = np.array([axis.get_xlim3d(), axis.get_ylim3d(), axis.get_zlim3d()])
        centers = limits.mean(axis=1)
        radius = max(np.ptp(limits, axis=1).max() / 2, 1e-6)
        axis.set_xlim(centers[0] - radius, centers[0] + radius)
        axis.set_ylim(centers[1] - radius, centers[1] + radius)
        axis.set_zlim(centers[2] - radius, centers[2] + radius)
        axis.set_box_aspect((1, 1, 1))
    trajectory_figure.savefig(output_dir / 'trajectory_3d.png', dpi=180, bbox_inches='tight')
    figure.savefig(output_dir / 'points_3d.png', dpi=180, bbox_inches='tight')

    figure, axes = plt.subplots()
    seconds = (timestamps - timestamps[0]).astype(np.float64) * 1e-9
    axes.plot(seconds, errors)
    axes.set_xlabel('Time since first evaluated pose (s)')
    axes.set_ylabel('Aligned position error (m)')
    axes.set_title(f'Absolute trajectory error: {label}')
    axes.grid(True)
    figure.savefig(output_dir / 'position_error.png', dpi=180, bbox_inches='tight')


def run_comparison(args):
    trajectory, points3d, poses = estimate_trajectory(
        camera_dir=args.camera_dir, start_index=args.start_frame, end_index=args.end_frame,
        use_bundle_adjustment=args.ba, return_poses=True,
    )
    frame_ids = np.array(list(poses), dtype=int)
    valid, timestamps, reference, reference_rotations = load_ground_truth(args.camera_dir, frame_ids)
    estimated = np.asarray(trajectory)[valid]
    estimated_rotations = np.array([R.T for R, _ in poses.values()])[valid]
    aligned, scale, alignment_R, alignment_t = align_sim3(estimated, reference)
    geometry_ratios = []
    for positions in (estimated, reference):
        singular_values = np.linalg.svd(positions - positions.mean(axis=0), compute_uv=False)
        geometry_ratios.append(float(singular_values[1] / max(singular_values[0], 1e-12)))
    aligned_rotations = alignment_R @ estimated_rotations
    metrics, errors = compute_metrics(aligned, reference, aligned_rotations, reference_rotations, timestamps)
    metrics.update({
        'bundle_adjustment': args.ba,
        'estimated_poses': len(trajectory),
        'excluded_without_gt': int(np.count_nonzero(~valid)),
        'first_tracked_frame': int(frame_ids[0]),
        'last_tracked_frame': int(frame_ids[-1]),
        'first_evaluated_frame': int(frame_ids[valid][0]),
        'last_evaluated_frame': int(frame_ids[valid][-1]),
        'scale_m_per_estimated_unit': float(scale),
        'alignment_rotation': alignment_R.tolist(),
        'alignment_translation_m': alignment_t.tolist(),
        'alignment': 'One global Sim(3), fitted to all timestamp-matched camera positions',
        'gt_interpolation': 'Linear body position and quaternion SLERP; no extrapolation; maximum GT gap 0.02 s',
        'timestamp_offset_s': 0.0,
        'alignment_nearly_collinear': min(geometry_ratios) < 0.05,
    })
    requested_end = args.end_frame
    with (Path(args.camera_dir) / 'data.csv').open() as file:
        total_frames = sum(1 for line in file if line.strip() and not line.startswith('#'))
    requested_end = total_frames if requested_end is None else min(requested_end, total_frames)
    metrics['last_requested_frame'] = requested_end - 1
    metrics['tracking_lost'] = int(frame_ids[-1]) < requested_end - 1
    label = 'BA on' if args.ba else 'BA off'
    output_dir = Path(args.output_dir) / ('ba_on' if args.ba else 'ba_off')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    with (output_dir / 'trajectory.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp_ns', 'frame_id', 'est_x_m', 'est_y_m', 'est_z_m',
                         'gt_x_m', 'gt_y_m', 'gt_z_m', 'position_error_m'])
        for timestamp, frame, position, gt, error in zip(timestamps, frame_ids[valid], aligned, reference, errors):
            writer.writerow([int(timestamp), int(frame), *position, *gt, error])
    print(f'\nGround-truth comparison ({label})')
    print(f'Tracked frames: {frame_ids[0]} to {frame_ids[-1]}; tracking lost: {metrics["tracking_lost"]}')
    print(f'Evaluated: {len(timestamps)} poses over {metrics["evaluated_duration_s"]:.2f} s')
    print(f'Excluded {metrics["excluded_without_gt"]} poses outside valid GT coverage')
    print(f'Sim(3) scale: {scale:.6f} m / estimated unit')
    print('GT: cam0 extrinsics applied; timestamp offset 0 s; interpolation without extrapolation')
    print(f'GT distance: {metrics["gt_path_length_m"]:.3f} m')
    print(f'ATE position: RMSE {metrics["ate_rmse_m"]:.4f} m, median {metrics["ate_median_m"]:.4f} m, max {metrics["ate_max_m"]:.4f} m')
    print(f'Orientation RMSE: {metrics["orientation_rmse_deg"]:.3f} deg')
    if metrics['rpe_pairs']:
        print(f'RPE (1 s, {metrics["rpe_pairs"]} pairs): {metrics["rpe_translation_rmse_m"]:.4f} m, {metrics["rpe_rotation_rmse_deg"]:.3f} deg')
    else:
        print('RPE (1 s): insufficient overlapping duration')
    if np.linalg.norm(np.std(reference, axis=0)) < 0.1:
        print('Limited motion: alignment and scale may be poorly constrained')
    if metrics['alignment_nearly_collinear']:
        print('Nearly collinear path: alignment rotation and absolute orientation error may be unstable')
    print(f'Results: {output_dir.resolve()}')
    aligned_points = scale * (np.asarray(points3d) @ alignment_R.T) + alignment_t
    plot_comparison(aligned, reference, aligned_points, timestamps, errors, output_dir, label)
    if not args.no_show:
        plt.show()
    else:
        plt.close('all')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate monocular cam0 trajectory against EuRoC ground truth')
    parser.add_argument('--camera-dir', type=Path, default=DEFAULT_CAMERA_DIR)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--ba', dest='ba', action='store_true')
    mode.add_argument('--no-ba', dest='ba', action='store_false')
    parser.set_defaults(ba=USE_BUNDLE_ADJUSTMENT)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'evaluation')
    parser.add_argument('--no-show', action='store_true')
    run_comparison(parser.parse_args())
