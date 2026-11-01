import cv2
import numpy as np


def orbmatch(image1, image2):
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    if descriptors1 is None or descriptors2 is None:
        return keypoints1, keypoints2, []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda match: match.distance)

    return keypoints1, keypoints2, matches

def ransac_estimate_essential(matches, keypoints1, keypoints2, K, dist_coeffs=None):
    if len(matches) < 5:
        raise ValueError("At least 5 matches are required")

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    K = np.asarray(K, dtype=np.float64)

    if dist_coeffs is not None:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        points1 = cv2.undistortPoints(points1.reshape(-1, 1, 2), K, dist_coeffs, P=K)
        points2 = cv2.undistortPoints(points2.reshape(-1, 1, 2), K, dist_coeffs, P=K)

    essential, _ = cv2.findEssentialMat(
        points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if essential is None:
        raise ValueError("Essential matrix estimation failed")

    return essential


def recover_pose(essential, matches, keypoints1, keypoints2, K, dist_coeffs=None):
    if len(matches) < 5:
        raise ValueError("At least 5 matches are required")

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    K = np.asarray(K, dtype=np.float64)

    if dist_coeffs is not None:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        points1 = cv2.undistortPoints(points1.reshape(-1, 1, 2), K, dist_coeffs, P=K)
        points2 = cv2.undistortPoints(points2.reshape(-1, 1, 2), K, dist_coeffs, P=K)

    inliers, R, t, _ = cv2.recoverPose(essential, points1, points2, K)
    if inliers == 0:
        raise ValueError("Pose recovery failed")

    return R, t


def triangulate_points(matches, keypoints1, keypoints2, K, R, t, dist_coeffs=None):
    if not matches:
        return np.empty((0, 3)), []

    points1 = np.float64([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float64([keypoints2[m.trainIdx].pt for m in matches])
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    if dist_coeffs is not None:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    points1 = cv2.undistortPoints(points1.reshape(-1, 1, 2), K, dist_coeffs)
    points2 = cv2.undistortPoints(points2.reshape(-1, 1, 2), K, dist_coeffs)
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))

    points4d = cv2.triangulatePoints(P1, P2, points1.reshape(-1, 2).T, points2.reshape(-1, 2).T)
    valid = np.abs(points4d[3]) > 1e-10
    points3d = np.full((len(matches), 3), np.nan)
    points3d[valid] = (points4d[:3, valid] / points4d[3, valid]).T
    points_camera2 = (R @ points3d.T + t).T
    valid &= np.isfinite(points3d).all(axis=1)
    valid &= (points3d[:, 2] > 0) & (points_camera2[:, 2] > 0)

    return points3d[valid], [m for m, keep in zip(matches, valid) if keep]


def solve_pnp(points3d, points2d, K, dist_coeffs=None):
    points3d = np.ascontiguousarray(points3d, dtype=np.float64).reshape(-1, 3)
    points2d = np.ascontiguousarray(points2d, dtype=np.float64).reshape(-1, 2)
    if len(points3d) != len(points2d) or len(points3d) < 4:
        raise ValueError("At least 4 corresponding 3D and 2D points are required")

    K = np.asarray(K, dtype=np.float64)
    if dist_coeffs is not None:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    success, rvec, t, inliers = cv2.solvePnPRansac(
        points3d, points2d, K, dist_coeffs,
        iterationsCount=100, reprojectionError=3.0, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 4:
        raise ValueError("PnP pose estimation failed")

    R, _ = cv2.Rodrigues(rvec)
    return R, t
