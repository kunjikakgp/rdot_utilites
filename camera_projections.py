import cv2
import numpy as np
elev=60.0

def fisheye_unwrap_map(src_w, src_h, lens_fov_deg = 200.0, camera_yaw_offset_deg = 90.0):

    cx = src_w / 2.0
    cy = src_h / 2.0
    r_valid = min(src_w, src_h) / 2.0

    dst_h = int(r_valid)
    dst_w = int(2 * r_valid)

    half_fov_rad = np.deg2rad(lens_fov_deg / 2.0)
    f = r_valid / half_fov_rad

    az_min = -np.pi
    az_max = np.pi

    elevation_min_deg = (180 - lens_fov_deg) / 2
    elevation_max_deg = 90

    el_min = np.deg2rad(elevation_min_deg)
    el_max = np.deg2rad(elevation_max_deg)

    yaw_offset = np.deg2rad(camera_yaw_offset_deg)
    
    # Normalized output coordinates
    u = np.linspace(0.0, 1.0, dst_w, dtype=np.float32)
    v = np.linspace(0.0, 1.0, dst_h, dtype=np.float32)

    # x controls azimuth, y controls elevation
    az = az_min + u * (az_max - az_min)
    el = el_max - v * (el_max - el_min)

    az_grid, el_grid = np.meshgrid(az, el)
    az_cam = az_grid + yaw_offset

    cos_e = np.cos(el_grid)
    sin_e = np.sin(el_grid)

    # 3D camera rays
    Xc = cos_e * np.cos(az_cam)
    Yc = cos_e * np.sin(az_cam)
    Zc = sin_e

    rho = np.sqrt(Xc * Xc + Yc * Yc)
    theta = np.arctan2(rho, Zc)

    valid = theta <= half_fov_rad

    # Equidistant fisheye projection
    r = f * theta

    map_x = np.full((dst_h, dst_w), -1, dtype=np.float32)
    map_y = np.full((dst_h, dst_w), -1, dtype=np.float32)

    eps = 1e-9
    safe_rho = np.maximum(rho, eps)

    src_x = cx + r * Xc / safe_rho
    src_y = cy - r * Yc / safe_rho

    # Optical-axis special case
    center_mask = rho < eps
    src_x = np.where(center_mask, cx, src_x)
    src_y = np.where(center_mask, cy, src_y)

    # Also reject projections outside the input image
    valid &= src_x >= 0
    valid &= src_x < src_w
    valid &= src_y >= 0
    valid &= src_y < src_h

    map_x[valid] = src_x[valid].astype(np.float32)
    map_y[valid] = src_y[valid].astype(np.float32)

    return map_x, map_y

def fisheye_perspective_map(src_w, src_h, lens_fov_deg = 200.0, perspective_hfov_deg = 120.0, view_azimuth_deg = 0.0, camera_yaw_offset_deg = 90.0):
    cx = src_w / 2.0
    cy = src_h / 2.0
    r_valid = min(src_w, src_h) / 2.0

    half_fov_rad = np.deg2rad(lens_fov_deg / 2.0)
    f_fisheye = r_valid / half_fov_rad

    elevation_min_deg = (180.0 - lens_fov_deg) / 2.0
    elevation_max_deg = elev

    perspective_vfov_deg = elevation_max_deg - elevation_min_deg
    center_elevation_deg = 0.5 * (elevation_min_deg + elevation_max_deg)

    out_h = int(src_h)

    vfov = np.deg2rad(perspective_vfov_deg)
    hfov = np.deg2rad(perspective_hfov_deg)

    fy = (out_h / 2.0) / np.tan(vfov / 2.0)
    fx = fy

    out_w = int(round(2.0 * fx * np.tan(hfov / 2.0)))

    px = out_w / 2.0
    py = out_h / 2.0

    u, v = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))

    # Perspective-local rays:
    # x = right, y = up, z = forward
    x = (u - px) / fx
    y = -(v - py) / fy
    z = np.ones_like(x, dtype=np.float32)

    norm = np.sqrt(x * x + y * y + z * z)
    x /= norm
    y /= norm
    z /= norm

    az_cam_deg = view_azimuth_deg + camera_yaw_offset_deg

    az = np.deg2rad(az_cam_deg)
    el = np.deg2rad(center_elevation_deg)

    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], dtype = np.float32)

    right = np.array([-np.sin(az), np.cos(az), 0.0], dtype=np.float32)
    right /= np.linalg.norm(right)

    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    Xc = x * right[0] + y * up[0] + z * forward[0]
    Yc = x * right[1] + y * up[1] + z * forward[1]
    Zc = x * right[2] + y * up[2] + z * forward[2]

    rho = np.sqrt(Xc * Xc + Yc * Yc)
    theta = np.arctan2(rho, Zc)

    valid = theta <= half_fov_rad

    r = f_fisheye * theta

    eps = 1e-9
    safe_rho = np.maximum(rho, eps)

    src_x = cx + r * Xc / safe_rho
    src_y = cy - r * Yc / safe_rho

    center_mask = rho < eps
    src_x = np.where(center_mask, cx, src_x)
    src_y = np.where(center_mask, cy, src_y)

    valid &= src_x >= 0
    valid &= src_x < src_w
    valid &= src_y >= 0
    valid &= src_y < src_h

    map_x = np.full((out_h, out_w), -1, dtype=np.float32)
    map_y = np.full((out_h, out_w), -1, dtype=np.float32)

    map_x[valid] = src_x[valid].astype(np.float32)
    map_y[valid] = src_y[valid].astype(np.float32)

    #Return the intrinsic
    K = np.array([
    [fx, 0.0, px],
    [0.0, fy, py],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

    return map_x, map_y,K


def fisheye_remap(frame, map_x, map_y, border_value=(0, 0, 0)):
    """
    Apply precomputed fisheye transformation.
    """
    return cv2.remap(frame, map_x, map_y, interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = border_value)