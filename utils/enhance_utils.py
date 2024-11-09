import os
import numpy as np
import json
import open3d as o3d
from PIL import Image
from plyfile import PlyData
from utils.sh_utils import SH2RGB

SAGA_FEATURE_DIM = 32


def project_3d_to_2d(pts, w2c, K, return_dists=False, return_depth=False):
    """Project 3D points to 2D (nerfstudio format)."""
    pts = np.array(pts)
    K = np.hstack([K, np.zeros((3, 1))])
    pts = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    pts = np.dot(pts, w2c.T)
    pts[:, [1, 2]] *= -1
    if return_dists:
        dists = np.linalg.norm(pts[:, :3], axis=-1)
    if return_depth:
        depths = pts[:, 2]
    pts = np.dot(pts, K.T)
    pts_2d = pts[:, :2] / pts[:, 2:]
    if return_dists:
        return pts_2d, dists
    if return_depth:
        return pts_2d, depths
    return pts_2d


def parse_transforms_json(t_file, return_w2c=False, different_Ks=False):
    with open(t_file, "rb") as f:
        transforms = json.load(f)

    if different_Ks:
        Ks = []
        for i in range(len(transforms["frames"])):
            K = np.array(
                [
                    [transforms["frames"][i]["fl_x"], 0, transforms["frames"][i]["cx"]],
                    [0, transforms["frames"][i]["fl_y"], transforms["frames"][i]["cy"]],
                    [0, 0, 1],
                ]
            )
            Ks.append(K)
        K = Ks
    else:
        K = np.array(
            [
                [transforms["fl_x"], 0, transforms["cx"]],
                [0, transforms["fl_y"], transforms["cy"]],
                [0, 0, 1],
            ]
        )

    n_frames = len(transforms["frames"])
    c2ws = [
        np.array(transforms["frames"][i]["transform_matrix"]) for i in range(n_frames)
    ]
    if return_w2c:
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        return w2cs, K
    return c2ws, K


def read_saga_ply_file(pcd_file):
    pcd_data = PlyData.read(pcd_file)
    vertices = pcd_data["vertex"]
    xyz = np.array([vertices["x"], vertices["y"], vertices["z"]]).T
    saga_features = np.array([vertices[f"f_{i}"] for i in range(SAGA_FEATURE_DIM)]).T
    scales = np.array([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).T
    opacitys = np.array(vertices['opacity'])
    return xyz, saga_features, scales, opacitys


def read_gaussian_ply_file(pcd_file):
    pcd_data = PlyData.read(pcd_file)
    vertices = pcd_data["vertex"]
    xyz = np.array([vertices["x"], vertices["y"], vertices["z"]]).T
    f_dc = np.array([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).T
    colors = SH2RGB(f_dc)
    scales = np.array([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).T
    opacitys = np.array(vertices['opacity'])
    return xyz, colors, scales, opacitys


def bounding_box_size(xyz):
    min_coords = np.min(xyz, axis=0)
    max_coords = np.max(xyz, axis=0)
    bbox_size = np.linalg.norm(max_coords - min_coords)
    return bbox_size


def bounding_box_volume(xyz):
    min_coords = np.min(xyz, axis=0)
    max_coords = np.max(xyz, axis=0)
    bbox_volume = np.prod(max_coords-min_coords)
    return bbox_volume


def load_saga_point_cloud(pcd_file, ds_size=0.01, return_gaussian_property=False):
    xyz, saga_features, scales, opacitys = read_saga_ply_file(pcd_file)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    downsampled_pcd = pcd
    if ds_size is not None:
        downsampled_pcd = pcd.voxel_down_sample(ds_size)
        downsampled_indices = np.asarray(downsampled_pcd.points)
        tree = o3d.geometry.KDTreeFlann(pcd)
        indices = []
        for point in downsampled_indices:
            _, idx, _ = tree.search_knn_vector_3d(point, 1)
            indices.append(idx[0])
        saga_features = saga_features[indices]
        scales = scales[indices]
        opacitys = opacitys[indices]
        xyz = xyz[indices]
    if return_gaussian_property:
        return xyz, saga_features, scales, opacitys
    return downsampled_pcd, saga_features


def load_gaussian_point_cloud(pcd_file, ds_size=0.01):
    xyz, colors, scales, opacitys = read_gaussian_ply_file(pcd_file)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    downsampled_pcd = pcd
    if ds_size is not None:
        downsampled_pcd = pcd.voxel_down_sample(ds_size)
        downsampled_indices = np.asarray(downsampled_pcd.points)
        tree = o3d.geometry.KDTreeFlann(pcd)
        indices = []
        for point in downsampled_indices:
            _, idx, _ = tree.search_knn_vector_3d(point, 1)
            indices.append(idx[0])
        colors = colors[indices]
        scales = scales[indices]
        opacitys = opacitys[indices]
        xyz = xyz[indices]
    return xyz, colors, scales, opacitys


def load_images(img_dir, bg_change=255, return_masks=False):
    img_files = os.listdir(img_dir)
    img_files.sort()
    imgs = []
    masks = []
    for img_file in img_files:
        # load RGBA image
        img = np.array(Image.open(os.path.join(img_dir, img_file)))
        if return_masks or bg_change is not None:
            mask = img[:, :, 3] > 0
            if bg_change is not None:
                img[~mask] = bg_change
            masks.append(mask)
        imgs.append(img[:, :, :3])

    if return_masks:
        return imgs, masks
    return imgs


def get_last_file_in_folder(folder):
    files = os.listdir(folder)
    return os.path.join(folder, sorted(files, reverse=True)[0])


def get_scenes_list(args):
    if args.split != "all":
        with open(os.path.join(args.data_dir, "splits.json"), "r") as f:
            splits = json.load(f)
        if args.split == "train+val":
            scenes = splits["train"] + splits["val"]
        else:
            scenes = splits[args.split]
    else:
        scenes = sorted(os.listdir(os.path.join(args.data_dir, "scenes")))

    if args.end_idx != -1:
        scenes = scenes[args.start_idx : args.end_idx]
    else:
        scenes = scenes[args.start_idx :]
    return scenes


def unproject_point(pt_2d, depth, c2w, K):
    """Unproject a single point from 2D to 3D (nerfstudio format)."""
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    x = (pt_2d[0] - cx) / fx
    y = (pt_2d[1] - cy) / fy
    pt_3d = np.array([x, -y, -1])
    pt_3d *= depth[pt_2d[1], pt_2d[0]]
    pt_3d = np.concatenate([pt_3d, np.ones((1,))], axis=0)
    pt_3d = np.dot(c2w, pt_3d)
    pt_3d = pt_3d[:3]
    return pt_3d


def point_to_distance_map(pts, w2c, K, width, height):
    pts = np.array(pts)
    K = np.hstack([K, np.zeros((3, 1))])
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_cam = np.dot(pts_hom, w2c.T)
    pts_cam[:, [1, 2]] *= -1
    pts_cam = pts_cam[pts_cam[:, 2] > 0]

    distances = np.linalg.norm(pts_cam[:, :3], axis=1)
    pts_2d_hom = np.dot(pts_cam, K.T)
    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:]

    distance_image = np.full((height, width), np.inf, dtype=np.float32)

    for i in range(pts_2d.shape[0]):
        x, y = int(pts_2d[i, 0]), int(pts_2d[i, 1])
        if 0 <= x < width and 0 <= y < height:
            distance_image[y, x] = min(distance_image[y, x], distances[i])

    distance_image[distance_image == np.inf] = 0

    return distance_image


def filter_random_init_points(xyz, path):
    w2cs, K = parse_transforms_json(
        os.path.join(path, "transforms.json"), return_w2c=True
    )
    imgs, masks = load_images(os.path.join(path, "images"), return_masks=True)

    filtered_indices = []
    xyz_indices = np.arange(len(xyz))
    for mask, w2c in zip(masks, w2cs):
        pts_2d, depths = project_3d_to_2d(xyz, w2c, K, return_depth=True)

        height, width = mask.shape[:2]
        valid_pts_mask = (
            (pts_2d[:, 0] >= 0)
            & (pts_2d[:, 0] < width)
            & (pts_2d[:, 1] >= 0)
            & (pts_2d[:, 1] < height)
        )
        valid_pts, valid_indices = xyz[valid_pts_mask], xyz_indices[valid_pts_mask]
        valid_project_pts, valid_project_depths = (
            pts_2d[valid_pts_mask],
            depths[valid_pts_mask],
        )

        for idx, point, proj_point, depth in zip(
            valid_indices, valid_pts, valid_project_pts, valid_project_depths
        ):
            x, y = int(proj_point[0]), int(proj_point[1])
            if mask[y, x] and depth > 0:
                filtered_indices.append(idx)
    filtered_points = xyz[np.isin(xyz_indices, filtered_indices)]

    return filtered_points


def get_initial_sacle(metadata_json_path, scene_name):
    scene_name = scene_name.split('_')[0]
    with open(metadata_json_path, 'r') as f:
        metadata = json.load(f)
    item_dimensions = metadata[scene_name]['item_dimensions']
    assert len(item_dimensions) == 3
    return sum(item_dimensions) / len(item_dimensions) / 100
