import os
import json
import torch
import cv2
import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image
from sklearn.decomposition import PCA
from plyfile import PlyData
from utils.enhance_utils import parse_transforms_json
from utils.sh_utils import SH2RGB
from settings import get_args


def load_saga_feature(ply_data):
    vertices = ply_data['vertex']
    features = [vertices[f'f_{i}'] for i in range(32)]
    return np.array(features).T


def load_saga_xyz(ply_data):
    vertices = ply_data['vertex']
    xyz = [vertices['x'], vertices['y'], vertices['z']]
    return np.array(xyz).T


def load_gaussian_color(ply_data):
    vertices = ply_data['vertex']
    f_dc = np.array([vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']]).T
    return SH2RGB(f_dc)


def features_to_colors(features):
    """Convert feature vectors to RGB colors using PCA."""
    pca = PCA(n_components=3)
    pca.fit(features)
    transformed = pca.transform(features)
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    transformed = (transformed - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    colors = np.clip(transformed, 0, 1)
    return colors


def values_to_colors(values, low=500, high=3500):
    """Convert scalar values to RGB colors."""
    cmap = mpl.colormaps['inferno']
    colors = cmap((values - low) / (high - low))
    return colors[:, :3]


def plot_color_gradient(low=500, high=3500):
    values = np.linspace(low, high, 100).reshape(1, -1)

    cmap = plt.get_cmap('inferno')
    colors = cmap((values - low) / (high - low))
    plt.figure(figsize=(16, 1))
    plt.imshow(colors, aspect='auto')

    plt.axis('off')
    plt.show()


def similarities_to_colors(similarities, temperature=None):
    """Convert CLIP similarity values to RGB colors."""
    cmap = mpl.colormaps['tab10']
    mat_colors = [cmap(i)[:3] for i in range(similarities.shape[1])]
    if temperature is None:
        argmax_similarities = np.argmax(similarities, axis=1)
        colors = np.array([mat_colors[i] for i in argmax_similarities])
    else:
        softmax_probs = torch.softmax(torch.tensor(similarities) / temperature, dim=1)
        colors = softmax_probs @ torch.tensor(mat_colors).float()
        colors = colors.numpy()
    return colors


def render_pcd(pcd, w2c, K, hw=(1024, 1024), pt_size=8, title=None, savefile=None, show=False):
    h, w = hw

    # set pinhole camera parameters from K
    render_camera = o3d.camera.PinholeCameraParameters()
    render_camera.extrinsic = w2c

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(h, w, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    render_camera.intrinsic = intrinsic

    # visualize pcd from camera view with intrinsics set to K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=show, window_name=title)

    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(render_camera, allow_arbitrary=True)

    # rendering options
    render_option = vis.get_render_option()
    render_option.point_size = pt_size
    render_option.point_show_normal = False
    render_option.light_on = False
    vis.update_renderer()

    if show:
        vis.run()

    if savefile is not None:
        vis.capture_screen_image(savefile, do_render=True)
        vis.destroy_window()
        return Image.open(savefile)
    else:
        render = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        return np.array(render)


def render_video(pcd, w2c, K, hw=(1024, 1024), center=np.array([0, 0, 0]), scene_dir=None, video_name=None):
    h, w = hw
    # set initial camera parameters
    render_camera = o3d.camera.PinholeCameraParameters()
    render_camera.extrinsic = w2c
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(h, w, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    render_camera.intrinsic = intrinsic

    # render
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h)
    vis.add_geometry(pcd)
    # view control
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(render_camera, allow_arbitrary=True)

    video_dir = os.path.join(scene_dir, 'vis_result')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    video_path = os.path.join(video_dir, f'{video_name}.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))

    distance = 5.0
    num_frames = 120
    initial_camera_position = w2c[:3, 3]

    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        # camera position in circular track
        cam_x = initial_camera_position[0] + distance * np.cos(angle)
        cam_y = initial_camera_position[1] + distance * np.sin(angle)
        cam_z = initial_camera_position[2]
        # set the camera position and lookat the center
        ctr.set_lookat(center)
        ctr.set_up([0, 0, 1])
        ctr.set_front([cam_x - center[0], cam_y - center[1], cam_z - center[2]])

        # render and save
        vis.poll_events()
        vis.update_renderer()

        image = vis.capture_screen_float_buffer(False)
        image = (np.asarray(image) * 255).astype(np.uint8)
        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    vis.destroy_window()
    video_writer.release()
    print(f"save {video_path} successfully...")


def render_video_with_track(pcd, w2c_list, K, hw=(1024, 1024), scene_dir=None, video_name=None):
    h, w = hw
    # video save path
    video_dir = os.path.join(scene_dir, 'vis_result')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    video_path = os.path.join(video_dir, f'{video_name}.mp4')
    # video params
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 60, (w, h))
    # render in open3d
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h)
    vis.add_geometry(pcd)
    # intrinsic of camera
    render_camera = o3d.camera.PinholeCameraParameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(h, w, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    render_camera.intrinsic = intrinsic
    # view control
    ctr = vis.get_view_control()

    for w2c in w2c_list:
        # set extrinsic
        render_camera.extrinsic = w2c
        ctr.convert_from_pinhole_camera_parameters(render_camera, allow_arbitrary=True)
        # render and save frame
        vis.poll_events()
        vis.update_renderer()

        image = vis.capture_screen_float_buffer(False)
        image = (np.asarray(image) * 255).astype(np.uint8)
        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    vis.destroy_window()
    video_writer.release()
    print(f"save {video_path} successfully...")


@torch.no_grad()
def get_agg_patch_features(patch_features, is_visible):
    """Get aggregated patch features by averaging over visible patches."""
    n_visible = is_visible.sum(0)
    is_valid = n_visible > 0

    visible_patch_features = patch_features * is_visible.unsqueeze(-1)
    avg_visible_patch_features = visible_patch_features.sum(0) / n_visible.unsqueeze(-1)
    avg_visible_patch_features = avg_visible_patch_features / avg_visible_patch_features.norm(dim=1, keepdim=True)
    return avg_visible_patch_features[is_valid], is_valid


def composite_and_save(img1, img2, alpha, savefile):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img = img1 * alpha + img2 * (1 - alpha)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(savefile)
    return img


def parse_materials_str(material_json_path, material_key='candidate_materials_density'):
    with open(material_json_path, 'r') as f:
        material_json = json.load(f)
    material_str = material_json[material_key]
    seg_list = material_str.split(';')
    return [item.split(':')[0][1:] for item in seg_list]


def make_legend(colors, names, ncol=1, figsize=(2.0, 2.5), savefile=None, show=False):
    plt.style.use('fast')
    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    plt.axis('off')

    # creating legend with color boxes
    ptchs = []
    for color, name in zip(colors, names):
        if len(name) > 10:  # wrap long names
            name = name.replace(' ', '\n')
        ptchs.append(mpatches.Patch(color=color[:3], label=name))
    leg = plt.legend(handles=ptchs, ncol=ncol, loc='center left', prop={'size': 18},
                     handlelength=1, handleheight=1, facecolor='white', framealpha=0)
    plt.tight_layout()

    if show:
        plt.show()
    if savefile is not None:
        plt.savefig(savefile, dpi=400)
    plt.close()


if __name__ == '__main__':
    args = get_args()

    scene_dir = os.path.join(args.data_dir, 'scenes', args.scene_name)
    tran_file = os.path.join(scene_dir, 'transforms.json')

    # camera for rendering
    w2cs, K = parse_transforms_json(tran_file, return_w2c=True)
    view_idx = 0
    w2c = w2cs[view_idx]
    w2c[[1, 2]] *= -1

    # color render[color range in open3d is [0, 1]]
    pcd = o3d.geometry.PointCloud()
    gaussian_pcd = PlyData.read(os.path.join(scene_dir, 'saga', 'point_cloud', 'iteration_30000',
                                             'scene_point_cloud.ply'))
    pcd.points = o3d.utility.Vector3dVector(load_saga_xyz(gaussian_pcd))
    pcd.colors = o3d.utility.Vector3dVector(load_gaussian_color(gaussian_pcd))
    render_pcd(pcd, w2c, K, show=True, title='color render')

    # saga feature
    pcd = o3d.geometry.PointCloud()
    feature_pcd = PlyData.read(os.path.join(scene_dir, 'saga', 'point_cloud', 'iteration_10000',
                                            'contrastive_feature_point_cloud.ply'))
    pcd.points = o3d.utility.Vector3dVector(load_saga_xyz(feature_pcd))
    colors_pca = features_to_colors(load_saga_feature(feature_pcd))
    pcd.colors = o3d.utility.Vector3dVector(colors_pca)
    render_pcd(pcd, w2c, K, show=True, title='saga feature')

    # saga feature for source point [down sampled]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.load(os.path.join(scene_dir, 'clip_features', 'source_points.npy')))
    colors_pca = features_to_colors(np.load(os.path.join(scene_dir, 'clip_features', 'saga_features.npy')))
    pcd.colors = o3d.utility.Vector3dVector(colors_pca)
    render_pcd(pcd, w2c, K, show=True, title='down sampled saga feature')

    # clip feature for source point [down sampled]
    patch_features = torch.load(os.path.join(scene_dir, 'clip_features', 'patch_features.pt'), map_location=args.device)
    is_visible = torch.load(os.path.join(scene_dir, 'clip_features', 'is_visible.pt'), map_location=args.device)
    agg_patch_features, is_valid = get_agg_patch_features(patch_features, is_visible)
    source_pts = np.load(os.path.join(scene_dir, 'clip_features', 'source_points.npy'))[is_valid.cpu().numpy()]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(source_pts)
    colors_pca = features_to_colors(agg_patch_features.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors_pca)
    render_pcd(pcd, w2c, K, show=True, title='clip feature')

    # legend for materials
    mat_names = parse_materials_str(os.path.join(scene_dir, f"{args.mats_load_name}.json"))
    cmap_tab10 = mpl.colormaps['tab10']
    make_legend([cmap_tab10(i) for i in range(len(mat_names))], mat_names, show=True)

    # materials of dense point & density of dense point
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.load(os.path.join(scene_dir, f'pred_result_{args.property_name}', 'dense_points.npy')))
    dense_pred_probs = np.load(os.path.join(scene_dir, f'pred_result_{args.property_name}', 'dense_pred_probs.npy'))
    pcd.colors = o3d.utility.Vector3dVector(similarities_to_colors(dense_pred_probs))
    render_pcd(pcd, w2c, K, show=True, title='dense pred probs')

    dense_pred_property = np.load(
        os.path.join(scene_dir, f'pred_result_{args.property_name}', f'dense_pred_{args.property_name}.npy'))
    pcd.colors = o3d.utility.Vector3dVector(
        values_to_colors(np.mean(dense_pred_property, axis=1), low=args.value_low, high=args.value_high))
    render_pcd(pcd, w2c, K, show=True, title='dense pred property')

    plot_color_gradient(low=args.value_low, high=args.value_high)
