import os
import json
import torch
import open_clip
import numpy as np
import shutil

from clip_feature_fusion import CLIP_BACKBONE, CLIP_CHECKPOINT
from utils.gpt_inference_utils import parse_material_list, parse_material_hardness
from utils.enhance_utils import load_saga_point_cloud, get_scenes_list, bounding_box_size, load_gaussian_point_cloud
from settings import get_args


omega_acc_rate = [0.6826**3, 0.9545**3, 0.9973**3]  # +-1omega, +-2omega, +-3omega


@torch.no_grad()
def get_text_features(texts, clip_model, clip_tokenizer, prefix='', suffix='', device='cuda'):
    """Get CLIP text features, optionally with a fixed prefix and suffix."""
    extended_texts = [prefix + text + suffix for text in texts]
    tokenized = clip_tokenizer(extended_texts).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = clip_model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return text_features


@torch.no_grad()
def get_agg_patch_features(patch_features, is_visible):
    """Get aggregated patch features by averaging over visible patches."""
    n_visible = is_visible.sum(0)
    is_valid = n_visible > 0

    visible_patch_features = patch_features * is_visible.unsqueeze(-1)
    avg_visible_patch_features = visible_patch_features.sum(0) / n_visible.unsqueeze(-1)
    avg_visible_patch_features = avg_visible_patch_features / avg_visible_patch_features.norm(dim=1, keepdim=True)
    return avg_visible_patch_features[is_valid], is_valid


@torch.no_grad()
def get_interpolated_values(source_pts, source_vals, inner_pts, batch_size=2048, k=1):
    """Interpolate values by k nearest neighbor."""
    n_inner = len(inner_pts)
    inner_vals = torch.zeros(n_inner, source_vals.shape[1], device=inner_pts.device)
    for batch_start in range(0, n_inner, batch_size):
        curr_batch_size = min(batch_size, n_inner - batch_start)
        curr_inner_pts = inner_pts[batch_start:batch_start + curr_batch_size]

        dists = torch.cdist(curr_inner_pts, source_pts)
        _, idxs = torch.topk(dists, k=k, dim=1, largest=False)
        curr_inner_vals = source_vals[idxs].mean(1)

        inner_vals[batch_start:batch_start + curr_batch_size] = curr_inner_vals
    return inner_vals


@torch.no_grad()
def get_gaussian_volume_with_opacity(scales, opacitys):
    return (4.0/3.0) * torch.pi * scales.prod(dim=1) * opacitys * omega_acc_rate[0]


@torch.no_grad()
def get_interpolated_values_by_saga_features(source_feature, source_vals, dense_feature, args):
    """Interpolate value based on feature similarities"""
    similarities = dense_feature @ source_feature.T
    similarities = torch.softmax(similarities / args.temperature, dim=1)
    pred_vals = similarities @ source_vals
    return pred_vals


@torch.no_grad()
def predict_physical_property_integral(args, scene_dir, clip_model, clip_tokenizer):
    """Predict the volume integral of a physical property (e.g. for mass). Returns a [low, high] range."""

    scene_name = os.path.basename(scene_dir)
    pcd_file = os.path.join(scene_dir, 'saga', 'point_cloud','iteration_10000','contrastive_feature_point_cloud.ply')
    color_pcd_file = os.path.join(scene_dir, 'saga', 'point_cloud','iteration_30000','scene_point_cloud.ply')
    info_file = os.path.join(scene_dir, '%s.json' % args.mats_load_name)

    with open(info_file, 'r') as f:
        info = json.load(f)

    # loading source point info
    source_point_file = os.path.join(scene_dir, 'clip_features', 'source_points.npy')
    source_pts = torch.Tensor(np.load(source_point_file)).to(args.device)
    source_saga_feature_file = os.path.join(scene_dir, 'clip_features', 'saga_features.npy')
    source_pts_saga_features = torch.Tensor(np.load(source_saga_feature_file)).to(args.device)
    patch_features = torch.load(os.path.join(scene_dir, 'clip_features', 'patch_features.pt'))
    is_visible = torch.load(os.path.join(scene_dir, 'clip_features', 'is_visible.pt'))

    # preparing material info
    mat_val_list = info[f'candidate_materials_{args.property_name}']
    mat_names, mat_vals = parse_material_list(mat_val_list)
    mat_vals = torch.Tensor(mat_vals).to(args.device)

    # predictions on source points
    text_features = get_text_features(mat_names, clip_model, clip_tokenizer, device=args.device)
    agg_patch_features, is_valid = get_agg_patch_features(patch_features, is_visible)
    source_pts = source_pts[is_valid]
    source_pts_saga_features = source_pts_saga_features[is_valid]

    similarities = agg_patch_features @ text_features.T
    source_pred_probs = torch.softmax(similarities / args.temperature, dim=1)
    sample_voxel_size = args.sample_voxel_size * bounding_box_size(source_pts.cpu().numpy())

    if args.volume_method == 'thickness':
        mat_tn_list = info['thickness']
        mat_names, mat_tns = parse_material_list(mat_tn_list)
        mat_tns = torch.Tensor(mat_tns).to(args.device) / 100  # cm to m

        dense_pts, dense_saga_features = load_saga_point_cloud(pcd_file, ds_size=sample_voxel_size)
        _, dense_colors, _, _ = load_gaussian_point_cloud(color_pcd_file, ds_size=sample_voxel_size)
        dense_pts = torch.Tensor(np.asarray(dense_pts.points)).to(args.device)
        dense_saga_features = torch.Tensor(dense_saga_features).to(args.device)

        if not np.isnan(source_pts_saga_features.cpu().numpy()).any():
            dense_pred_probs = get_interpolated_values_by_saga_features(
                source_pts_saga_features, source_pred_probs, dense_saga_features, args=args
            )
        else:
            dense_pred_probs = get_interpolated_values(source_pts, source_pred_probs, dense_pts, batch_size=2048, k=1)
        
        # volume integral
        surface_cell_size = sample_voxel_size  # m
        mat_cell_volumes = surface_cell_size**2 * mat_tns  # m^3
        mat_cell_products = mat_vals * mat_cell_volumes  # the weight of each material cell(kg)

        dense_pred_products = dense_pred_probs @ mat_cell_products
        total_pred_val = (dense_pred_products).sum(0)
        total_pred_val *= args.correction_factor
    elif args.volume_method == 'gaussian':
        dense_pts, dense_saga_features, dense_scales, dense_opacitys = load_saga_point_cloud(pcd_file, ds_size=None, return_gaussian_property=True)
        _, dense_colors, _, _ = load_gaussian_point_cloud(color_pcd_file, ds_size=None)
        dense_pts = torch.Tensor(dense_pts).to(args.device)
        dense_saga_features = torch.Tensor(dense_saga_features).to(args.device)
        dense_scales = torch.exp(torch.Tensor(dense_scales).to(args.device))
        dense_opacitys = torch.sigmoid(torch.Tensor(dense_opacitys).to(args.device))

        surface_indices = dense_opacitys>0.95
        dense_pts = dense_pts[surface_indices]
        dense_saga_features = dense_saga_features[surface_indices]
        dense_scales = dense_scales[surface_indices]
        dense_opacitys = dense_opacitys[surface_indices]

        if not np.isnan(source_pts_saga_features.cpu().numpy()).any():
            dense_pred_probs = get_interpolated_values_by_saga_features(
                source_pts_saga_features, source_pred_probs, dense_saga_features, args=args
            )
        else:
            dense_pred_probs = get_interpolated_values(source_pts, source_pred_probs, dense_pts, batch_size=2048, k=1)
        dense_gs_volumes = get_gaussian_volume_with_opacity(dense_scales, dense_opacitys)
        total_pred_val = dense_gs_volumes @ dense_pred_probs @ mat_vals
        
        gs_volume = dense_gs_volumes.sum().item()
        pure_volume = float(info['pure_volume'])
        gs_scale_factor = pure_volume / gs_volume
        
        total_pred_val *= gs_scale_factor

    else:
        raise NotImplementedError
    
    result_dir = os.path.join(scene_dir, f'pred_result_{args.property_name}')
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    dense_pred_property = dense_pred_probs @ mat_vals
    np.save(os.path.join(result_dir, f'dense_pred_{args.property_name}.npy'), dense_pred_property.cpu().numpy())
    np.save(os.path.join(result_dir, 'dense_pred_probs.npy'), dense_pred_probs.cpu().numpy())
    np.save(os.path.join(result_dir, 'dense_points.npy'), dense_pts.cpu().numpy())
    np.save(os.path.join(result_dir, 'dense_colors.npy'), dense_colors)

    print('-' * 50)
    print('scene:', scene_name)
    print('-' * 50)
    print('num. dense points:', len(dense_pts))
    print('caption:', info['caption'])
    print('candidate materials:')
    for mat_i, mat_name in enumerate(mat_names):
        print('%16s: %8.1f -%8.1f kg/m^3' % (mat_name, mat_vals[mat_i][0], mat_vals[mat_i][1]))
    if args.volume_method == 'thickness':
        print('surface cell size: %.4f cm' % (surface_cell_size * 100))
    print('predicted total mass: [%.4f - %.4f kg]' % (total_pred_val[0], total_pred_val[1]))

    return total_pred_val.tolist()


@torch.no_grad()
def predict_physical_property_query(args, query_mode, scene_dir, clip_model, clip_tokenizer, return_all=False):

    scene_name = os.path.basename(scene_dir)
    pcd_file = os.path.join(scene_dir, 'saga', 'point_cloud', 'iteration_10000', 'contrastive_feature_point_cloud.ply')
    color_pcd_file = os.path.join(scene_dir, 'saga', 'point_cloud','iteration_30000','scene_point_cloud.ply')
    info_file = os.path.join(scene_dir, '%s.json' % args.mats_load_name)

    with open(info_file, 'r') as f:
        info = json.load(f)

    # loading source point info
    source_point_file = os.path.join(scene_dir, 'clip_features', 'source_points.npy')
    source_pts = torch.Tensor(np.load(source_point_file)).to(args.device)
    source_saga_feature_file = os.path.join(scene_dir, 'clip_features', 'saga_features.npy')
    source_pts_saga_features = torch.Tensor(np.load(source_saga_feature_file)).to(args.device)
    patch_features = torch.load(os.path.join(scene_dir, 'clip_features', 'patch_features.pt'))
    is_visible = torch.load(os.path.join(scene_dir, 'clip_features', 'is_visible.pt'))

    # preparing material info
    mat_val_list = info[f'candidate_materials_{args.property_name}']
    if args.property_name == 'hardness':
        mat_names, mat_vals = parse_material_hardness(mat_val_list)
    else:
        mat_names, mat_vals = parse_material_list(mat_val_list)
    mat_vals = torch.Tensor(mat_vals).to(args.device)

    # predictions on source points
    text_features = get_text_features(mat_names, clip_model, clip_tokenizer, device=args.device)
    agg_patch_features, is_valid = get_agg_patch_features(patch_features, is_visible)
    source_pts = source_pts[is_valid]
    source_pts_saga_features = source_pts_saga_features[is_valid]

    similarities = agg_patch_features @ text_features.T

    source_pred_probs = torch.softmax(similarities / args.temperature, dim=1)
    
    if query_mode == 'grid':
        sample_voxel_size = args.sample_voxel_size * bounding_box_size(source_pts.cpu().numpy())
        query_pts, query_pts_saga_feature = load_saga_point_cloud(pcd_file, ds_size=sample_voxel_size)
        _, dense_colors, _, _ = load_gaussian_point_cloud(color_pcd_file, ds_size=sample_voxel_size)
        query_pts = torch.Tensor(np.asarray(query_pts.points)).to(args.device)
        query_pts_saga_feature = torch.Tensor(query_pts_saga_feature).to(args.device)

        if not np.isnan(source_pts_saga_features.cpu().numpy()).any():
            query_pred_probs = get_interpolated_values_by_saga_features(
                source_pts_saga_features, source_pred_probs, query_pts_saga_feature, args=args
            )
        else:
            query_pred_probs = get_interpolated_values(source_pts, source_pred_probs, query_pts, batch_size=2048, k=1)
    elif query_mode == 'gaussian':
        raise NotImplementedError
    else:
        raise NotImplementedError     

    query_pred_vals = query_pred_probs @ mat_vals

    result_dir = os.path.join(scene_dir, f'pred_result_{args.property_name}')
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    np.save(os.path.join(result_dir, f'dense_pred_{args.property_name}.npy'), query_pred_vals.cpu().numpy())
    np.save(os.path.join(result_dir, 'dense_pred_probs.npy'), query_pred_probs.cpu().numpy())
    np.save(os.path.join(result_dir, 'dense_points.npy'), query_pts.cpu().numpy())
    np.save(os.path.join(result_dir, 'dense_colors.npy'), dense_colors)

    print('-' * 50)
    print('scene:', scene_name)
    print('-' * 50)
    print('num. query points:', len(query_pts))
    print('caption:', info['caption'])
    print('candidate materials (%s):' % args.property_name)
    for mat_i, mat_name in enumerate(mat_names):
        print('%16s: %8.1f -%8.1f' % (mat_name, mat_vals[mat_i][0], mat_vals[mat_i][1]))
    return query_pred_vals


if __name__ == '__main__':

    args = get_args()

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_CHECKPOINT)
    clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer(CLIP_BACKBONE)

    preds = {}
    for i, scene in enumerate(scenes): 
        print(f'===process no.{i+1}/{len(scenes)} scene:{scene}===')
        scene_dir = os.path.join(scenes_dir, scene)
        if args.prediction_mode == 'integral':
            pred = predict_physical_property_integral(args, scene_dir, clip_model, clip_tokenizer)
        elif args.prediction_mode == 'grid':
            pred = predict_physical_property_query(args, 'grid', scene_dir, clip_model, clip_tokenizer)
        else:
            raise NotImplementedError
        preds[scene] = pred
    
    if args.prediction_mode == 'integral' and args.save_preds:
        os.makedirs('preds', exist_ok=True)
        with open(os.path.join('preds', f'preds_{args.property_name}.json'), 'w') as f:
            json.dump(preds, f, indent=4)
