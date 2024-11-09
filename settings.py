import argparse


def get_args():
    parser = argparse.ArgumentParser(description='SAGA+Physics')

    # General arguments
    parser.add_argument('--data_dir', type=str, default="./data/abo_500/",
                        help='path to data (default: ./data/abo_500/)')
    parser.add_argument('--split', type=str, default="all",
                        help='dataset split, either train, val, train+val, test, or all (default: all)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='starting scene index, useful for evaluating only a few scenes (default: 0)')
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='ending scene index, useful for evaluating only a few scenes (default: -1)')
    parser.add_argument('--different_Ks', action='store_true',
                        help='whether data has cameras with different intrinsic matrices (default: 0)')
    parser.add_argument('--device', type=str, default="cuda",
                        help='device for torch (default: cuda)')

    # Gaussian training
    parser.add_argument('--training_iters', type=int, default=10000,
                        help='number of iterations for training contrastive feature (default: 10000)')
    parser.add_argument('--num_sampled_rays', type=int, default=1000,
                        help='number of sampled rays for training contrastive feature (default: 1000)')

    # CLIP feature fusion
    parser.add_argument('--patch_size', type=int, default=56,
                        help='patch size (default: 56)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument('--feature_voxel_size', type=int, default=0.02,
                        help='voxel downsampling size for features, relative to scaled scene (default: 0.02)')
    parser.add_argument('--occ_thr', type=float, default=0.02,
                        help='occlusion threshold, relative to scaled scene (default: 0.01)')

    # Material proposal
    parser.add_argument('--caption_load_name', type=str, default="info",
                        help='name of saved caption to load (default: info)')
    parser.add_argument('--additional_material', action='store_true',
                        help='give LLM additional material information.')
    parser.add_argument('--proposal_type', type=str, default="gpt4o",
                        help='material proposal type: [text-reasoning, gpt4v, gpt4o] (default: gpt4o)')
    parser.add_argument('--property_name', type=str, default="density",
                        help='property to predict (default: density)')
    parser.add_argument('--mats_save_name', type=str, default="info",
                        help='candidate materials save name (default: info)')

    # Physical property prediction (uses property_name argument from above)
    parser.add_argument('--mats_load_name', type=str, default="info",
                        help='candidate materials load name (default: info)')
    parser.add_argument('--prediction_mode', type=str, default="integral",
                        help="can be either 'integral' or 'grid' (default: integral)")
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='softmax s for kernel regression (default: 0.1)')
    parser.add_argument('--sample_voxel_size', type=float, default=0.005,
                        help='voxel downsampling size for sampled points, relative to scaled scene (default: 0.005)')
    parser.add_argument('--volume_method', type=str, default="gaussian",
                        help="method for volume estimation, either 'thickness' or 'gaussian' (default: gaussian)")
    parser.add_argument('--save_preds', type=int, default=1,
                        help='whether to save predictions (default: 1)')
    parser.add_argument('--preds_save_name', type=str, default="mass",
                        help='predictions save name (default: mass)')

    # Evaluation
    parser.add_argument('--preds_json_path', type=str, default="./preds/preds_mass.json",
                        help='path to predictions JSON file (default: ./preds/preds_mass.json)')
    parser.add_argument('--gts_json_path', type=str, default="./data/abo_500/filtered_product_weights.json",
                        help='path to ground truth JSON file (default: ./data/abo_500/filtered_product_weights.json)')
    parser.add_argument('--clamp_min', type=float, default=0.01,
                        help='minimum value to clamp predictions (default: 0.01)')
    parser.add_argument('--clamp_max', type=float, default=100.,
                        help='maximum value to clamp predictions (default: 100.)')

    # Visualization
    parser.add_argument('--scene_name', type=str,
                        help='scene name for visualization (must be provided)')
    parser.add_argument('--value_low', type=float, default=500,
                        help='minimum physical property value for colormap (default: 500)')
    parser.add_argument('--value_high', type=float, default=3500,
                        help='maximum physical property value for colormap (default: 3500)')

    args = parser.parse_args()

    return args