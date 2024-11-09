import os
import time
import json
import openai
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from utils.gpt_inference_utils import gpt_candidate_materials, parse_material_list, \
    parse_material_hardness, gpt4v_candidate_materials, parse_material_json
from utils.enhance_utils import load_images, get_scenes_list
from settings import get_args
from my_api_key import OPENAI_API_KEY


BASE_SEED = 100


def gpt_wrapper(gpt_fn, parse_fn, max_tries=10, sleep_time=3):
    """Wrap gpt_fn with error handling and retrying."""
    tries = 0
    # sleep to avoid overloading openai api
    time.sleep(sleep_time)
    try:
        gpt_response = gpt_fn(BASE_SEED + tries)
        result = parse_fn(gpt_response)
    except Exception as error:
        print('error:', error)
        result = None
    while result is None and tries < max_tries:
        tries += 1
        time.sleep(sleep_time)
        print('retrying...')
        try:
            gpt_response = gpt_fn(BASE_SEED + tries)
            result = parse_fn(gpt_response)
        except:
            result = None
    return gpt_response


def show_img_to_caption(scene_dir, idx_to_caption):
    img_dir = os.path.join(scene_dir, 'images')
    imgs = load_images(img_dir, bg_change=None, return_masks=False)
    img_to_caption = imgs[idx_to_caption]
    plt.imshow(img_to_caption)
    plt.show()
    plt.close()
    return


def predict_candidate_materials(args, scene_dir, show=False):
    # load caption info
    with open(os.path.join(scene_dir, f'{args.caption_load_name}.json'), 'r') as f:
        info = json.load(f)
    
    if f'candidate_materials_{args.property_name}' in info.keys():
        print(f"{os.path.basename(scene_dir)} have completed info, pass...")
        return info
    
    caption = info['caption']
    if args.additional_material:
        additional_material = info['possible_material']
        caption = f'{caption}({additional_material}).'

    gpt_fn = lambda seed: gpt_candidate_materials(caption, property_name=args.property_name, seed=seed, enhanced=args.additional_material)
    parse_fn = parse_material_hardness if args.property_name == 'hardness' else parse_material_list
    candidate_materials = gpt_wrapper(gpt_fn, parse_fn)

    info[f'candidate_materials_{args.property_name}'] = candidate_materials
    
    print('-' * 50)
    print(f'scene: {os.path.basename(scene_dir)}, info: {info}')
    print(f'candidate materials ({args.property_name}):')
    mat_names, mat_vals = parse_fn(candidate_materials)
    for mat_i, mat_name in enumerate(mat_names):
        print('%16s: %8.1f -%8.1f' % (mat_name, mat_vals[mat_i][0], mat_vals[mat_i][1]))
    if show:
        show_img_to_caption(scene_dir, int(info['idx_to_caption']))
    
    # save info to json
    with open(os.path.join(scene_dir, f'{args.mats_save_name}.json'), 'w') as f:
        json.dump(info, f, indent=4)

    return info


def predict_object_info_gpt4v(args, scene_dir, show=False):
    img_dir = os.path.join(scene_dir, 'images')
    imgs, masks = load_images(img_dir, return_masks=True)
    mask_areas = [np.mean(mask) for mask in masks]

    idx_to_caption = np.random.choice(list(range(len(imgs))))
    img_to_caption = imgs[idx_to_caption]

    # save img_to_caption in img_dir
    img_to_caption = Image.fromarray(img_to_caption)
    img_path = os.path.join(scene_dir, 'img_to_caption.png')
    img_to_caption.save(img_path)

    model_name = None
    if args.proposal_type == 'gpt4v':
        model_name = 'gpt-4-turbo'
    elif args.proposal_type == 'gpt4o':
        model_name = 'gpt-4o'
    else:
        raise ValueError(f"Unknown proposal type: {args.proposal_type}")

    save_json_path = os.path.join(scene_dir, f'{args.mats_save_name}.json')
    if os.path.exists(save_json_path):
        print(f"{os.path.basename(scene_dir)} have completed gpt4 pred, pass...")
        return None

    gpt_fn = lambda seed: gpt4v_candidate_materials(img_path, property_name=args.property_name, seed=seed, model_name=model_name)
    gpt_str_result = gpt_wrapper(gpt_fn, parse_material_json)
    result = parse_material_json(gpt_str_result)
    description, mat_names, mat_vals, pure_volume = result[0], result[1], result[2], result[3]
    candidate_materials = ';'.join([f"({mat_name}: {mat_val[0]}-{mat_val[1]} kg/m^3)" for mat_name, mat_val in zip(mat_names, mat_vals)])

    info = {'idx_to_caption': str(idx_to_caption), 
            'caption': str(description),
            f'candidate_materials_{args.property_name}': candidate_materials,
            'pure_volume': float(pure_volume)
        }
    
    print('-' * 50)
    print(f'scene: {os.path.basename(scene_dir)}, info: {info}')
    print(f'candidate materials ({args.property_name}):')
    mat_names, mat_vals = parse_material_list(candidate_materials)
    for mat_i, mat_name in enumerate(mat_names):
        print('%16s: %8.1f -%8.1f' % (mat_name, mat_vals[mat_i][0], mat_vals[mat_i][1]))
    if show:
        show_img_to_caption(scene_dir, int(info['idx_to_caption']))
    
    # save info to json
    with open(save_json_path, 'w') as f:
        json.dump(info, f, indent=4)

    return info


if __name__ == '__main__':

    args = get_args()

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)

    openai.api_key = OPENAI_API_KEY
    
    for j, scene in enumerate(scenes): 
        print(f"=====no.{j+1}/{len(scenes)}:{scene}=====")
        if args.proposal_type == 'text-reasoning':
            mats_info = predict_candidate_materials(args, os.path.join(scenes_dir, scene))
        elif args.proposal_type in ['gpt4v', 'gpt4o']:
            mats_info = predict_object_info_gpt4v(args, os.path.join(scenes_dir, scene))
        else:
            raise ValueError(f"Unknown proposal type: {args.proposal_type}")
