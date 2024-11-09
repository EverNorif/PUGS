import os
import sys
import subprocess

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from settings import get_args
from utils.enhance_utils import get_scenes_list


def run_python_script(script_name, args=None, run_as_module=True):
    if args is None:
        args = []
    run_cmd = ["python", "-m"] if run_as_module else ["python"]
    run_cmd.extend([script_name] + args)
    
    print(f"exec: {' '.join(run_cmd)}")
    try:
        result = subprocess.run(run_cmd)
        if result.stderr: 
            print("stderr:", result.stderr.strip())
        return result.returncode

    except subprocess.CalledProcessError as e:
        print("Error occurred while running script:", script_name)
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        print("returncode:", e.returncode)
        return e.returncode

    except FileNotFoundError as e:
        print("Error: script not found:", script_name)
        print(e)
        return -1

    except Exception as e:
        print("An unexpected error occurred")
        print(e)
        return -1


if __name__ == "__main__":
    args = get_args()

    scenes_dir = os.path.join(args.data_dir, "scenes")
    scenes = get_scenes_list(args)

    for i, scene in enumerate(scenes):
        print(f'===process no.{i+1}/{len(scenes)} scene:{scene}===')
        scene_dir = os.path.join(scenes_dir, scene)
        output_dir = os.path.join(scene_dir, "saga")

        # train scene to get raw 3dgs
        run_python_script('reconstruction.train_scene', [
            '-s', scene_dir,
            '--model_path', output_dir,
        ])

        # sam mask extract
        run_python_script('reconstruction.extract_segment_everything_masks', [
            '--image_root', scene_dir,
        ])

        # get scale
        run_python_script('reconstruction.get_scale', [
            '--image_root', scene_dir,
            '--model_path', output_dir,
        ])

        # contrastive feature training
        run_python_script("reconstruction.train_contrastive_feature", [
            "-m", output_dir,
            "--iterations", str(args.training_iters),
            "--num_sampled_rays", str(args.num_sampled_rays),
        ])
