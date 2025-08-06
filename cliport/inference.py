"""Ravens main training script."""

import os
import pickle
import json
import pdb

import numpy as np
import hydra
from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.environments.environment import Environment


@hydra.main(config_path='./cfg', config_name='eval')
def main(vcfg):
    # Load train cfg
    # pdb.set_trace()
    tcfg = utils.load_hydra_config(vcfg['inference_config'])

    # Initialize environment and task.
    env = Environment(
        vcfg['assets_root'],
        disp=vcfg['disp'],
        shared_memory=vcfg['shared_memory'],
        hz=480,
        record_cfg=vcfg['record']
    )

    # Choose eval mode and task.
    mode = vcfg['mode']
    eval_task = vcfg['eval_task']
    if mode not in {'train', 'val', 'test'}:
        raise Exception("Invalid mode. Valid options: train, val, test")

    # Load eval dataset.
    dataset_type = vcfg['type']

    if 'multi' in dataset_type:
        ds = dataset.RavensMultiTaskDataset(vcfg['data_dir'],
                                            tcfg,
                                            group=eval_task,
                                            mode=mode,
                                            n_demos=vcfg['n_demos'],
                                            augment=False)
    else:
        ds = dataset.RavensDataset(os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"),
                                   tcfg,
                                   n_demos=vcfg['n_demos'],
                                   augment=False)

    all_results = {}
    name = '{}-{}-n{}'.format(eval_task, vcfg['agent'], vcfg['n_demos'])

    # Save path for results.
    json_name = f"multi-results-{mode}.json" if 'multi' in vcfg['model_path'] else f"results-{mode}.json"
    save_path = vcfg['save_path']
    print(f"Save path for results: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_json = os.path.join(save_path, f'{name}-{json_name}')

    # Load existing results.
    existing_results = {}
    if os.path.exists(save_json):
        with open(save_json, 'r') as f:
            existing_results = json.load(f)

    print("Skipping checkpoint loading: No pre-trained weights required.")
    results = []
    mean_reward = 0.0

    # Run testing for each training run.
    for train_run in range(vcfg['n_repeats']):

        # Initialize agent.
        utils.set_seed(train_run, torch=True)
        # pdb.set_trace()
        agent = agents.names[vcfg['agent']](name, tcfg, None, ds)
        print("Agent initialized with random weights.")

        record = vcfg['record']['save_video']
        n_demos = vcfg['n_demos']

        # Run testing and save total rewards with last transition info.
        for i in range(0, n_demos):
            print(f'Test: {i + 1}/{n_demos}')
            episode, seed = ds.load(i)
            goal = episode[-1]
            total_reward = 0
            np.random.seed(seed)

            # set task
            if 'multi' in dataset_type:
                task_name = ds.get_curr_task()
                task = tasks.names[task_name]()
                print(f'Evaluating on {task_name}')
            else:
                task_name = vcfg['eval_task']
                task = tasks.names[task_name]()

            task.mode = mode
            env.seed(seed)
            env.set_task(task)
            obs = env.reset()
            info = env.info
            reward = 0

            # Start recording video (NOTE: super slow)
            if record:
                video_name = f'{task_name}-{i+1:06d}'
                if 'multi' in vcfg['model_task']:
                    video_name = f"{vcfg['model_task']}-{video_name}"
                env.start_rec(video_name)

            for _ in range(task.max_steps):
                # pdb.set_trace()
                act = agent.act(obs, info, goal)
                lang_goal = info['lang_goal']
                print(f'Lang Goal: {lang_goal}')
                obs, reward, done, info = env.step(act)
                total_reward += reward
                print(f'Total Reward: {total_reward:.3f} | Done: {done}\n')
                if done:
                    break

            results.append((total_reward, info))
            mean_reward = np.mean([r for r, i in results])
            print(f'Mean: {mean_reward} | Task: {task_name}')

            # End recording video
            if record:
                env.end_rec()

    # Save results in a json file.
    if vcfg['save_results']:

        # Load existing results
        if os.path.exists(save_json):
            with open(save_json, 'r') as f:
                existing_results = json.load(f)
            existing_results.update(all_results)
            all_results = existing_results

        with open(save_json, 'w') as f:
            json.dump(all_results, f, indent=4)


if __name__ == '__main__':
    main()
