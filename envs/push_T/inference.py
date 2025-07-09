from envs.push_T.env import PushTImageEnv
from models.unet import get_unet
from envs.push_T.dataset import normalize_data, unnormalize_data, get_data

from diffusers import EMAModel
import torch
import argparse
import os
import collections
from tqdm import tqdm
import numpy as np

obs_horizon = 2
pred_horizon = 16
action_horizon = 8
action_dim = 2
num_diffusion_iters = 10

def load_checkpoint(checkpoint_path, nets, ema):
    """Load model checkpoint and return the model state"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if 'model_state_dict' in checkpoint:
        nets.load_state_dict(checkpoint['model_state_dict'])
        if 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
            ema.copy_to(nets.parameters())
    else:
        raise ValueError("Invalid checkpoint format: missing model_state_dict")
    
    return checkpoint

def main(checkpoint_path, start_env_seed=100000, num_iterations=10):
    successes = 0
    total_reward = 0

    for iteration in range(num_iterations):
        env_seed = start_env_seed + iteration
        # limit enviornment interaction to 200 steps before termination
        max_steps = 500
        env = PushTImageEnv()
        env.seed(env_seed)

        nets, noise_scheduler, device, obs_horizon = get_unet()

        ema = EMAModel(
                parameters=nets.parameters(),
                power=0.75
            )

        _, stats = get_data()

        # Load checkpoint if provided
        if checkpoint_path:
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path, nets, ema)
            if 'loss' in checkpoint:
                print(f"Checkpoint loss: {checkpoint['loss']}")

        # get first observation
        obs, info = env.reset()

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        # save visualization and rewards
        imgs = [env.render(mode='rgb_array')]
        rewards = list()
        done = False
        step_idx = 0

        with tqdm(total=max_steps, desc=f"Eval PushTImageEnv (Seed: {env_seed})") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon number of observations
                images = np.stack([x['image'] for x in obs_deque])
                agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

                # normalize observation
                nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
                # images are already normalized to [0,1]
                nimages = images

                # device transfer
                nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
                # (2,3,96,96)
                nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
                # (2,2)

                # infer action
                with torch.no_grad():
                    # get image features
                    image_features = nets['vision_encoder'](nimages)
                    # (2,512)

                    # concat with low-dim observations
                    obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, pred_horizon, action_dim), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    for k in noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = nets['noise_pred_net'](
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )

                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=stats['action'])

                # only take action_horizon number of actions
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end,:]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, _, info = env.step(action[i])
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    imgs.append(env.render(mode='rgb_array'))

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break

        # print out the maximum target coverage
        max_reward = max(rewards)
        print(f'Seed {env_seed} - Score: {max_reward}')

        # Update statistics
        if max_reward > 0.95:
            successes += 1
        total_reward += max_reward

        # visualize
        from IPython.display import Image
        import imageio
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs('out', exist_ok=True)
        
        # Save video as gif
        base_name = args.checkpoint.split('/')[-1].split('.')[0]
        out_path = f'out/{base_name}_seed_{env_seed}.gif'
        imageio.mimsave(out_path, imgs, fps=30)
        
        Image(out_path, width=256, height=256)

    # Calculate and print overall statistics
    success_rate = (successes / num_iterations) * 100
    average_reward = total_reward / num_iterations
    print(f"\nOverall Statistics:")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {average_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with a specific checkpoint')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file to load')
    parser.add_argument('--start_env_seed', type=int, default=100000, help='Starting environment seed')
    parser.add_argument('--num_iterations', type=int, default=10, help='Number of iterations to run')
    args = parser.parse_args()
    
    main(args.checkpoint, args.start_env_seed, args.num_iterations)