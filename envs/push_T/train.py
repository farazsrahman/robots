import env
from dataset import get_data
from unet import get_unet

import numpy as np
import torch
import torch.nn as nn
from diffusers import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import os

#@markdown ### **Training**
#@markdown
#@markdown Takes about 2.5 hours. If you don't want to wait, skip to the next cell
#@markdown to load pre-trained weights

num_epochs = 100

nets, noise_scheduler, device, obs_horizon = get_unet()
noise_pred_net = nets['noise_pred_net']

dataloader, stats = get_data()

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

if __name__ == "__main__":
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = "out/checkpoint/original_code_tests"
    os.makedirs(checkpoint_dir, exist_ok=True)

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch['image'][:,:obs_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    # encoder vision features
                    image_features = nets['vision_encoder'](
                        nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(
                        *nimage.shape[:2],-1)
                    # (B,obs_horizon,D)

                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            
            # Save checkpoint after each epoch
            if epoch_idx % 10 == 0:
                checkpoint = {
                    'epoch': epoch_idx,
                    'model_state_dict': nets.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'loss': np.mean(epoch_loss)
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, f'original_code_checkpoint_epoch_{epoch_idx}.pt'))
                
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    # Weights of the EMA model
    # is used for inference
    ema_nets = nets
    ema.copy_to(ema_nets.parameters())
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': ema_nets.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'final_loss': np.mean(epoch_loss)
    }
    torch.save(final_checkpoint, os.path.join(checkpoint_dir, 'final_model.pt'))

