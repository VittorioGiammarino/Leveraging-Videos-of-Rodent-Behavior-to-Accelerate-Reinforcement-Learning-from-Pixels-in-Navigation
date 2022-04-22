#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:56:41 2022

@author: vittoriogiammarino
"""

''
import torch
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image

import numpy as np
import argparse
import os
import pickle

from models.LSUR_model import reparameterize
from models.LSUR_model import minigridDisentangledVAE, CarlaDisentangledVAE, DisentangledVAE, SawyerDisentangledVAE

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=10, type=int)  
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--learning-rate', default=3e-4, type=float)
parser.add_argument('--beta', default=0.1, type=int)
parser.add_argument('--save-freq', default=1000, type=int)
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--class-latent-size', default=8, type=int)
parser.add_argument('--content-latent-size', default=64, type=int)
args = parser.parse_args()

Model = SawyerDisentangledVAE

def vae_loss(x, mu, logsigma, recon_x, beta=1):
    recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    kl_loss = kl_loss / torch.numel(x)
    return recon_loss + kl_loss * beta

def forward_loss(x, model, beta):
    mu, logsigma, classcode = model.encoder(x)
    contentcode = reparameterize(mu, logsigma)
    shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]

    latentcode1 = torch.cat([contentcode, shuffled_classcode], dim=1)
    latentcode2 = torch.cat([contentcode, classcode], dim=1)

    recon_x1 = model.decoder(latentcode1)
    recon_x2 = model.decoder(latentcode2)

    return vae_loss(x, mu, logsigma, recon_x1, beta) + vae_loss(x, mu, logsigma, recon_x2, beta)

def backward_loss(x, model, device):
    mu, logsigma, classcode = model.encoder(x)
    shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]
    randcontent = torch.randn_like(mu).to(device)

    latentcode1 = torch.cat([randcontent, classcode], dim=1)
    latentcode2 = torch.cat([randcontent, shuffled_classcode], dim=1)

    recon_imgs1 = model.decoder(latentcode1).detach()
    recon_imgs2 = model.decoder(latentcode2).detach()

    cycle_mu1, cycle_logsigma1, cycle_classcode1 = model.encoder(recon_imgs1)
    cycle_mu2, cycle_logsigma2, cycle_classcode2 = model.encoder(recon_imgs2)

    cycle_contentcode1 = reparameterize(cycle_mu1, cycle_logsigma1)
    cycle_contentcode2 = reparameterize(cycle_mu2, cycle_logsigma2)

    bloss = F.mse_loss(cycle_contentcode1, cycle_contentcode2)
    return bloss    

def main():
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    
    if not os.path.exists('checkimages'):
        os.makedirs("checkimages")
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    with open('data_set/human_data_set/obs_humans_processed.npy', 'rb') as f:
        off_policy_observations = np.load(f, allow_pickle=True)
        
    modified_observations = off_policy_observations
    
    with open('data_set/human_data_set/Sawyer_random_obs.npy', 'rb') as f:
        data_sawyer = np.load(f, allow_pickle=True)
        
    real_observations = data_sawyer.transpose(0,3,1,2)
        
    number_of_classes = 2

    # state_dim = data_set_real['observations'][0].shape
    
    # create model
    model = Model(class_latent_size = args.class_latent_size, content_latent_size = args.content_latent_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # do the training
    batch_count = 0
    num_observations_mod = len(modified_observations)
    num_observations_real = len(real_observations)
    max_steps = 50000
    
    for _ in range(int(max_steps)):
        batch_count += 1
        minibatch_indices_mod = np.random.choice(range(num_observations_mod), args.batch_size, False)
        minibatch_indices_real = np.random.choice(range(num_observations_real), args.batch_size, False)
        
        optimizer.zero_grad()

        floss = 0
        
        images_real = torch.FloatTensor((real_observations[minibatch_indices_real])/(255)).to(device)
        floss += forward_loss(images_real, model, args.beta)
        
        images_mod = torch.FloatTensor((modified_observations[minibatch_indices_mod])/(255)).to(device)
        floss += forward_loss(images_mod, model, args.beta)
        
        floss = floss / number_of_classes

        # backward circle
        images_tot = torch.cat((images_real, images_mod))
        bloss = backward_loss(images_tot, model, device)

        (floss + bloss * args.bloss_coef).backward()
        optimizer.step()

        # save image to check and save model 
        if batch_count % args.save_freq == 0:
            print(f'floss: {floss.item()}')
            print(f'bloss: {bloss.item()}')
            print("%d Epochs." % (batch_count))
            rand_idx = torch.randperm(images_tot.shape[0])
            imgs1 = images_tot[rand_idx[:9]]
            imgs2 = images_tot[rand_idx[-9:]]
            with torch.no_grad():
                mu, _, classcode1 = model.encoder(imgs1)
                _, _, classcode2 = model.encoder(imgs2)
                recon_imgs1 = model.decoder(torch.cat([mu, classcode1], dim=1))
                recon_combined = model.decoder(torch.cat([mu, classcode2], dim=1))
                
            saved_imgs = torch.cat([imgs1, imgs2, recon_imgs1, recon_combined], dim=0)
            save_image(saved_imgs, "./checkimages/%d.png" % (batch_count), nrow=9)

            torch.save(model.state_dict(), "./checkpoints/model.pt")
            torch.save(model.encoder.state_dict(), "./checkpoints/encoder.pt")

if __name__ == '__main__':
    main()
    



