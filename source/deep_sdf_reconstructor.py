# !/usr/bin/env python3

import os
import torch

from .learning_rate import LearningRateSchedule
from .sdf_samples import SDFSamples
from .deep_sdf import Decoder

class DeepSDFReconstructor:

    def __init__(
        self,
        decoder,
        num_iterations,
        latent_size,
        std,
        clamp_dist,
        num_samples=8000,
        lr=5e-4,
        l2reg=True,
        *,
        device="cpu"
    ):
        self.decoder = decoder
        self.std = std
        self.latent_size = latent_size
        self.num_iterations = num_iterations
        self.num_samples = num_samples
        self.clamp_dist = clamp_dist
        self.lr = lr
        self.l2reg = l2reg

    def _init_latent(self):
        self.latent = torch.ones(1, self.latent_size).normal_(mean=0, std=self.std)
        self.latent.requires_grad = True

    def _init_learning_scedules(self):
        self.lr_schedules = []
        initial = self.lr
        interval = int(self.num_iterations / 2)
        factor = 0.1
        self.lr_schedules.append(
            LearningRateSchedule(initial, interval, factor)
            )

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            [self.latent], 
            lr=self.lr_schedules[0].get_learning_rate(0)
        )

    def _adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.lr_schedules[i].get_learning_rate(epoch)

    def reconstruct(self, test_paths):
        ''' Reconstruct latent vector for object SDF samples from test_path.
        '''
        self._init_learning_scedules()
        self._init_latent()
        self._init_optimizer()

        loss_num = 0
        loss_l1 = torch.nn.L1Loss()

        for epoch in range(self.num_iterations):

            self.decoder.eval()
            # May be slow due to the loading from the disc.
            sdf_data = SDFSamples(test_paths, self.num_samples)
            sdf_data, _ = sdf_data[0]

            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3].unsqueeze(1)

            sdf_gt = torch.clamp(sdf_gt, -self.clamp_dist, self.clamp_dist)

            self._adjust_learning_rate(epoch)

            self.optimizer.zero_grad()

            latent_inputs = self.latent.expand(self.num_samples, -1)

            inputs = torch.cat([latent_inputs, xyz], 1)

            pred_sdf = self.decoder(inputs.float())
            pred_sdf = torch.clamp(pred_sdf, -self.clamp_dist, self.clamp_dist)

            loss = loss_l1(pred_sdf, sdf_gt)
            
            if self.l2reg:
                loss += 1e-4 * torch.mean(self.latent.pow(2))

            loss.backward()
            self.optimizer.step()

            loss_num = loss.cpu().data.numpy()
            print('Epoch {}: loss = {}'.format(epoch, loss_num))
            
        return loss_num, self.latent
