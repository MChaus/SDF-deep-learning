# !/usr/bin/env python3

import os
import json
import torch

from deep_sdf import Decoder
from sdf_samples import SDFSamples
from learning_rate import LearningRateSchedule

class DeepSDFTrainer:
    '''
    Class for loading specs, training and saving DeepSDF NN.

    Loads specs from spec_path file.
    Trains model accroding to specs.
    Saves model, optimizer, latent vectors periodicly to detect overfitting in
    the future.
    '''
    def __init__(
        self, 
        spec_path: str, 
        train_paths: list, 
        results_path: str, 
        *, 
        device='cpu'
    ):
        self.spec_path = spec_path
        self.train_paths = train_paths
        self.results_path = results_path
        self.specs = self._load_specs(spec_path)
        self._extract_params()
        self._init_learning_scedules()
        self._init_data_loader()
        self._init_latent_vecs()
        self._init_optimizer()

    def _load_specs(self, filename: str):
        with open(filename) as f:
            specs = json.load(f)
        return specs

    def _extract_params(self):
        self.latent_size = self.specs['CodeLength']
        self.decoder = Decoder(self.latent_size, **self.specs['NetworkSpecs'])

        self.subsample_num = self.specs['SamplesPerScene']
        self.batch_size = self.specs['ScenesPerBatch']
        self.network_specs = self.specs['NetworkSpecs']
        self.batch_split = self.specs['BatchSplit']
    
        self.clamp_dist = self.specs['ClampingDistance']
        self.minT = -self.clamp_dist
        self.maxT = self.clamp_dist

        self.do_code_regularization = self.specs['CodeRegularization']
        self.code_reg_lambda = self.specs['CodeRegularizationLambda']
        self.code_bound = self.specs['CodeBound']
        self.num_epochs = self.specs['NumEpochs']

        self.specs_schedule = self.specs['LearningRateSchedule']
        self.snapshot_frequency = self.specs['SnapshotFrequency']

    def _init_learning_scedules(self):
        self.lr_schedules = []
        for schedule in self.specs_schedule:
            initial = schedule['Initial']
            interval = schedule['Interval']
            factor = schedule['Factor']
            self.lr_schedules.append(
                LearningRateSchedule(initial, interval, factor)
                )

    def _init_data_loader(self):
        self.sdf_dataset = SDFSamples(self.train_paths, self.subsample_num)
        self.sdf_loader = torch.utils.data.DataLoader(
            self.sdf_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.dataset_len = len(self.sdf_dataset)

    def _init_latent_vecs(self):
        self.lat_vecs = torch.nn.Embedding(
            self.dataset_len, 
            self.latent_size, 
            max_norm=self.code_bound
        )

        torch.nn.init.normal_(
            self.lat_vecs.weight.data,
            0.0,
            1 / (self.latent_size ** 0.5),
        )

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            [
                {
                    'params': self.decoder.parameters(),
                    'lr': self.lr_schedules[0].get_learning_rate(0),
                },
                {
                    'params': self.lat_vecs.parameters(),
                    'lr': self.lr_schedules[1].get_learning_rate(0),
                },
            ]
        )

    def _adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.lr_schedules[i].get_learning_rate(epoch)

    def _save_snapshot(self, epoch):
        self.save_model(epoch)
        self.save_optimizer(epoch)
        self.save_latent_vectors(epoch)

    def save_model(self, epoch):
        file_path = os.path.join(self.results_path, 'model_{}.pt'.format(epoch))
        torch.save(
            {
                "epoch": epoch, 
                "model_state_dict": self.decoder.state_dict()
            },
            file_path
        )
    
    def save_optimizer(self, epoch):
        file_path = os.path.join(self.results_path, 'optimizer_{}.pt'.format(epoch))
        torch.save(
            {
                "epoch": epoch, 
                "optimizer_state_dict": self.optimizer.state_dict()
            },
            file_path
        )
    
    def save_loss(self, epoch, losses):
        file_path = os.path.join(self.results_path, 'losses.pt')
        torch.save(
            {
                "epoch": epoch,
                "loss": losses
            },
            file_path
        )
     

    def save_latent_vectors(self, epoch):
        file_path = os.path.join(self.results_path, 'latent_vecs_{}.pt'.format(epoch))
        all_latents = self.lat_vecs.state_dict()
        torch.save(
            {
                "epoch": epoch, 
                "latent_codes": all_latents
            },
            file_path
        )

    def train(self):
        loss_l1 = torch.nn.L1Loss(reduction='sum')
        for epoch in range(self.num_epochs):
            print('Epoch {}'.format(epoch))

            self.decoder.train()
            self._adjust_learning_rate(epoch)
            loss_log = []

            for b_iter, (sdf_data, indices) in enumerate(self.sdf_loader):
                # Stack all points in a 2D tensor 
                sdf_data = sdf_data.reshape(-1, 4)
                
                num_sdf_samples = sdf_data.shape[0]
                sdf_data.requires_grad = False
                xyz = sdf_data[:, 0:3]
                
                sdf_gt = sdf_data[:, 3].unsqueeze(1)
                sdf_gt = torch.clamp(sdf_gt, self.minT, self.maxT)

                # Make chunks
                xyz = torch.chunk(xyz, self.batch_split)
                indices = torch.chunk(
                    indices.unsqueeze(-1).repeat(1, self.subsample_num).view(-1),
                    self.batch_split,
                )
                sdf_gt = torch.chunk(sdf_gt, self.batch_split)
                
                batch_loss = 0.0
                self.optimizer.zero_grad()

                for i in range(self.batch_split):
                    batch_vecs = self.lat_vecs(indices[i])
                    
                    input = torch.cat([batch_vecs, xyz[i]], dim=1)

                    # NN optimization
                    pred_sdf = self.decoder(input.float())

                    pred_sdf = torch.clamp(pred_sdf, self.minT, self.maxT)

                    chunk_loss = loss_l1(pred_sdf, sdf_gt[i]) / num_sdf_samples

                    if self.do_code_regularization:
                        l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                        reg_loss = (
                            self.code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                        ) / num_sdf_samples

                        chunk_loss = chunk_loss + reg_loss

                    chunk_loss.backward()
                    batch_loss += chunk_loss.item()
                
                print("Batch {}: loss = {}".format(b_iter, batch_loss))
                loss_log.append(batch_loss)

                self.optimizer.step()
                
            if epoch % self.snapshot_frequency == 0:
                self._save_snapshot(epoch)
            self.save_loss(epoch, loss_log)
