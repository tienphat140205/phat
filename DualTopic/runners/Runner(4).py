import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from models.DualTopic import DualTopic

class Runner:
    def __init__(self, args, params_list, lang1, lang2):
        self.args = args
        self.lang1 = lang1
        self.lang2 = lang2
        self.model = DualTopic(*params_list)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.align_start_epoch = getattr(args, 'align_start_epoch', args.epochs + 1)

    def make_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if getattr(self.args, 'lr_scheduler', None) == 'StepLR':
            lr_scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma=self.args.lr_gamma, verbose=False)
        else:
            raise NotImplementedError("Only StepLR is supported")
        return lr_scheduler

    def train(self, data_loader):
        data_size = len(data_loader.dataset)
        num_batch = len(data_loader)
        optimizer = self.make_optimizer()
        lr_scheduler = None
        if getattr(self.args, 'lr_scheduler', None) == 'StepLR':
            lr_scheduler = self.make_lr_scheduler(optimizer)

        for epoch in range(1, self.args.epochs + 1):
            sum_loss = 0.
            loss_rst_dict = defaultdict(float)
            self.model.train()
            for batch_data in data_loader:
                batch_bow_lang1 = batch_data['bow_lang1']
                batch_bow_lang2 = batch_data['bow_lang2']
                cluster_info = batch_data.get('cluster_info', None)

                if epoch < self.align_start_epoch:
                    self.model.gamma_align = 0.0
                else:
                    self.model.gamma_align = self.args.gamma_align

                rst_dict = self.model(batch_bow_lang1, batch_bow_lang2, cluster_info)
                batch_loss = rst_dict['total_loss']

                for key in ['tm_loss_lang1', 'tm_loss_lang2', 'contrast_loss', 'align_loss']:
                    if key in rst_dict:
                        loss_rst_dict[key] += rst_dict[key].item()

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                sum_loss += batch_loss.item() * len(batch_bow_lang1)

            if lr_scheduler is not None:
                lr_scheduler.step()

            sum_loss /= data_size
            output_log = f'Epoch: {epoch:03d}, Total Loss: {sum_loss:.3f}'
            for key in loss_rst_dict:
                output_log += f', {key}: {loss_rst_dict[key] / num_batch:.3f}'
            print(output_log)

        beta_lang1, beta_lang2 = self.model.get_beta()
        return beta_lang1.detach().cpu().numpy(), beta_lang2.detach().cpu().numpy()

    def get_theta(self, data_loader, lang):
        theta_list = []
        self.model.eval()
        with torch.no_grad():
            for batch_data in data_loader:
                batch_bow = batch_data[f'bow_{lang}']
                theta = self.model.get_theta(batch_bow, lang)
                theta_list.append(theta.detach().cpu().numpy())
        return np.concatenate(theta_list, axis=0)

    def test(self, data_loader):
        theta_lang1 = self.get_theta(data_loader, 'lang1')
        theta_lang2 = self.get_theta(data_loader, 'lang2')
        return theta_lang1, theta_lang2