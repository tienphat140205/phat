import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from models.DualTopic import DualTopic

class Runner:
    def __init__(self, args, params_list, lang1, lang2):
        """Khởi tạo Runner với mô hình DualTopic và thiết bị."""
        self.args = args
        self.lang1 = lang1
        self.lang2 = lang2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DualTopic(*params_list).to(self.device)

    def make_optimizer(self):
        """Tạo optimizer Adam."""
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def make_lr_scheduler(self, optimizer):
        """Tạo scheduler StepLR."""
        if getattr(self.args, 'lr_scheduler', None) != 'StepLR':
            raise NotImplementedError("Only StepLR is supported")
        return StepLR(optimizer, step_size=self.args.lr_step_size, gamma=self.args.lr_gamma, verbose=False)

    def train(self, data_loader):
        """Huấn luyện mô hình và trả về beta cùng các loss."""
        data_size = len(data_loader.dataset)
        num_batch = len(data_loader)
        optimizer = self.make_optimizer()
        lr_scheduler = self.make_lr_scheduler(optimizer) if getattr(self.args, 'lr_scheduler', None) == 'StepLR' else None

        all_epoch_losses = []
        self.model.gamma_align = self.args.gamma_align

        for epoch in range(1, self.args.epochs + 1):
            total_loss = 0.0
            batch_losses = defaultdict(float)
            self.model.train()

            for batch in data_loader:
                bow_lang1 = batch['bow_lang1'].to(self.device, non_blocking=True)
                bow_lang2 = batch['bow_lang2'].to(self.device, non_blocking=True)
                cluster_info = batch.get('cluster_info', None)
                if isinstance(cluster_info, torch.Tensor):
                    cluster_info = cluster_info.tolist()

                outputs = self.model(bow_lang1, bow_lang2, cluster_info)
                loss = outputs['total_loss']

                for key in ['tm_loss_lang1', 'tm_loss_lang2', 'contrast_loss', 'align_loss']:
                    batch_losses[key] += outputs[key].item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(bow_lang1)

            if lr_scheduler:
                lr_scheduler.step()

            avg_loss = total_loss / data_size
            epoch_losses = {
                'total_loss': avg_loss,
                'tm_loss': (batch_losses['tm_loss_lang1'] + batch_losses['tm_loss_lang2']) / num_batch,
                'contrast_loss': batch_losses['contrast_loss'] / num_batch,
                'align_loss': batch_losses['align_loss'] / num_batch
            }
            all_epoch_losses.append(epoch_losses)

            print(f"Epoch: {epoch:03d}, Total Loss: {avg_loss:.3f}, "
                  f"TM Loss: {epoch_losses['tm_loss']:.3f}, "
                  f"Contrast Loss: {epoch_losses['contrast_loss']:.3f}, "
                  f"Align Loss: {epoch_losses['align_loss']:.3f}")

        beta_lang1, beta_lang2 = self.model.get_beta()
        return {
            'beta_lang1': beta_lang1.detach().cpu().numpy(),
            'beta_lang2': beta_lang2.detach().cpu().numpy(),
            'losses': all_epoch_losses
        }

    def get_theta(self, data_loader, lang):
        """Lấy phân phối theta cho một ngôn ngữ."""
        self.model.eval()
        theta_list = []
        with torch.no_grad():
            for batch in data_loader:
                bow = batch[f'bow_{lang}'].to(self.device, non_blocking=True)
                theta = self.model.get_theta(bow, lang)
                theta_list.append(theta.detach().cpu().numpy())
        return np.concatenate(theta_list, axis=0)

    def test(self, data_loader):
        """Lấy theta cho cả hai ngôn ngữ."""
        return self.get_theta(data_loader, 'lang1'), self.get_theta(data_loader, 'lang2')