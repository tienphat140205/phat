import torch
import torch.nn as nn
import torch.nn.functional as F
from ot import sinkhorn
from models.networks.Encoder import MLPEncoder

class DualTopic(nn.Module):
    def __init__(self, vocab_size_lang1, vocab_size_lang2, num_topics, hidden_dim, word_embeddings_lang1, word_embeddings_lang2, dropout=0.3, tau=1.0, lambda_contrast=5, gamma_align=3, epsilon=0.05):
        super(DualTopic, self).__init__()
        self.num_topics = num_topics
        self.tau = tau
        self.lambda_contrast = lambda_contrast
        self.gamma_align = gamma_align
        self.epsilon = epsilon

        assert word_embeddings_lang1.shape[1] == word_embeddings_lang2.shape[1], "Embedding dimensions mismatch"
        self.encoder_lang1 = MLPEncoder(vocab_size_lang1, num_topics, hidden_dim, dropout)
        self.encoder_lang2 = MLPEncoder(vocab_size_lang2, num_topics, hidden_dim, dropout)

        self.word_embeddings_lang1 = nn.Parameter(torch.from_numpy(word_embeddings_lang1).float(), requires_grad=False)
        self.word_embeddings_lang2 = nn.Parameter(torch.from_numpy(word_embeddings_lang2).float(), requires_grad=False)
        self.topic_embeddings_lang1 = nn.Parameter(torch.randn(num_topics, self.word_embeddings_lang1.shape[1]))
        self.topic_embeddings_lang2 = nn.Parameter(torch.randn(num_topics, self.word_embeddings_lang2.shape[1]))

    def compute_beta(self, word_embeddings, topic_embeddings):
        dist = torch.cdist(word_embeddings.unsqueeze(0), topic_embeddings.unsqueeze(0), p=2).squeeze(0)
        beta = torch.exp(-dist / self.tau)
        beta = beta / beta.sum(dim=0, keepdim=True)
        return beta.transpose(0, 1)
 
    def tm_loss(self, x_bow, theta, beta, mu, logvar):
        recon = torch.matmul(theta, beta)
        recon_loss = -torch.sum(x_bow * torch.log_softmax(recon, dim=-1), dim=-1).mean()
        kl_loss = 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar, dim=-1).mean()
        return recon_loss + kl_loss

    def contrastive_loss(self, theta_lang1, theta_lang2, cluster_info, batch_size):
        """Tính contrastive loss với phiên bản vector hóa."""
        theta_all = torch.cat([theta_lang1, theta_lang2], dim=0)
        cluster_all = [c[0] for c in cluster_info] + [c[1] for c in cluster_info]
        
        # Chuẩn hóa vector và tính similarity matrix
        theta_norm = F.normalize(theta_all, dim=-1)
        sim_matrix = torch.matmul(theta_norm, theta_norm.T) / self.tau
        sim_exp = torch.exp(sim_matrix)
        
        # Loại bỏ similarity của chính nó
        eye_mask = torch.eye(2 * batch_size, device=theta_all.device, dtype=torch.bool)
        sim_exp = sim_exp * (~eye_mask).float()
        
        # Tạo mask cho positive samples
        cluster_all_tensor = torch.tensor(cluster_all, device=theta_all.device)
        pos_mask = (cluster_all_tensor.unsqueeze(0) == cluster_all_tensor.unsqueeze(1)).float()
        pos_mask = pos_mask * (~eye_mask).float()
        
        # Tính positive và negative sum
        pos_sim = torch.sum(sim_exp * pos_mask, dim=1)
        neg_sum = torch.sum(sim_exp, dim=1) - sim_exp.diag()
        
        # Tính loss
        loss_per_anchor = -torch.log(pos_sim / (pos_sim + neg_sum + 1e-8) + 1e-8)
        valid_anchors = (pos_sim > 0).float()
        count = torch.sum(valid_anchors)
        total_loss = torch.sum(loss_per_anchor * valid_anchors)
        return total_loss / (count + 1e-8)

    def alignment_loss(self):
        # Lấy topic embeddings
        t1 = self.topic_embeddings_lang1
        t2 = self.topic_embeddings_lang2
        # Tính cosine similarity
        cos_sim = F.cosine_similarity(t1.unsqueeze(1), t2.unsqueeze(0), dim=-1)
        # Tạo ma trận chi phí và chuẩn hóa
        C = 1 - cos_sim
        C = torch.clamp(C, min=0.0, max=1.0)  # Đảm bảo C nằm trong [0, 1]
        # Tạo phân phối đều a, b với no_grad để đảm bảo không tính gradient
        with torch.no_grad():
            a = torch.ones(self.num_topics, device=C.device) / self.num_topics
            b = torch.ones(self.num_topics, device=C.device) / self.num_topics
        # Tính Sinkhorn với numItermax cao hơn để tránh warning
        Q = sinkhorn(a, b, C, self.epsilon, numItermax=1000)
        # Tính OT loss
        ot_loss = torch.sum(Q * C)
        return ot_loss

    def forward(self, x_bow_lang1, x_bow_lang2, cluster_info=None):
        theta_lang1, mu_lang1, logvar_lang1 = self.encoder_lang1(x_bow_lang1)
        theta_lang2, mu_lang2, logvar_lang2 = self.encoder_lang2(x_bow_lang2)
        beta_lang1 = self.compute_beta(self.word_embeddings_lang1, self.topic_embeddings_lang1)
        beta_lang2 = self.compute_beta(self.word_embeddings_lang2, self.topic_embeddings_lang2)
        tm_loss_lang1 = self.tm_loss(x_bow_lang1, theta_lang1, beta_lang1, mu_lang1, logvar_lang1)
        tm_loss_lang2 = self.tm_loss(x_bow_lang2, theta_lang2, beta_lang2, mu_lang2, logvar_lang2)
        contrast_loss = self.contrastive_loss(theta_lang1, theta_lang2, cluster_info, x_bow_lang1.shape[0]) if cluster_info else 0.0
        align_loss = self.alignment_loss()
        total_loss = tm_loss_lang1 + tm_loss_lang2 + self.lambda_contrast * contrast_loss + self.gamma_align * align_loss
        return {
            'total_loss': total_loss,
            'tm_loss_lang1': tm_loss_lang1,
            'tm_loss_lang2': tm_loss_lang2,
            'contrast_loss': contrast_loss,
            'align_loss': align_loss,
            'theta_lang1': theta_lang1,
            'theta_lang2': theta_lang2,
            'beta_lang1': beta_lang1,
            'beta_lang2': beta_lang2
        }

    def get_beta(self):
        """Trả về phân phối topic-word (beta) cho cả hai ngôn ngữ."""
        beta_lang1 = self.compute_beta(self.word_embeddings_lang1, self.topic_embeddings_lang1)
        beta_lang2 = self.compute_beta(self.word_embeddings_lang2, self.topic_embeddings_lang2)
        return beta_lang1, beta_lang2

    def get_theta(self, bow, lang):
        """Trả về phân phối topic (theta) cho một ngôn ngữ cụ thể."""
        self.eval()
        with torch.no_grad():
            if lang == 'lang1':
                theta, _, _ = self.encoder_lang1(bow)
            elif lang == 'lang2':
                theta, _, _ = self.encoder_lang2(bow)
            else:
                raise ValueError(f"Unsupported language: {lang}")
        return theta