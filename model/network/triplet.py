import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, batch_size, hard_or_full, margin):
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin

    def forward(self, feature, label):
        # feature: [n, m, d], label: [n, m]
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).byte().view(-1)
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).byte().view(-1)

        dist = self.batch_dist(feature)
        mean_dist = dist.mean(1).mean(1)
        dist = dist.view(-1)
        # hard
        hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]
        hard_hn_dist = torch.min(torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]
        hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)

        hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)

        # non-zero full
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)
        full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)
        full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)

        full_loss_metric_sum = full_loss_metric.sum(1)
        full_loss_num = (full_loss_metric != 0).sum(1).float()

        full_loss_metric_mean = full_loss_metric_sum / full_loss_num
        full_loss_metric_mean[full_loss_num == 0] = 0

        return full_loss_metric_mean, hard_loss_metric_mean, mean_dist, full_loss_num

    def batch_dist(self, x):
        x2 = torch.sum(x ** 2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist
