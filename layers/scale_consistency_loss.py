import torch.nn.functional as F
import torch


def compute_scale_consistency_loss(final_features, intermediate_features):
    """
    计算多尺度特征一致性损失
    Args:
        final_features: 最终特征 [batch_size, feat_dim]
        intermediate_features: 中间特征 [batch_size, feat_dim]
        weight: 损失权重
    Returns:
        scale_consistency_loss: 一致性损失
    """
    if intermediate_features is None:
        return torch.tensor(0.0, device=final_features.device)
    
    # 特征归一化
    final_norm = F.normalize(final_features, p=2, dim=1)
    intermediate_norm = F.normalize(intermediate_features, p=2, dim=1)
    
    # 计算MSE损失
    mse_loss = F.mse_loss(final_norm, intermediate_norm)
    
    # 可选：添加余弦相似度损失
    cos_sim = F.cosine_similarity(final_norm, intermediate_norm, dim=1)
    cos_loss = 1 - cos_sim.mean()
    
    # 组合损失
    total_loss = mse_loss + 0.1 * cos_loss
    
    return total_loss