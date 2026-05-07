import sys
import torch
def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
  if x_out.dim() == 1:
    x_out = x_out.unsqueeze(0)
    x_tf_out = x_tf_out.unsqueeze(0)
  _, k = x_out.size()
  bn_, k_ = x_out.size()
  assert (x_tf_out.size(0) == bn_ and x_tf_out.size(1) == k_)
  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  
  p_i_j = p_i_j.sum(dim=0)  
  p_i_j = (p_i_j + p_i_j.t()) / 2.  
  p_i_j = p_i_j / p_i_j.sum()  
  assert (p_i_j.size() == (k, k))
  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
  p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  
  p_i_j[(p_i_j < EPS).data] = EPS
  loss = - p_i_j * (torch.log(p_i_j+1e-5)                    - lamb * torch.log(p_j+1e-5)                    - lamb * torch.log(p_i+1e-5))
  loss = loss.sum()
  return loss
def compute_joint(x_out, x_tf_out):
  bn, k = x_out.size()
  assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)
  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  
  p_i_j = p_i_j.sum(dim=0)  
  p_i_j = (p_i_j + p_i_j.t()) / 2.  
  p_i_j = p_i_j / p_i_j.sum()  
  return p_i_j