import torch
import torch.nn as nn

'''
     Instantiate the object like:
          criterion = AdaptiveWingLoss(whetherWeighted=True)
     #param whetherWeighted: whether use weighted loss map
     #param dilaStru: size of dilation structure
'''
class AdaptiveWingLoss(nn.Module):
  def __init__(self, alpha=2.1, omega=14.0, theta=0.5, epsilon=1.0,\
               whetherWeighted=False, dilaStru=3, w=10, device="cuda"):
    super(AdaptiveWingLoss, self).__init__()
    self.device = device
    self.alpha = torch.Tensor([alpha]).to(device)
    self.omega = torch.Tensor([omega]).to(device)
    self.theta = torch.Tensor([theta]).to(device)
    self.epsilon = torch.Tensor([epsilon]).to(device)
    self.dilationStru = dilaStru
    self.w = torch.Tensor([w]).to(device)
    self.tmp = torch.Tensor([self.theta / self.epsilon]).to(device)
    self.wetherWeighted = whetherWeighted

'''
   #param predictions: predicted heat map with dimension of batchSize * landmarkNum * heatMapSize * heatMapSize  
   #param targets: ground truth heat map with dimension of batchSize * landmarkNum * heatMapSize * heatMapSize  
'''
  def forward(self, predictions, targets):
    deltaY = predictions - targets
    deltaY = torch.abs(deltaY)
    alphaMinusY = self.alpha - targets
    a = self.omega / self.epsilon * alphaMinusY / (1 + self.tmp.pow(alphaMinusY))\
        * self.tmp.pow(alphaMinusY - 1)
    c = self.theta * a - self.omega * torch.log(1 + self.tmp.pow(alphaMinusY))

    l = torch.where(deltaY < self.theta,
                    self.omega * torch.log(1 + (deltaY / self.epsilon).pow(alphaMinusY)),
                    a * deltaY - c)
    if self.wetherWeighted:
      weightMap = self.grayDilation(targets, self.dilationStru)
      weightMap = torch.where(weightMap >= 0.2, torch.Tensor([1]).to(self.device),\
                              torch.Tensor([0]).to(self.device))
      l = l * (self.w * weightMap + 1)

    l = torch.mean(l)

    return l
    
  def grayDilation(self, heatmapGt, structureSize):
    batchSize, landmarkNum, heatmapSize, _ = heatmapGt.shape
    weightMap = heatmapGt.clone()
    step = structureSize // 2
    for i in range(1, heatmapSize-1, 1):
      for j in range(1, heatmapSize-1, 1):
        weightMap[:, :, i, j] = torch.max(heatmapGt[:, :, i - step: i + step + 1,\
                                j - step: j + step + 1].contiguous().view(batchSize,\
                                landmarkNum, structureSize * structureSize), dim=2)[0]

    return weightMap  
