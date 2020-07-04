# adaptive-wing-loss
Pytorch implementation of adaptive wing loss and weighted adaptive wing loss

# Performance of adaptive wing loss
In my experiments, the performance of adaptive wing loss is excellent.
I get a 3.16% inter-ocular distance normalized mean error（ION NME）on 300W valid set （fullset）
in a facial landmark detection task。

# Usage:
    from adaptiveWingLoss import AdaptiveWingLoss
    
    # Instantiate the AdaptiveWingLoss like the way that instantiate MSELoss
    criterion = AdaptiveWingLoss(whetherWeighted=True) 
    # criterion = torch.nn.MSELoss(size_average=True)
    
    for epoch in range(maxEpochNum):
      for img, heatMapGT in dataloader:
         
         # get the predicted heatmap
         heatMapPredicted = model(img)
         
         # calculate adaptive wing loss
         # Note that the dimension of heatMapPredicted and heatMapGT is
         # batchSize * keyPointsNum * heatMapSize * heatMapSize
         loss = criterion(heatMapPredicted, heatMapGT)
         
         # Optimize the model by adaptive wing loss
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         
 
# Thanks the great work from Wang, Xinyao.
# Paper: 
    Wang X , Bo L , Fuxin L . Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression[C]// 2019 IEEE/CVF International Conference on Computer Vision (ICCV). arXiv, 2019.
