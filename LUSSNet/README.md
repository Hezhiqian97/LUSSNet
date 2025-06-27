##  LUSSNet: Lightweight Underwater Semantic Segmentation Network for Aquaculture Edge Devices
---

### Training steps
  

1. This paper uses the VOC format for training.  
2. Before training, place the label file in the SegmentationClass folder under the VOC2007 folder in the VOCdevkit folder.    
3. Before training, place the image files in the JPEGImages folder under the VOC2007 folder in the VOCdevkit folder.     
4. Note that you need to modify the num_classes parameter in train.py to the number of classes plus one.    
5. Run train.py to start training.




### Prediction steps
#### 一、Use training weights
##### a、SIUM training weights
1. After downloading the library, unzip it. If you want to use the weights trained by SUIM for prediction, download the weights from Githup, place them in model_data, and run to predict.
```python
img/street.jpg
```    
2. Settings in predict.py allow for fps testing and video detection.   

### 评估步骤
1. Set num_classes in get_miou.py to the number of classes to be predicted plus 1.  
2. Set name_classes in get_miou.py to the categories to be distinguished.  
3. Run get_miou.py to obtain the MIOU size.

