## Statoil/C-CORE Iceberg Classifier Challenge from Kaggle

https://www.kaggle.com/c/statoil-iceberg-classifier-challenge

## Getting started

### Prerequisites
Environment: Python3 
Frameworks: Pytorch, Caffe, Keras

###Datasets
Source: https://drive.google.com/drive/folders/1Y8XDPPWFv_BaP6207K23d3WnmeqC1eBa?usp=sharing
Original: train.json, test.json (from Kaggle)
After adding band_3: data_1(including train and validation)
After data augmentation : data_2_train, data_2_val

## Data preprocess

###Adding band_3 to original data using formula: band_3 = band_1+ band_2
The output file is data_ 1(including train and validation).

###Data Augmentation based on band_3
Apply data augmentation on band_3 and then add all data together. 
The output file are: data_2_train, data_2_val.

Our models will be trained on data_1 and data_2_train/val separately in order to check the performance using data augmentation.

##Training on CNN models in three different frameworks
###Python

###Caffe

###Keras
     

