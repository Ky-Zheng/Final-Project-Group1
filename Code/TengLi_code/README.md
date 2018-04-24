This repo is for the competition on Kaggle. The competition is Statoil/C-CORE Iceberg Classifier Challenge. The website is https://www.kaggle.com/c/statoil-iceberg-classifier-challenge.


PreprocessData is for preprocessing data and data augmentation.
	1. input folder stores the raw training json file, train.json. 
	2. Preprocess.py is for preprocess data. Add two band, "band_1" and "band_2" to "band_3" and save the 3rd band to a new json file (train_add_band3.json) The input is the train.json
	3. Data_augmentation.py is for data augmentation. Flip each image to horizontal, vertical, and rotate to 90 degrees, then save to a new json file (train_addDA.json)
	4. GenerateImages_json1.py is for generating images from the train_add_band3.json file.
	5. GenerateImages_json2.py is for generating images from the train_addDA.json file.
	6. SeparateData.py is for separate all data to validation data and training data. train:validation = 8:2
The order of running this folder follows: 2 -> 3 -> 4 -> 5 -> 6
NOTE: Before running SeparateData.py, please add images from ./input/data_1 to ./input/data_2



CNNModel is CNN model on Caffe.
	1. data folder stores 2 datasets, data_1 and data_2
	2. Network.prototxt -- CNN network(4 layers)
	3. Solver.prototxt -- solver prototxt file
	4. Train.py -- training python file
	5. Create_LMDB.py -- create lmdb database python file
The order of running this folder follows: 5 -> 4

Data resource: https://drive.google.com/drive/folders/1Y8XDPPWFv_BaP6207K23d3WnmeqC1eBa?usp=sharing
