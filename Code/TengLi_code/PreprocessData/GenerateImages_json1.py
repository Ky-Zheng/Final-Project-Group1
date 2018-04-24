import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os

#file1_path is the path of json file with 3 band data for each image
file1_path = './input/train_add_band3.json'

#I ziped the json file and store under the input folder, in case the data file doesn't exists
if not os.path.exists(file1_path):
    print("unzipping file...")
    file1_zip_path = './input/train_add_band3.json.zip'
    file1_extract_path = './input/'
    zip_ref = zipfile.ZipFile(file1_zip_path, 'r')
    zip_ref.extractall(file1_extract_path)
    zip_ref.close()
else:
    print("json file exists, reading data...")

#where images save
ice_file_path = './data_1/1/'
ship_file_path = './data_1/0/'

if not os.path.exists(ice_file_path):
    os.makedirs(ice_file_path)
if not os.path.exists(ship_file_path):
    os.makedirs(ship_file_path)

train_1 = pd.read_json(file1_path)

##set index of image
#num_ice = INDEX_OF_ICE_IMAGE
#num_ship = INDEX_OF_SHIP_IMAGE

for i in range(len(train_1['is_iceberg'])):
#    if i >= INDEX_OF_IMAGE:
        print(i+1, "/", len(train_1['is_iceberg']))
        plt.figure(0)
        plt.xticks(())
        plt.yticks(())
        if str(train_1['is_iceberg'][i]) == "0":
            plt.imshow(np.array(train_1["band_1"][i]).reshape(75, 75))
            plt.savefig(ship_file_path + 'ship_1_' + str(num_ship) + '.jpg')
            num_ship += 1

            plt.imshow(np.array(train_1["band_2"][i]).reshape(75, 75))
            plt.savefig(ship_file_path + 'ship_1_' + str(num_ship) + '.jpg')
            num_ship += 1

            plt.imshow(np.array(train_1["band_3"][i]).reshape(75, 75))
            plt.savefig(ship_file_path + 'ship_1_' + str(num_ship) + '.jpg')
            num_ship += 1
        else:
            if str(train_1['is_iceberg'][i]) == "1":
                plt.imshow(np.array(train_1["band_1"][i]).reshape(75, 75))
                plt.savefig(ice_file_path + 'ice_1_' + str(num_ice) + '.jpg')
                num_ice += 1

                plt.imshow(np.array(train_1["band_2"][i]).reshape(75, 75))
                plt.savefig(ice_file_path + 'ice_1_' + str(num_ice) + '.jpg')
                num_ice += 1

                plt.imshow(np.array(train_1["band_3"][i]).reshape(75, 75))
                plt.savefig(ice_file_path + 'ice_1_' + str(num_ice) + '.jpg')
                num_ice += 1

