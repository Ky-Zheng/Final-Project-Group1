import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

ice_file_path = './data_2/1/'
ship_file_path = './data_2/0/'

if not os.path.exists(ice_file_path):
    os.makedirs(ice_file_path)
if not os.path.exists(ship_file_path):
    os.makedirs(ship_file_path)

file2_path = './input/train_addDA.json'
if not os.path.exists(file2_path):
    print("unzipping file...")
    file1_zip_path = './input/train_addDA.json.zip'
    file1_extract_path = './input/'
    zip_ref = zipfile.ZipFile(file1_zip_path, 'r')
    zip_ref.extractall(file1_extract_path)
    zip_ref.close()
else:
    print("json file exists, reading data...")

train_2 = pd.read_json(file2_path)

##set index of image
#num_ice = INDEX_OF_ICE_IMAGE
#num_ship = INDEX_OF_SHIP_IMAGE

# generate images from 2nd file(band_h, band_v, band_r)
for i in range(len(train_2['is_iceberg'])):
#    if i >= INDEX_OF_IMAGE:
        print(i+1, "/", len(train_2['is_iceberg']))
        plt.figure(0)
        plt.xticks(())
        plt.yticks(())
        if str(train_2['is_iceberg'][i]) == "0":
            plt.imshow(np.array(train_2["band_h_3"][i]).reshape(75, 75))
            plt.savefig(ship_file_path + 'ship_2_' + str(num_ship) + '.jpg')
            num_ship += 1

            plt.imshow(np.array(train_2["band_v_3"][i]).reshape(75, 75))
            plt.savefig(ship_file_path + 'ship_2_' + str(num_ship) + '.jpg')
            num_ship += 1

            plt.imshow(np.array(train_2["band_r_3"][i]).reshape(75, 75))
            plt.savefig(ship_file_path + 'ship_2_' + str(num_ship) + '.jpg')
        else:
            if str(train_2['is_iceberg'][i]) == "1":
                plt.imshow(np.array(train_2["band_h_3"][i]).reshape(75, 75))
                plt.savefig(ice_file_path + 'ice_2_' + str(num_ice) + '.jpg')
                num_ice += 1

                plt.imshow(np.array(train_2["band_v_3"][i]).reshape(75, 75))
                plt.savefig(ice_file_path + 'ice_2_' + str(num_ice) + '.jpg')
                num_ice += 1

                plt.imshow(np.array(train_2["band_r_3"][i]).reshape(75, 75))
                plt.savefig(ice_file_path + 'ice_2_' + str(num_ice) + '.jpg')
