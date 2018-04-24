import os
import shutil

#separate data
#train:test -> 8:2
#before run this script, please add images from ./input/data_1 to ./input/data_2


file_train_ice_path = './data_group2/train/1/'
file_train_ship_path = './data_group2/train/0/'
file_test_ice_path = './data_group2/test/1/'
file_test_ship_path = './data_group2/test/0/'

os.system('rm -r ' + file_train_ice_path)
os.system('rm -r ' + file_train_ship_path)
os.system('rm -r ' + file_test_ice_path)
os.system('rm -r ' + file_test_ship_path)


if not os.path.exists(file_train_ice_path):
    os.makedirs(file_train_ice_path)
if not os.path.exists(file_train_ship_path):
    os.makedirs(file_train_ship_path)
if not os.path.exists(file_test_ice_path):
    os.makedirs(file_test_ice_path)
if not os.path.exists(file_test_ship_path):
    os.makedirs(file_test_ship_path)

ice_image_path = './data_2/1/'
ship_image_path = './data_2/0/'

#ice images
num_ice = 0
for dirpath, subdirs, files in os.walk(ice_image_path):
    for file in files:
        num_ice += 1

print("number of ice image: ", num_ice)

train_num = 0.8 * num_ice
test_num = 0.2 * num_ice

for dirpath, subdirs, files in os.walk(ice_image_path):
    count = 0
    for file in files:
        if count < train_num:
            shutil.copy2(ice_image_path + file, file_train_ice_path)
        else:
            shutil.copy2(ice_image_path + file, file_test_ice_path)

        count += 1

#ship images
num_ship = 0
for dirpath, subdirs, files in os.walk(ship_image_path):
    for file in files:
        num_ship += 1

print("number of ship images: ", num_ship)

train_num = 0.8 * num_ship
test_num = 0.2 * num_ship

for dirpath, subdirs, files in os.walk(ship_image_path):
    count = 0
    for file in files:
        if count < train_num:
            shutil.copy2(ship_image_path + file, file_train_ship_path)
        else:
            shutil.copy2(ship_image_path + file, file_test_ship_path)
        count += 1

#print number of image
num_train_ice = 0
num_test_ice = 0
num_train_ship = 0
num_test_ship = 0
for dirpath, subdirs, files in os.walk(file_train_ice_path):
    for file in files:
        num_train_ice += 1
print("train data of ice = ", num_train_ice)

for dirpath, subdirs, files in os.walk(file_test_ice_path):
    for file in files:
        num_test_ice += 1
print("test data of ice = ", num_test_ice)

for dirpath, subdirs, files in os.walk(file_train_ship_path):
    for file in files:
        num_train_ship += 1
print("train data of ship = ", num_train_ship)

for dirpath, subdirs, files in os.walk(file_test_ship_path):
    for file in files:
        num_test_ship += 1
print("test data of ship= ", num_test_ship)
