#!/bin/bash

# You have to download archive file related to yolov3's coco dataset.
# Download link: https://drive.google.com/file/d/1wJn0GOGcWpHwe60tN3lP63cb28lEqemJ/view?usp=sharing
# And move the archive file to coco dataset directory.
# coco dataset directory is ../../data/coco (start at this project directory)

# cd to dataset directory.
cd ../../data/coco

# Download coco dataset archive files.
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

# Make images directory and cd.
mkdir images
cd images

# Unzip and delete archive files.
unzip -q ../train2014.zip
unzip -q ../val2014.zip
rm ../train2014.zip
rm ../val2014.zip

# cd to ../../data/coco
cd ..

# Unzip archive file related to yolov3's coco dataset.
unzip yolov3.zip
rm yolov3.zip

# Move list files and unzip annotations file.
mv ./yolov3/*.txt ./
tar xzf ./yolov3/yolov3_annotations.tgz

# Delete yolov3 directory.
rm -r ./yolov3
