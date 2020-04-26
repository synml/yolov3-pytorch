#!/bin/bash

# cd to dataset directory.
cd ../../data/coco

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

# Move list files and unzip annotations file.
mv ./yolov3/*.txt ./
tar xzf ./yolov3/yolov3_annotations.tgz

# Delete yolov3 directory
rm -r ./yolov3
