##!/bin/bash

## Define source and destination directories
#Where you have extracted the RGB or Depth archive
source_dir="/mnt/data-tmp/ghazal/DARai_DATA/"
#Where you want to create the split the data
test_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/depth_dataset/test/depth_2"
validation_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/depth_dataset/validation/depth_2"
train_dest_dir="/mnt/data-tmp/ghazal/DARai_DATA/depth_dataset/train/depth_2"

for activity_dir in "$source_dir"/*; do
    activity_name=$(basename "$activity_dir")

    # Create destination directories
    mkdir -p "$test_dest_dir/$activity_name"
    mkdir -p "$validation_dest_dir/$activity_name"
    mkdir -p "$train_dest_dir/$activity_name"

#    # Sync test patterns
    find "$activity_dir/Depth/camera_2_fps_15" -type f \( -name '10_*.png' -o -name '16_*.png' -o -name '19_*.png' \) -exec rsync -azvh --ignore-existing {} "$test_dest_dir/$activity_name" \;

    # Sync validation patterns
    find "$activity_dir/Depth/camera_2_fps_15" -type f \( -name '02_*.png' -o -name '20_*.png' \) -exec rsync -azvh --ignore-existing {} "$validation_dest_dir/$activity_name" \;

    # Sync training patterns (everything else)
    find "$activity_dir/Depth/camera_2_fps_15" -type f -not \( -name '10_*.png' -o -name '16_*.png' -o -name '19_*.png' -o -name '02_*.png' -o -name '20_*.png' \) -exec rsync -azvh --ignore-existing {} "$train_dest_dir/$activity_name" \;
done