# Multimodal HAR


For consistency in results, the versions that we used are:
* Python: `3.10.14`
* Pytorch: `2.2.2`
* torchvision : `0.17.2`
* CUDA: `12.1`



# Datasets

Before you can run the training scripts, you must download the datasets and generate the expected folder structure.
For this specific experiment you need to download at least one of the RGB modality and/or Depth modality folder.

## Downloading the data

### DARai

For DARai, download and extract the data yourself ([link](https://ieee-dataport.org/open-access/darai-daily-activity-recordings-ai-and-ml-aplications)), 

### Split data for training and evaluation

1-Run following bash file to create train, validation and test splits
2-Edit the source and destination in the file according to your prefrence.
Test subjects id: 10 , 16 , 19
Validation subjects id: 02 , 20

'''
cd ~path to this directory
bash ./split data.sh
'''

## Folder structure

The final folder structure could look like:
```
YOUR_PATH/
        darai_data/
            RGB/
            Activities/    #EX: Writing
            camera views/  #EX: camera_1_fps_15

```

The training scripts require a data path and a base path for models and figures and checkpoints
```
--data_dir
--base_dir

```



# Training

```
python -m "./video_action_recognition/run_experiments.py" /
 --epochs 10 /
 --data_dir "path to/rgb_dataset/" /
 --base_dir "path to/Activity_Recognition_benchmarking/" /
 --device your_device /
 --env "livingroom" /
 --batch_size 16 /
 --cam_view "cam_1" /
 --backbone "MViT" /
 --weights True

```


**Note:** by default, model checkpoints (parameters), figures and results are saved in `base_dir/`. 




## Acknowledgements



