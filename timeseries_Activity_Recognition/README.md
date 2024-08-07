# Multimodal HAR


For consistency in results, the versions that we used are:
* Python: `3.8.11`
* Pytorch: `1.7.1`
* Pytorch Lightning: `1.4.9`
* CUDA: `10.1`



# Datasets

Before you can run the training scripts, you must download the datasets and generate the expected folder structure.

## Downloading the data

### DARai

For DARai, download and extract the data yourself ([link](https://ieee-dataport.org/open-access/darai-daily-activity-recordings-ai-and-ml-aplications)), 


## Dataset preprocessing

```
python convert_emg.py --data_path '/path/to/emg/data/'
python convert_imu.py --data_path '/path/to/imu/data/'
python convert_bio.py --data_path '/path/to/bio/data/'
python convert_gaze.py --data_path '/path/to/gaze/data/'
python convert_insole.py --data_path '/path/to/insole/data/'
```


## Folder structure

The final folder structure could look like:
```
YOUR_PATH/
    data/
        darai/
            emg/
            inertial/
            bio/
            gaze/
            insole/
```

The training scripts require a `--data_path` parameter which must point to the root path of the dataset:
```
--data_path YOUR_PATH/data/darai/

```



# Training

```
python -m supervised_training  \
  --dataset l1 \
  --data_path /YOUR_PATH/data/darai/ \
  --model transformer \
  --experiment_config_path configs/supervised/emg_supervised.yaml

```




## Supervised training (unimodal and multimodal)

Example 1: emg unimodal training on DARai:

```
python -m supervised_training  \
  --dataset l1 \
  --data_path /YOUR_PATH/data/darai/ \
  --model transformer \
  --experiment_config_path configs/supervised/emg_supervised.yaml

```

Example 2: insole unimodal training on DARai:
```
python -m supervised_training  \
  --dataset l1 \
  --data_path /home/yavuz/ws/cmc-cmkm/data/darai/ \
  --model transformer \
  --experiment_config_path configs/supervised/insole_supervised.yaml
```

Example 3: multimodal fine-tuning of individually trained supervised encoders on DARai (emg + insole):
```
# Train supervised inertial and skeleton encoders separately (see example 1).
python -m supervised_training_mm ...
python -m supervised_training_mm ...

# Train a multimodal encoder.
python -m supervised_training_mm \
  --dataset l1 \
  --data_path /YOUR_PATH/data/darai/ \
  --modalities  emg insole  \
  --models transformer transformer  \
  --experiment_config_path configs/supervised/mm_all.yaml \
  --pre_trained_paths ./model_weights/emg/epoch.ckpt ./model_weights/insole/epoch.ckpt 
```

**Note:** by default, model checkpoints (parameters) are saved in `model_weights/`. The default path for the weights can be changed using the `--model_save_path` argument.





## Acknowledgements


The code is adopted from origial code repository of CMM-CMKM paper : https://github.com/razzu/cmc-cmkm

