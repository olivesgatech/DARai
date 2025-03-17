# Action Anticipation with DARAI Dataset

## 📂 Data Preparation

1. Download the dataset from our [dataport](https://ieee-dataport.org/open-access/darai-daily-activity-recordings-ai-and-ml-applications).  
2. Unzip `features_img.zip` into the following directory:  

   ```bash
   FUTR_proposed/datasets/darai/features_img
   ```

## 🚀 Training

To start training the model, run the following command:

```bash
python3 FUTR_proposed/main_darai.py --mode=train
```

## 🔍 Inference
For making predictions, use:
```bash
python3 FUTR_proposed/main_darai.py --predict
```