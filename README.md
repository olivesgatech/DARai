# DARai: 
## Abstract
The DARai dataset is a comprehensive multimodal multi-view collection designed to capture daily activities in diverse indoor environments. This dataset incorporates 20 heterogeneous modalities, including environmental sensors, biomechanical measures, and physiological signals, providing a detailed view of human interactions with their surroundings. The recorded activities cover a wide range, such as office work, household chores, personal care, and leisure activities, all set within realistic contexts. The structure of DARai features a hierarchical annotation scheme that classifies human activities into multiple levels: activities, actions, procedures, and interactions. This taxonomy aims to facilitate a thorough understanding of human behavior by capturing the variations and patterns inherent in everyday activities at different levels of granularity. DARai is intended to support research in fields such as activity recognition, contextual computing, health monitoring, and human-machine collaboration. By providing detailed, contextual, and diverse data, this dataset aims to enhance the understanding of human behavior in naturalistic settings and contribute to the development of smart technology applications. Further details along with the dataset links and codes are available at [this link](https://alregib.ece.gatech.edu/darai-daily-activity-recordings-for-artificial-intelligence-and-machine-learning/).

## Dataset
Access the **dataset** [here](https://ieee-dataport.org/open-access/darai-daily-activity-recordings-ai-and-ml-applications).
dataset structure are based on data modalities as the top folder and under each modality we have Activity labels (level 1).

There will be some super group of modalities which contains at least 2 data modality folder. For some group such as bio monitors we are providing all 4 data modality under these group. 
```
Local Dataset Path/ 

└── Modality/ 				#Ex: RGB

    └── Activity Label / 

        └── View/ 

            └── data samples
```
You can download each group of modalities separatly and use them in your machine learning and deep learning pipeline.
The structure appears as the following:


## Code Usage

For Timeseries activity recognition go to: [this codebase](https://github.com/olivesgatech/DARai/tree/main/timeseries_Activity_Recognition)

For RGB/D activity recognition go to: [this codebase](https://github.com/olivesgatech/DARai/tree/main/video_Action_Recognition)

For action anticipation go to: [this codebase](https://github.com/olivesgatech/DARai/tree/main/video_Action-Anticipation)

For VQA benchmarking go to: [this codebase](https://github.com/olivesgatech/DARai/tree/main/video_VQA_Benchmarking)


## Citation

@article{kaviani2025hierarchical,
  title={Hierarchical and Multimodal Data for Daily Activity Understanding},
  author={Kaviani, Ghazal and Yarici, Yavuz and Kim, Seulgi and Prabhushankar, Mohit and AlRegib, Ghassan and Solh, Mashhour and Patil, Ameya},
  journal={arXiv preprint arXiv:2504.17696},
  year={2025}
}

@data{ecnr-hy49-24,
doi = {10.21227/ecnr-hy49},
author = {Ghazal Kaviani and Yavuz Yarici and Mohit Prabhushankar and Ghassan AlRegib and Mashhour Solh and Ameya Patil},
publisher = {IEEE Dataport},
title = {DARai: Daily Activity Recordings for AI and ML applications},
year = {2024} }
