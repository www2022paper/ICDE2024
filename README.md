# Improve ROI with Causal Learning and Conformal Prediction


### ***Directory Structure***  UPDATED ON 2024.03.04
The code to replicate the offline results in **Section V. Experiments --> A. Offline Test B. Ablation Study**
```

|----- code_Criteo       # Various benchmark methods when the dataset is CRITEO-UPLIFT v2.
|----- code_MT           # Various benchmark methods when the dataset is Meituan-LIFT.
|----- code_Ali          # Various benchmark methods when the dataset is Alibaba-LIFT.
|-----figure
|     |-----xxx.pdf                 # The images resulting from the code in the aforementioned 'code' directory.
|     |-----xxx.csv                 # The intermediate results produced by the code in the 'code' directory, used for calculating AUCC and plotting.
|-----metric
|     |-----Metric.py               # The evaluation metrics: AUCC
|-----model
|     |-----uplift_model.py         # The model to predict CATE
|     |-----roi_model.py            # The model to predict ROI
|-----model_file
|     |-----xxx                     # Saved model files trained by various benchmark methods.       
|-----README.txt
```


### ***Three Real-world Public Industrial Dataset***
```
1a. Dataset name: CRITEO-UPLIFT v2
    Download link: https://ailab.criteo.com/criteo-uplift-prediction-dataset/, rename it as "criteo-uplift-v2.1.csv"
1b. Dataset name: Meituan-LIFT
    Download link: https://github.com/MTDJDSP/MT-LIFT, rename it as "/MT-LIFT/train.csv"
1c. Dataset name: Alibaba-LIFT
    Download link: https://tianchi.aliyun.com/dataset/94883, rename it as "Alibaba-lift.csv"
2. Make a "data" directory, and put these three datasets in the data directory.
```


### ***Setup Details***
```
1. Tensorflow 2.14.0 is used in this experiment.
2. Neural network ran on a machine with a GPU RTX 4090(24GB) and 90GB memory.
3. To support the use of Generalized Random Forests (GRF) , install econML from https://github.com/microsoft/EconML.
4. GRF ran on a machine with 32 vCPU (AMD EPYC 7742 64-Core Processor) and 96GB memory.

```

