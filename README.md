# Improve ROI with Causal Learning and Conformal Prediction


### ***Directory Structure***
The code to replicate the offline results in **Section V. Experiments --> A. Offline Simulation**
```
|-----code
|     |-----【InCo】xxx.ipynb       # Various benchmark methods when the setting is Insufficient data and Covariant shift.
|     |-----【InNo】xxx.ipynb       # Various benchmark methods when the setting is Insufficient data and No covariant shift.
|     |-----【SuCo】xxx.ipynb       # Various benchmark methods when the setting is Sufficient data and Covariant shift.
|     |-----【SuNo】xxx.ipynb       # Various benchmark methods when the setting is Sufficient data and No covariant shift.
|     |-----【SuNo】xxx.ipynb       # Various benchmark methods when the setting is Sufficient data and No covariant shift.
|     |-----【Offline Evaluation】xxx.ipynb       # Notebook codes that ultimately generate Fig.5 and Table 1.
|     |----- Fig.1【a】and Fig.1【b】.ipynb       # Notebook codes that generate Fig.1a and Fig.1b.
|-----figure
|     |-----xxx.pdf                 # The images resulting from the code in the aforementioned 'code' directory.
|     |-----xxx.csv                 # The intermediate results produced by the code in the 'code' directory, used for calculating AUCC and plotting.
|-----metric
|     |-----Metric.py               # The evaluation metrics: AUCC
|-----model
|     |-----uplift_model.py         # The model to predict CATE
|     |-----roi_model.py            # The model to predict ROI
|-----model_file
|     |-----xxx                     # Save the model files trained by various benchmark methods.       
|-----README.txt
```


### ***Public Dataset***
```
1. Dataset name: CRITEO-UPLIFT v2
2. Download link: https://ailab.criteo.com/criteo-uplift-prediction-dataset/, rename it as "criteo-uplift-v2.1.csv"
3. Make a "data" directory, and put this dataset in the data directory.
```


### ***Setup Details***
```
1. Tensorflow 2.14.0 is used in this experiment.
2. To support the use of Generalized Random Forests (GRF) , install econML from https://github.com/microsoft/EconML.
```

