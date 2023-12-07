
## NOTE: WE ARE PREPARING AND CLEANING THE CODE. THE CODE WILL BE READY BY 2023/12/09. THANKS.


# Improve ROI with Causal Learning and Conformal Prediction

The code to replicate the offline results for paper "**Improve ROI with Causal Learning and Conformal Prediction**".

## **Reproduction Instructions**

#### ***Section V. Experiments --> A. Offline Simulation***





1. To support the use of Area under Uplift Curve (AUUC), install causalML from https://causalml.readthedocs.io/en/latest/installation.html.

2. To support the use of Generalized Random Forests (GRF) , install econML from https://github.com/microsoft/EconML.

3. Tensorflow 2.14.0 is used in this experiment.

### ***Directory Structure***
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
5. Download the dataset named CRITEO-UPLIFT v2 from https://ailab.criteo.com/criteo-uplift-prediction-dataset/. You can put this dataset in the data directory, and rename it as "criteo-uplift-v2.1.csv". By this way, you can run the demo in the code directory based on this dataset.


