
## NOTE: WE ARE PREPARING AND CLEANING THE CODE. THE CODE WILL BE READY BY 2023/12/09. THANKS.


# Improve ROI with Causal Learning and Conformal Prediction

The code to replicate the offline results for paper "**Improve ROI with Causal Learning and Conformal Prediction**".

## **Reproduction Instructions**

#### ***Section V. Experiments --> A. Offline Simulation***

Steps to reproduce the results:
```
1. Add your work directory to "homePath" in generateSimulationData.R under Code/Data_generation folder.
2. Run generateSimulationData.R.
3. Follow the instructions under each model's folder for training and prediction; run budget_allocation.py for each model.
4. Run Simulation_analysis.py under Code/Evaluation folder.



1. To support the use of Area under Uplift Curve (AUUC), install causalML from https://causalml.readthedocs.io/en/latest/installation.html.

2. To support the use of Generalized Random Forests (GRF) , install econML from https://github.com/microsoft/EconML.

3. Tensorflow 2.14.0 is used in this experiment.

4. The directory structure is shown as follows.

|-----code
|     |-----【InCo】xxx.ipynb       # Various benchmark methods when the setting is Insufficient data and Covariant shift.
|     |-----【InNo】xxx.ipynb       # Various benchmark methods when the setting is Insufficient data and No covariant shift.
|     |-----【SuCo】xxx.ipynb       # Various benchmark methods when the setting is Sufficient data and Covariant shift.
|     |-----【SuNo】xxx.ipynb       # Various benchmark methods when the setting is Sufficient data and No covariant shift.
|     |-----【SuNo】xxx.ipynb       # Various benchmark methods when the setting is Sufficient data and No covariant shift.
|     |-----【Offline Evaluation】xxx.ipynb       # Notebook codes that ultimately generate Fig.5 and Table 1.
|     |----- Fig.1【a】and Fig.1【b】.ipynb       # Notebook codes that generate Fig.1a and Fig.1b.
|-----metric
|     |-----Metric.py                                  # The evaluation metrics
|-----model
|     |-----uplift_model.py                            # The model to predict CATE
|     |-----roi_model.py                               # The model to predict ROI
|     |-----mt_roi_model.py                           
|-----README.txt

5. Download the dataset named CRITEO-UPLIFT v2 from https://ailab.criteo.com/criteo-uplift-prediction-dataset/. You can put this dataset in the data directory, and rename it as "criteo-uplift-v2.1.csv". By this way, you can run the demo in the code directory based on this dataset.

6. Due to data privacy, the real business marketing data used in this experiment is not provided. There may be multiple notebook files in the "code/marketing_data_code" directory, which represent the training demos of different models.
