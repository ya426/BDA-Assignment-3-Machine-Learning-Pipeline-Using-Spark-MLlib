# BDA Assignment 3 — Healthcare Disease Prediction Using Apache Spark MLlib

Use Python 3.10 or 3.11 and JDK 17. In PowerShell in this folder:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_dataset.py
.venv\Scripts\python.exe diabetes_analysis.py --ui
```

Dataset: https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv (public Pima Indians Diabetes mirror, 768 records). The CSV is fetched automatically if absent. This package does not include the CSV because network download in the preparation environment failed. Alternatively download it in a browser and save as data/diabetes.csv.

Pipeline: parse numeric fields; treat nonphysical zeros and negatives as missing (zero pregnancies is valid); remove invalid labels; split approximately 80/20 with seed 42; fit median imputer, VectorAssembler, StandardScaler and classifier on training data only; transform test and new records. Compare positive-class accuracy, precision, recall and F1 using confusion counts. Select highest test F1, break ties by accuracy. Test-set selection is for assignment comparison only; an independent validation set is necessary for unbiased estimates. Predictions are for fictional examples and not for clinical use.

Spark partitions DataFrames, schedules stages and tasks and distributes model computation across executors in cluster mode. Local mode uses two threads, not a multi-machine cluster. MLlib estimators learn parameters from training data; transformers apply fitted transformations. Spark SQL optimizes DataFrame operations and evaluates transformations when actions require them. Distributed algorithms aggregate statistics across partitions and may shuffle data.

Outputs are created in output/ and automatically zipped as BDA-Assignment-3-Actual-Output.zip after successful completion. Capture actual screenshots from the printed Spark UI URL while --ui is active. No actual Spark results are bundled because PySpark was unavailable during preparation.
