# Punishment and Deterrence: Replication of Drunk Driving Study

## Description
This repository contains data and Python code developed to replicate the findings of Benjamin Hansen's 2015 study, "Punishment and Deterrence: Evidence from Drunk Driving," published in the American Economic Review. The original study examines the effects of stricter punishments on reducing recidivism among drunk drivers, using regression discontinuity design. The analysis is based on a rich dataset of over 500,000 DUI stops in Washington State, exploring whether increased sanctions at blood alcohol content (BAC) thresholds effectively reduce repeat drunk driving offenses.

### Repostory Contents

- replication.ipynb : Jupyter notebook with Python code to recreate the paper's results.
- rdd_ml.py : Python script using Cross Validation to optimize bandwidth.
- hansen_dwi.csv : CSV file with data used for the replication.
- RDD_Presentation.pdf : Poster presentation summarizing this project.

### Data Description
The dataset includes variables such as BAC levels, recidivism indicators, and demographic details of offenders, originally sourced from administrative records of DUI stops. Please refer to the data dictionary for detailed descriptions of all variables.

### Methodology
The original study utilized a regression discontinuity design to estimate the impact of BAC thresholds on the probability of recidivism. This repository attempts to recreate that analysis using Python, and add to the empirical analysis by using machine learning methods, providing scripts for both the statistical tests and the generation of relevant visualizations.

### Citation
Hansen, B. (2015). Punishment and Deterrence: Evidence from Drunk Driving. American Economic Review, 105(4), 1581–1617. DOI:10.1257/aer.20130189

### Contributors

- [Jose Hernandez-Balsamo](https://www.linkedin.com/in/jose-hernandezb-24d05)
- [Patrick Tierney](https://www.linkedin.com/in/patrick-tierney-4bb579265/)
- Canyon Marshall
- Luke Mynatt
