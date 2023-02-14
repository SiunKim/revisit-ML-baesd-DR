# revisit-ML-baesd-DR
This repository is for "Revisiting Machine-Learning based Drug Repositioning: Drug Indication is not a Right Prediction Target" 


# revisit exisiting DR models
The folder 'revisit_existing_DR_models' contains datasets and codes for revisiting existing DR models in section 3.5.

To implement existing DR models, access the original datasets in the 'OriginalDatasets' folders located in each model folder.

Use 'AddFunctions_modelname.py' Python codes to set train/test datasets based on original/expanded indications.

The 'DiDrMat___.tsv' and 'drugDisease___.tsv' files provide the original/expanded indications split with a clustering hyperparameter setting denoted in the filename, such as 'drugDisease_expanded_ICD10_cluster2_complete_th0.9.tsv'. This file means agglomerative clustering was performed based on ICD10, using two clusters and complete linkage criterion, with a similarity threshold of 0.9.
