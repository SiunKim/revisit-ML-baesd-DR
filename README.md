# revisit-ML-baesd-DR
This repository is for "Revisiting Machine-Learning based Drug Repositioning: Drug Indication is not a Right Prediction Target" 

# revisit exisiting DR models
Datasets and codes for re-visiting existing DR models in section 3.5 are included in the folder 'revisit_existing_DR_models.' The original datasets needed to implement existing DR models are in the folders, 'OriginalDatasets', in each model folder. Python codes for setting train/test datasts based on original/expanded indications are 'AddFunctions_modelname.py'. The original/expanded indications split with a given clustering hyperparmeter setting are provided in 'DiDrMat___.tsv' and 'drugDisease___.tsv' files. The clustering hyperparmeter setting was denoted in a filename, such as 'drugDisease_expanded_ICD10_cluster2_complete_th0.9.tsv' which means that agglomerative clustering was performed based on ICD10, using two cluster and complete linkage criterion, with similarity threshold of 0.9.



