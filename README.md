This repository contains datasets and codes used in a research paper titled "Revisiting Machine-Learning based Drug Repositioning: Drug Indication is not a Right Prediction Target," which has been submitted to track 2 of CHIL 2023.

# Folder Descriptions
1. "tSNE_visualization" folder includes datasets and codes for tSNE analysis and plotting in section 4.2 of the paper.

2. "atc_prediction" folder includes datasets and codes for developing an XGBoost ATC prediction model in section 4.3 of the paper.

3. "split_indications" folder includes datasets and codes for splitting drug indications into original and expanded, as described in section 5.2 and appendix A.3 of the paper.

4. "revisit_existing_DR_models" folder includes datasets and codes for revisiting existing drug repositioning models, as explained in section 5.3 of the paper.

    - To implement existing DR models, access the original datasets in the "OriginalDatasets" folder located in each model folder. 
 
    - Use "AddFunctions_modelname.py" Python codes to set train/test datasets based on original/expanded indications. 

    - The "DiDrMat___.tsv" and "drugDisease___.tsv" files provide the split of original/expanded indications, with a clustering hyperparameter setting denoted in the filename (e.g., "drugDisease_expanded_ICD10_cluster2_complete_th0.9.tsv" means agglomerative clustering was performed based on ICD10, using two clusters and complete linkage criterion, with a similarity threshold of 0.9).
