import pickle
import ast

from collections import Counter
from itertools import combinations 

import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering



def get_common_start_string(string1: str, string2: str) -> str:
    '''
    find longest common string from start and return it as str
    '''
    common_start_string = ''
    
    for i in range(len(string1)+1):
        if string2.startswith(string1[0:i]):
            common_start_string = string1[0:i]
            
    return common_start_string


def find_least_common_ancestor(code1: str, code2: str) -> str:
    '''
    find least common ancestor in ICD10 and MeSH ontology
    '''
    common_code_string = get_common_start_string(code1, code2)
    
    if ONTOLOGY=='ICD10':
        if len(common_code_string)>=3:
            least_common_ancestor = (common_code_string.replace('.', '') 
                                        if common_code_string.endswith('.') 
                                        else common_code_string)
            
        else:
            level2_of_code1 = LEVEL3_TO_LEVEL2[code1[0:3]]
            level2_of_code2 = LEVEL3_TO_LEVEL2[code2[0:3]]
            
            if level2_of_code1==level2_of_code2:
                least_common_ancestor = level2_of_code1
            else:
                level1_of_code1 = LEVEL2_TO_LEVEL1[level2_of_code1]
                level1_of_code2 = LEVEL2_TO_LEVEL1[level2_of_code2]
                
                if level1_of_code1==level1_of_code2:
                    least_common_ancestor = level1_of_code1
                else:
                    least_common_ancestor = 'root'     
                            
    else: #ONTOLOGY=='MeSH'
        least_common_ancestor = ''
        for c1, c2 in zip(code1.split('.'), code2.split('.')):
            if c1==c2:
                least_common_ancestor += c1 + '.'
            else:
                if least_common_ancestor.endswith('.'):
                    least_common_ancestor = least_common_ancestor[:-1]
                break
        
        if not least_common_ancestor:
            if code1[0]==code2[0]:
                least_common_ancestor = code1[0]
            else:
                least_common_ancestor = 'root'
            
    return least_common_ancestor


def calculate_information_content(code: str) -> float:
    '''
    calculate information content of individual medical concept code
    * reference: Jia et al., 2020, equation (4)
    '''
    information_content = - np.log(((count_leaves(code)/count_subsumers(code))+1)/
                                   (count_leaves('root')+1))
    
    return information_content


def calculate_code_level_similarity(code1, code2) -> float:
    '''
    calculate the code level similarity between two medical concept codes based on information content
     - 0 for identical and very close, 1 for significanly different
    * reference: Jia et al., 2020, equation (5)
    '''
    lca = find_least_common_ancestor(code1, code2)
    ic1 = calculate_information_content(code1)
    ic2 = calculate_information_content(code2)
    ic_lca = calculate_information_content(lca)
    
    code_level_similarity = 1 - 2*ic_lca/(ic1+ic2)
    
    return code_level_similarity


def get_sum_of_min_cls_by_codes(codes_for_sum: list,
                                codes_for_min: list) -> float:
    '''
    * reference: Jia et al., 2020, equation (6)
    '''
    sum_of_min_cls_by_codes = 0
    
    for code1 in codes_for_sum:
        min_cls_by_code1 = min([calculate_code_level_similarity(code1, code2) 
                                for code2 in codes_for_min])
        sum_of_min_cls_by_codes += min_cls_by_code1
        
    return sum_of_min_cls_by_codes       
    

def calculate_set_level_similarity(codes1: list, 
                                   codes2: list) -> float:
    '''
    calculate the set level similarity between two medical concept code lists
    * reference: Jia et al., 2020, equation (6)
    '''
    sum_of_min_cls_by_code1 = get_sum_of_min_cls_by_codes(codes1, codes2)
    sum_of_min_cls_by_code2 = get_sum_of_min_cls_by_codes(codes2, codes1)
    
    set_level_similarity = ((sum_of_min_cls_by_code1 + sum_of_min_cls_by_code2)/
                            (len(codes1)+len(codes2)))
    
    return set_level_similarity


def count_leaves(code: str) -> int:
    '''
    get leaves count of a given medical concept code
    '''
    if ONTOLOGY=='ICD10':
        leaves_count = ICD10_LVS_SUBS_COUNTS.get(code, {'leave_count': 1})['leave_count']
    else: #ONTOLOGY=='MeSH'
        leaves_count = MESH_LVS_SUBS_COUNTS.get(code, {'leave_count': 1})['leave_count']
    
    return leaves_count
    

def count_subsumers(code: str) -> int:
    '''
    get subsumer count of a given medical concept code (include itself, +1)
    '''
    if ONTOLOGY=='ICD10':
        subs_count = ICD10_LVS_SUBS_COUNTS.get(code, {'subsumer_count': 4})['subsumer_count']
    else: #ONTOLOGY=='MeSH'
        subs_count = ICD10_LVS_SUBS_COUNTS.get(code, {'subsumer_count': len(code.split('.'))+2})['subsumer_count']
    
    return subs_count


def without_dot_codes(ICD10_codes_of_a_drug):
    '''
    remove '.' in ICD10_code
    '''
    ICD10_codes_of_a_drug_rev = [[ICD10_code.replace('.', '') 
                                    for ICD10_code in ICD10_codes] 
                                    for ICD10_codes in ICD10_codes_of_a_drug]
            
    return ICD10_codes_of_a_drug_rev
    
    
def calculate_sls_between_indications(disease_codes_of_a_drug: list) -> dict:
    '''
    calculate set level similarity between indications of two drugs and return it as dictionary format
    
    Arg:
        disease_codes_of_a_drug (list of list of str): ICD10/MeSH codes for indications a single drug 
        
    Output:
        sls_between_inds (dictionary of float): 
            - key: tuple expressing an indication index pair
            - value: set level similarity for a given indication pair
         ex. {'(0, 1)': 1.0, '(0, 2)': 1.0, '(1, 1)': 0.0}
           * again, 0 for identical and very close, 1 for significanly different
    '''
    indications = list(range(len(disease_codes_of_a_drug)))
    indi_combis = list(combinations(indications, 2))

    sls_between_inds = {}
    for indi_combi in indi_combis:
        codes1 = disease_codes_of_a_drug[indi_combi[0]]
        codes2 = disease_codes_of_a_drug[indi_combi[1]]
        set_level_similarity = calculate_set_level_similarity(codes1, codes2)

        sls_between_inds[str(indi_combi)] = set_level_similarity
        
    return sls_between_inds
    

def get_similarity_matrix_from_sls(sls_between_inds: dict) -> np.ndarray:
    '''
    return sls_between_inds as 2-d np.ndarray
    '''
    indications_number = max(ast.literal_eval(k) for k in sls_between_inds.keys())[1] + 1
    
    similarity_matrix = np.zeros((indications_number,indications_number))
    
    for indi_combi, sls_between_inds in sls_between_inds.items():
        indi_index1, indi_index2 = ast.literal_eval(indi_combi)
        similarity_matrix[indi_index1, indi_index2] = sls_between_inds
        similarity_matrix[indi_index2, indi_index1] = sls_between_inds
        
    return similarity_matrix
        

def get_similarity_matrix_from_codes(dataset_of_a_drug: pd.DataFrame) -> np.ndarray:
    '''
    get similarity matrix for drug indications of a drug
    '''    
    if ONTOLOGY=='ICD10':
        disease_codes_of_a_drug = without_dot_codes(list(dataset_of_a_drug['merged_ICD10_codes']))
    else: #ONTOLOGY=='MeSH'
        disease_codes_of_a_drug = list(dataset_of_a_drug['merged_MeSH_treenumbers'])
        
    sls_between_inds = calculate_sls_between_indications(disease_codes_of_a_drug)
    similarity_matrix = get_similarity_matrix_from_sls(sls_between_inds)
    
    return similarity_matrix


def calculate_linkage_distance(index_combis: list,
                               similarity_matrix: np.ndarray,
                               linkage: str) -> float:
    '''
    return maximum or average similarity among indication pairs of a given index_combis in similarity_matrix환
    '''
    if linkage=='complete':
        linkage_distance = np.max([similarity_matrix[index_combi] 
                                   for index_combi in index_combis])
    else: #linkage=='average'
        linkage_distance = np.average([similarity_matrix[index_combi] 
                                    for index_combi in index_combis])
        
    return linkage_distance


def calculate_within_cluster_similarity(labels: np.ndarray,
                              similarity_matrix: np.ndarray,
                              linkage: str) -> list:
    '''
    calculate within cluster similarity between indications
    
    Args:
        labels (np.ndarray): cluster labels from agglomerative clustering analysis
        similarity_matrix (np.ndarray): 2d similarity matrix used for indication clustering 
        linkage (str): linkage criterion
    
    Output:
        within_cluster_similarities (list of int): similarity within cluster 
         * if the number of indiction is 1 in a cluster, return 1
    '''
    assert linkage in ['complete', 'average'], f'Linkage({linkage}) must be "complete" or "average"!'
    
    within_cluster_similarities = []
    
    for cluster_class in range(max(labels)+1):
        indexes = [i for i, e in enumerate(labels) if e==cluster_class]
        idx_combis = list(combinations(indexes, 2))
        
        if idx_combis:
            if linkage=='complete':
                within_cluster_similarity = max([similarity_matrix[idx_combi] 
                                                    for idx_combi in idx_combis])
            else: #linkage=='average'
                within_cluster_similarity = np.average([similarity_matrix[idx_combi]
                                                          for idx_combi in idx_combis])
        
        else:
            within_cluster_similarity = 1
            
        within_cluster_similarities.append(within_cluster_similarity)
        
    return within_cluster_similarities


def classify_ori_expd_using_clustering(labels: np.ndarray,
                                       similarity_matrix: np.ndarray,
                                       linkage: str,
                                       threshold: 0=float) -> list:
    '''
    split indications into original and expanded based on agglomerative clustering analysis
    
    Args:
        labels (np.ndarray): cluster labels from agglomerative clustering analysis
        similarity_matrix (np.ndarray): 2d similarity matrix used for indication clustering 
        linkage (str): linkage criterion
    
    Output:
        original_or_expanded (list): list expressing whether an indication is original or expanded
    '''
    assert linkage in ['complete', 'average'], f'Linkage({linkage}) must be "complete" or "average"!'
    
    label_counter = Counter(labels)
    least_common_labels = [label for label, count in label_counter.items() 
                            if count==min(label_counter.values())]
    most_common_labels = [label for label, count in label_counter.items() 
                            if count==max(label_counter.values())]
    
    original_or_expanded = []
    
    #find one least_common_label
    if len(least_common_labels)==1:
        lcl = least_common_labels[0]
    #find two least_common_labels or more than two
    else:
        #find one most_common_label
        if len(most_common_labels)==1:
            mcl = most_common_labels[0]
        #find two most_common_label or more than two
        else:
            wcss = calculate_within_cluster_similarity(labels,
                                                      similarity_matrix,
                                                      linkage)
            #mcl is clsuter of which within-cluster similarity is the lowest
            mcl = [label for label, wcs in enumerate(wcss) 
                    if (label in most_common_labels) and (wcs==min(wcss))][0]
         
        #most dissimilar cluster with mcl as 'expanded' label      
        indexes_original = np.where(labels==mcl)[0]
        
        distances_by_label = {}
        for lcl in least_common_labels:
            indexes_lcl = np.where(labels==lcl)[0]
            #calculate distance from original cluster (mcl)
            indexes_combis = [(index_lcl, index_ori) for index_ori in indexes_original
                                for index_lcl in indexes_lcl]
            distance_by_label = calculate_linkage_distance(indexes_combis,
                                                           similarity_matrix,
                                                           linkage) 
            distances_by_label[lcl] = distance_by_label
        
        #find lcl for expanded indication
        lcl = [k for k, v in distances_by_label.items() 
                if v==max(distances_by_label.values())][0]
        
    #if distance between mcl and lcl, classified lcl as expanded
    if threshold:
        lcl_indexes = [i for i, l in enumerate(labels) if l==lcl]
        distances = [list(similarity_matrix[li][0:li]) + list(similarity_matrix[li][li+1:]) 
                     for li in lcl_indexes]
        distance_average = np.average([d for ds in distances for d in ds])
        
        if distance_average >= threshold:
            pass
        else:
            lcl = -1
            
    original_or_expanded = ['original' if label!=lcl else 'expanded' for label in labels]
        
    return original_or_expanded


def split_indications_based_on_IC(dataset: pd.DataFrame,
                                  n_clusters: int=2,
                                  linkage: str='average',
                                  threshold: float=0) -> pd.DataFrame:
    '''
    split indications into original and expanded based on agglomerative clustering and set level similarity, return the results as new column as dataset DataFrame
    
    Args:
        dataset (pd.DataFrame): DDA datasets for indication split
        n_clusters (int): number of clusters in AgglomerativeClustering
        linkage (str): linkage criterion in AgglomerativeClustering ('complete' or 'average')
        threshold (float): threshold for expanded indication split (min 0, max 1)
        
    Output:
        dataset_split (pd.DataFrame): DDA datasets with a new column expressing whether an indication is original or expanded
    '''
    assert n_clusters in [2,3], f'n_cluster ({n_clusters}) must be 2 or 3!'
    
    dataset_by_drug = dataset.groupby('db_id')
    db_id_keys = list(dataset_by_drug.groups.keys())
    
    original_or_expanded_total = []
    for db_id in db_id_keys:
        dataset_of_a_drug = dataset_by_drug.get_group(db_id)
            
        if len(dataset_of_a_drug)<(n_clusters + 1):
            original_or_expanded = ['original'] * len(dataset_of_a_drug)
        else: 
            similarity_matrix = get_similarity_matrix_from_codes(dataset_of_a_drug)
            #AgglomerativeClustering
            clustering = AgglomerativeClustering(affinity='precomputed', 
                                                n_clusters=n_clusters, 
                                                linkage=linkage).fit(similarity_matrix)
            
            #classify original_or_expanded
            labels = clustering.labels_
            original_or_expanded = classify_ori_expd_using_clustering(labels,
                                                                      similarity_matrix,
                                                                      linkage,
                                                                      threshold)
            
        assert len(original_or_expanded)==len(dataset_of_a_drug)
        original_or_expanded_total += original_or_expanded

    print(f"original: {original_or_expanded_total.count('original')}, expanded: {original_or_expanded_total.count('expanded')}")
    
    #add a new column: 'original_expanded_ic'
    dataset_split = pd.DataFrame({})
    for db_id in db_id_keys:
        dataset_of_a_drug = dataset_by_drug.get_group(db_id)
        dataset_split = pd.concat([dataset_split, dataset_of_a_drug])
        
    dataset_split['original_expanded_ic'] = original_or_expanded_total
    
    return dataset_split



#main code
#import LEVEL2_TO_LEVEL1 and LEVEL3_TO_LEVEL2 for ICD10 ontology
with open('preprocess_ICD10_codes\LEVEL2_TO_LEVEL1_dict.p', 'rb') as f:
    LEVEL2_TO_LEVEL1 = pickle.load(f)    
with open('preprocess_ICD10_codes\LEVEL3_TO_LEVEL2_dict.p', 'rb') as f:
    LEVEL3_TO_LEVEL2 = pickle.load(f)
#import ICD10_LVS_SUBS_COUNTS dictionary
with open('ICD10_leaves/ICD10_leaves_subsumers_dict.p', 'rb') as f:
    ICD10_LVS_SUBS_COUNTS = pickle.load(f)
#import MESH_LVS_SUBS_COUNTS dictionary
with open('MeSH_leaves/MeSH_leaves_subsumers_dict_230109.p', 'rb') as f:
    MESH_LVS_SUBS_COUNTS = pickle.load(f)

#import the Fdataset and the deepDR datasets
with open('MESH_leaves/Fdataset_with_ICD10_MeSH_tN_230109_DF.p', 'rb') as f:
    Fdataset = pickle.load(f)
with open('MESH_leaves/deepDR_with_ICD10_MeSH_tN_230109_DF.p', 'rb') as f:
    deepDR = pickle.load(f)

#set global variable (medical ontology and threshold for original/expanded indication split)
ONTOLOGIES = ['ICD10', 'MeSH']
THRESHOLD = {'ICD10': 0.9, 'MeSH': 0.7}

#split indications into original/expanded
from datetime import datetime
today = datetime.today().strftime('%y%m%d')
#split indications based on various clustering hyperparameter settings
for dataset, dname in zip([deepDR, Fdataset], ['deepDR', 'Fdataset']):
    for ONTOLOGY in ONTOLOGIES:
            for linkage in ['average', 'complete']:
                for n_cluster in [2, 3]:
                    dataset_split = split_indications_based_on_IC(dataset,
                                                                n_clusters=n_cluster,
                                                                linkage=linkage,
                                                                threshold=THRESHOLD[ONTOLOGY])
                    
                    # #save split datasets as csv
                    # csv_name = f'{dname}_{ONTOLOGY}_cluster{n_cluster}_{linkage}_th{THRESHOLD[ONTOLOGY]}_{today}.tsv'
                    # dataset_split.to_csv(csv_name, sep='\t', index=False)