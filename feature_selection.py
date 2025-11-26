import numpy as np
import itertools
from typing import List
import copent 
from sklearn.preprocessing import MinMaxScaler

def calc_f3(x,y):
    labels=np.unique(y)
    indexs={}
    c_mins={}
    c_maxs={}
    for label in labels:
        index=np.where(y==label)[0]
        indexs[label]=index
        c_min=np.min(x[index])
        c_max=np.max(x[index])
        c_mins[label]=c_min
        c_maxs[label]=c_max
    label_combin=list(itertools.combinations(labels,2))
    f3=0.0
    for combination in label_combin:
        # sample_num=len(indexs[combination[0]])+len(indexs[combination[1]])
        # print(sample_num)
        # print(combination)
        # print(sample_num)
        c1_max,c1_min=c_maxs[combination[0]],c_mins[combination[0]]
        c2_max,c2_min=c_maxs[combination[1]],c_mins[combination[1]]
        # print(c1_max,c1_min,c2_max,c2_min)
        if c1_max<c2_min or c2_max<c1_min:
            f3+=1
        else:
            interval=(max(c1_min,c2_min),min(c1_max,c2_max))
            sample=np.hstack((x[indexs[combination[0]]],x[indexs[combination[1]]]))
            # print(sample.shape[0])
            n_overlay=0
            for k in range(sample.shape[0]):
                if sample[k]>=interval[0] and sample[k]<=interval[1]:
                    n_overlay+=1
            f3+=1-n_overlay/sample.shape[0]
    f3/=len(label_combin)
    return f3

def calc_copent_feature_pair(feat_i: np.ndarray, feat_j: np.ndarray) -> float:
    data = np.column_stack((feat_i, feat_j))
    return copent.copent(data)

def select_feature_combined(
    x: np.ndarray, 
    y: np.ndarray, 
    k: int, 
    redundancy_threshold: float = 0,
    alpha: float = 0.7
) -> List[int]:
    n_features = x.shape[1]
    f3_scores = []
    for i in range(n_features):
        if len(np.unique(x[:, i])) <= 1:
            f3_scores.append(0.0)
        else:
            f3_scores.append(calc_f3(x[:, i], y))
    candidate_indices = np.argsort(f3_scores)[-2*k:]

    selected_original = []

    ce_targets = []
    for idx in candidate_indices:
        feat = x[:, idx]
        data_target = np.column_stack((feat, y))
        ce = copent.copent(data_target)
        ce_targets.append(ce)

    f3_scaled = MinMaxScaler().fit_transform(np.array(f3_scores)[candidate_indices].reshape(-1, 1)).flatten()
    ce_scaled = 1-MinMaxScaler().fit_transform(np.array(ce_targets).reshape(-1, 1)).flatten()
    combined_scores = alpha * f3_scaled + (1 - alpha) * ce_scaled

    sorted_order = np.argsort(-combined_scores)
    
    for order in sorted_order[:k]:
        current_idx = candidate_indices[order] 
        selected_original.append(current_idx)
                
    return selected_original[:k]
