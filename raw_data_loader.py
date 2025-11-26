from typing import List, Tuple

import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from utils import normalize, compute_relevance, RelevanceMetric, compute_correlation, CorrelationMetric


class SingleOmicsDataset:
    r"""The single omics dataset

    Args:
        path (str): The directory of the dataset.
        name (str): The name of the omics data.
        label (pd.DataFrame): A dataframe of the labels of samples.
        init_pheromone_val (float): Initial pheromone value.
        prob_val (float): The relative importance of the omics.
        sparsity_rate (float, optional): The sparsity rate in feature similarity network. (default: 0.8)
    """
    def __init__(
            self,
            path: str,
            name: str,
            label: pd.DataFrame,
            prob_val: float,
            sparsity_rate: float = 0.8
    ) -> None:
        self.path = path
        self.name = name
        self._label = label
        self.default_prob_val = prob_val
        self.prob_val = self.default_prob_val
        self.sparsity_rate = sparsity_rate
        self._data = pd.read_csv(path, sep=',', index_col=0)
        self._train_data = None
        self._train_label = None
        self._test_data = None
        self._test_label = None

        self._data.index.names = ['sample']
        self._data.columns.names = ['feature']

        self._relevance = None
        self._corr = None
        self._sparse_adj_mat = None
        self._selected_subset = None
        self._combined_data = None

    def build_full_data(self, common_indices, label):
        self._data = self._data[self._data.index.isin(common_indices)]
        self._data = self._data.sort_index(axis=0)
        self._label = label

    def clean_missing_values(self):
        self._data = self._data.dropna(axis=1)

    def normalize_data(self):
        self._data = normalize(self._data)

    def remove_low_variance_features(self, threshold):
        self._data = self._data.loc[:, self._data.var() >= threshold]

    def config_components(self):
        self._relevance = compute_relevance(self._train_data, self._train_label, RelevanceMetric.ANOVA)
        self._relevance = normalize(self._relevance, 0, 1)
        self._corr = compute_correlation(self._train_data, CorrelationMetric.PEARSON_CORRELATION)
        similarity_threshold = self._find_similarity_threshold()
        self._sparse_adj_mat = sp.csr_matrix(self._corr <= similarity_threshold)

    def _find_similarity_threshold(self):
        sorted_correlation = self._corr.unstack().sort_values(ascending=False)
        threshold_index = int(self._corr.size * self.sparsity_rate)
        threshold_value = sorted_correlation.iloc[threshold_index]

        return threshold_value

    def reduce_dimensionality(self, subset: List):  # 参数名修正
        self._selected_subset = subset
        self._train_data = self._train_data.iloc[:, subset]
        self._test_data = self._test_data.iloc[:, subset]

        self._update_combined_data()
    
    def _update_combined_data(self):
        """更新合并后的数据集"""
        if self._train_data is not None and self._test_data is not None:
            self._combined_data = pd.concat(
                [self._train_data, self._test_data], 
                axis=0
            )
        else:
            self._combined_data = self._data.copy()
        self._combined_data = torch.tensor(self._combined_data.values, dtype=torch.float32)
    
    @property
    def combined_data(self):
        """获取合并后的完整数据集"""
        return self._combined_data

    @property
    def data(self):
        return self._data

    @property
    def train_data(self):
        return self._train_data

    @property
    def test_data(self):
        return self._test_data

    @property
    def train_label(self):
        return self._train_label

    @property
    def test_label(self):
        return self._test_label

    def set_train_test(self, train_index, test_index):
        self._train_data = self._data.iloc[train_index]
        self._test_data = self._data.iloc[test_index]
        self._train_label = self._label.iloc[train_index]
        self._test_label = self._label.iloc[test_index]
        self._update_combined_data()

    @property
    def num_features(self) -> int:
        if self._data is not None:
            return len(self._data.columns)
        else:
            return len(self._train_data.columns)

    @property
    def num_samples(self) -> int:
        if self._data is not None:
            return len(self._data.index)
        else:
            return len(self._train_data.index) + len(self._test_data.index)

    @property
    def num_train_samples(self) -> int:
        return len(self._train_data.index)

    @property
    def num_test_samples(self) -> int:
        return len(self._test_data.index)

    def get_relevance(self, feat_idx=None):
        return self._relevance.to_numpy() if feat_idx is None else self._relevance.iloc[feat_idx]

    def get_correlation(self, feat_idx_1, feat_idx_2):
        return max(self._corr[feat_idx_1, feat_idx_2], self._corr[feat_idx_2, feat_idx_1])

    def is_connected(self, feat_idx_1, feat_idx_2):
        return self._sparse_adj_mat[feat_idx_1, feat_idx_2]

    def get_connected_feats(self, feat_idx):
        row = self._sparse_adj_mat.getrow(feat_idx)
        nonzero_indices = row.nonzero()[1]
        return nonzero_indices


    def check_index(self, df_label):
        indices_equal = np.array_equal(self.data.index.values, df_label.index.values)

        if not indices_equal:
            raise Exception("The omics data indices are not equal to label indices.")

    def get_data_structure(self):
        # 仅返回有效属性
        return self._sparse_adj_mat, self._corr, self._relevance,  self._selected_subset


class MultiOmicsDataset:
    r"""The multiomics dataset

    Args:
        dataset_name (str): The name of multiomics dataset.
        raw_file_paths (List[Tuple]): The path of the omics data files.
        raw_label_path (str): The path of the label data.
        num_omics (int): The total number of omics in the dataset.
        num_classes (int): The total number of classes in the dataset.
        init_pheromone_val (float): Initial pheromone value for MAS.
        sparsity_rates (List[float]): The sparsity rate for each omics in feature similarity network.
    """
    def __init__(
            self,
            dataset_name: str,
            raw_file_paths: List[Tuple],
            raw_label_path: str,
            num_omics: int,
            num_classes: int,
            sparsity_rates: List[float],
    ) -> None:
        self._dataset_name = dataset_name
        self._raw_file_paths = raw_file_paths
        self._raw_label_path = raw_label_path
        self._num_omics = num_omics
        self._num_classes = num_classes
        self.sparsity_rates = sparsity_rates
        self.feat_size = None
        self.data = []
        self.label = None

        self._process()

    def _process(self):
        prob_val = 1.0 / self.num_omics

        # generate binary class label
        self.label = pd.read_csv(self._raw_label_path, sep=',', index_col=0)

        # create single omics datasets
        for idx, (path, name) in enumerate(self._raw_file_paths):
            omics_data = SingleOmicsDataset(path, name, self.label, prob_val,
                                            self.sparsity_rates[idx])
            self.data.append(omics_data)

        # create a full dataset
        common_indices = self._find_common_samples()
        self.label = self.label[self.label.index.isin(common_indices)]
        self.label = self.label.sort_index(axis=0)

        for omics_idx in range(self.num_omics):
            self.data[omics_idx].build_full_data(common_indices, self.label)

        # check whether omics data indices are equal to label indices
        for omics_idx in range(self.num_omics):
            self.data[omics_idx].check_index(self.label)

    def _find_common_samples(self):
        common_indices = self.label.index
        for omics_idx in range(self.num_omics):
            common_indices = common_indices.intersection(self.data[omics_idx].data.index)
        return common_indices

    def get(self, omics_idx) -> SingleOmicsDataset:
        return self.data[omics_idx]

    @property
    def num_omics(self) -> int:
        return self._num_omics

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def set_train_test(self, train_index, test_index):
        for omics_idx in range(self.num_omics):
            self.data[omics_idx].set_train_test(train_index, test_index)

    def config_components(self):
        for omics in self.data:
            omics.config_components()


    def get_node_relevance(self, omics_idx=None):
        if omics_idx is not None:
            return self.data[omics_idx].get_relevance()

        return [single_omics.get_relevance() for single_omics in self.data]


    def reduce_dimensionality(self, subsets):
        """subsets结构: [omics1_indices, omics2_indices,...]"""
        for omics_idx in range(self.num_omics):
            self.data[omics_idx].reduce_dimensionality(subsets[omics_idx])

    def concatenate_data(self, subset=None, is_train=True):
        concat_data = []
        for omics_idx in range(self.num_omics):
            if is_train:
                final_data = self.data[omics_idx].train_data
            else:
                final_data = self.data[omics_idx].test_data

            if subset:
                omics_subset = subset.get(omics_idx, [])
                final_data = final_data.iloc[:, omics_subset]

            concat_data.append(final_data)

        concat_data = pd.concat(concat_data, axis=1)

        return concat_data

    def __repr__(self) -> str:
        multiomics_str = [
            "\nDataset info:",
            f"\n   dataset name: {self._dataset_name}",
            f"\n   number of omics: {self.num_omics}",
            f"\n   number of classes: {self.num_classes}",
            "\n\n   omics    | num samples | num features",
            f"\n   {'-' * 40}",
        ]
        for omics in range(self.num_omics):
            omics_data = self.get(omics)
            multiomics_str.append(
                f"\n   {omics_data.name:<8} | "
                f"{omics_data.num_samples:<11} | "
                f"{omics_data.num_features:<12}"
            )
        else:
            multiomics_str.append(f"\n   {'-' * 40}\n\n")

        flag = False
        for omics in range(self.num_omics):
            omics_data = self.get(omics)
            if omics_data.train_data is not None:
                flag = True
                multiomics_str.append(
                    f"\n   {omics_data.name:<8} | "
                    f"{omics_data.num_train_samples:<11} | "
                    f"{omics_data.train_data.shape[1]:<12}"
                )
        else:
            if flag:
                multiomics_str.append(f"\n   {'-' * 40}\n\n")

        flag = False
        for omics in range(self.num_omics):
            omics_data = self.get(omics)
            if omics_data.test_data is not None:
                flag = True
                multiomics_str.append(
                    f"\n   {omics_data.name:<8} | "
                    f"{omics_data.num_test_samples:<11} | "
                    f"{omics_data.test_data.shape[1]:<12}"
                )
        else:
            if flag:
                multiomics_str.append(f"\n   {'-' * 40}\n\n")

        return "".join(multiomics_str)
