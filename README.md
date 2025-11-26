
The source code, developed in Python 3.10, has been verified on both Linux and Windows and should be compatible with any operating system that supports Python. It is functional on standard computers and has been evaluated for both CPU and GPU usage. For optimal performance, it is recommended to train the GAT-Mamba-Trans module on a GPU for faster processing.

DBCL-DFNet requires the following dependencies:

```
torch>=2.1.0
torch_geometric>=2.4.0
scikit-learn>=1.3.0
pandas>=2.1.1
numpy>=1.26.0
scipy>=1.11.3
lightning>=2.1.3
```

## Installation Guide
We suggest installing DBCL-DFNet through conda. Clone the GitHub repository and create a new conda virtual environment using the command provided below. The installation typically completes in about 15 minutes on a standard desktop computer.

```shell
# create a new conda virtual environment
conda create -n heterogatomics python=3.10
conda activate heterogatomics

# install required python dependencies
conda install pytorch==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg=2.4.0=*cu* -c pyg
conda install pytorch-scatter pytorch-cluster pytorch-sparse -c pyg
conda install pandas
conda install lightning -c conda-forge
conda install xgboost
conda install yacs


The BLCA, LGG, and RCC datasets used in our manuscript are stored in the `raw_data` folder and were all downloaded from the [UCSC Xena platform](https://xenabrowser.net/datapages/).  
<pre>
=============================================================================
Folder/File name               Description              
=============================================================================
├─RCC                          RCC dataset		
|   └─DNA.csv                  DNA methylation data
|   └─mRNA.csv                 Gene expression RNAseq data
|   └─miRNA.csv                miRNA mature strand expression RNAseq data
|   └─ClinicalMatrix.csv       Patient labels
|   └─1-5                    Fold 1 to 5 details
|     └─train_index.csv        Patient indices for training set
|     └─train_labels.csv       Labels for training set patients
|     └─test_index.csv         Patient indices for test set
|     └─test_labels.csv        Labels for test set patients
=============================================================================


## Running DBCL-DFNet with Our Datasets for Result Reproduction
To train DBCL-DFNet, the basic configurations for all hyperparameters are provided in `config.py`. Dataset-specific custom configurations, derived from hyperparameter tuning discussed in the manuscript, are available in the `configs/*.yaml` files.

DBCL-DFNet is designed with a modular architecture to separate distinct components.



You can run the module to perform the cancer diagnosis task. By setting the `RESULT.SAVE_MODEL` parameter to `True` in `configs.py`, the trained models will be saved in the `model` folder, which currently contains the models from our training. To use the model, the following command can be used:

```
python main-te.py --cfg configs/RCC.yaml
```


