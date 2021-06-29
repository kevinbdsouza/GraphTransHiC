"""
    File to load dataset based on user control from main file
"""
from data.HiC import HiCDataset


def LoadData(dataset_name):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """
    # handling for HiC dataset
    if dataset_name == 'HiC':
        return HiCDataset(dataset_name)
