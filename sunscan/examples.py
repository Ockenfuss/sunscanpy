"""
Example datasets for sunscan package.
"""
import xarray as xr
import pandas as pd
from typing import List, Optional
from pathlib import Path
import warnings

from importlib import resources

def load_example_sunscan(dataset_name) -> xr.Dataset:
    """
    Load an example dataset.
    
    Parameters
    ----------
    dataset_name : str, default 'sunscan'
        Name of the dataset to load. Use list_available_datasets() 
        to see available options.
        
    Returns
    -------
    xr.Dataset
        The loaded dataset.
        
    Raises
    ------
    ValueError
        If dataset_name is not available.
    FileNotFoundError
        If the dataset file cannot be found.
    """
    available = list_available_datasets(filetype='nc')
    if dataset_name not in available:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
    
    traversable= resources.files('sunscan')
    
    with resources.as_file(traversable) as f:
        file_path = f / 'examples'/ dataset_name
        return xr.open_dataset(file_path)

def load_example_scan_analysis(dataset_name: str) -> pd.DataFrame:
    """
    Load example scan analysis results.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the scan analysis results.
        
    Raises
    ------
    ValueError
        If dataset_name is not available.
    FileNotFoundError
        If the dataset file cannot be found.
    """
    available = list_available_datasets(filetype='csv')
    if dataset_name not in available:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
    
    traversable= resources.files('sunscan')
    
    with resources.as_file(traversable) as f:
        file_path = f / 'examples'/ dataset_name
        df=pd.read_csv(file_path, header=0, sep=';', date_format="%Y%m%d_%H%M%S", parse_dates=['time'], index_col='time')
    return df

def list_available_datasets(filetype=None) -> List[str]:
    """
    List all available example datasets.
    
    Returns
    -------
    List[str]
        List of available dataset names.
    """
    data_files = resources.files('sunscan') / 'examples'
    allowed_types= {'nc', 'csv'}
    filetype=filetype.strip('.') if filetype else None
    if filetype and filetype not in allowed_types:
        raise ValueError(f"Invalid file type '{filetype}'. Allowed types: {allowed_types}")
    datasets = []
    for file in data_files.iterdir():
        suffix=file.suffix.strip('.')
        if filetype is not None:
            if suffix != filetype:
                continue
        if suffix in allowed_types:
            datasets.append(file.name)
    return sorted(datasets)

__all__ = ['load_example_sunscan', 'list_available_datasets']
