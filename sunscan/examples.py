"""
Example datasets for sunscan package.
"""
import xarray as xr
from typing import List, Optional
from pathlib import Path
import warnings

from importlib import resources

def load_example_dataset(dataset_name) -> xr.Dataset:
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
    available = list_available_datasets()
    if dataset_name not in available:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
    
    traversable= resources.files('sunscan')
    
    with resources.as_file(traversable) as f:
        file_path = f / 'examples'/ f'{dataset_name}.nc'
        return xr.open_dataset(file_path)

def list_available_datasets() -> List[str]:
    """
    List all available example datasets.
    
    Returns
    -------
    List[str]
        List of available dataset names.
    """
    data_files = resources.files('sunscan') / 'examples'
    
    datasets = []
    for file in data_files.iterdir():
        if file.name.endswith('.nc') and file.is_file():
            # Extract dataset name from filename (remove .nc extension)
            name = file.name.replace('.nc', '')
            datasets.append(name)
    return sorted(datasets)

__all__ = ['load_example_dataset', 'list_available_datasets']
