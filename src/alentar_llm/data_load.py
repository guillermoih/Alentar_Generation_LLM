# -*- coding: utf-8 -*-
import re
import pandas as pd


def get_cods_dict(file_path):
    """
    Read a CSV file and return a filtered dictionary mapping variable codes to labels. Only variables
    whose names start with 'F' or 'R' (except 'REVISADO') are included in the final dictionary.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing at least the columns 'Variable' and 'Label'.

    Returns
    -------
    dict
        A dictionary where keys are variable codes and values are their corresponding labels

    Example
    --------
    >>> cods = get_cods_dict("variables.csv")
    >>> cods["F01"]
    'Financial Performance'
    """
    
    
    df_cods = pd.read_csv(file_path)
    cod_names = df_cods.set_index('Variable')['Label'].to_dict()
    cod_names = {k: v for k, v in cod_names.items() \
                 if k.startswith('F') or (k.startswith('R') and k!='REVISADO')} # Filter non codes keys
    # cod_names = {k: v for k, v in cod_names.items() if v != '<none>'} # Filter keys with <none> value
    
    return cod_names


def clean_data(df, cods):
    df = df[df["REVISADO"] == 1] # Filter unrevised texts
    df = df[df['EJEIDiagnóstico'].str.len() > 1] # Filter short texts
    df = df[df['EJEIDiagnóstico'].str.contains(r'[a-zA-ZáéíóúÁÉÍÓÚñÑ]', na=False)]# Filter reports without text
    regex = '|'.join(map(re.escape, cods))
    df = df[~df['EJEIDiagnóstico'].str.contains(regex, na=False, regex=True)]
    return df
