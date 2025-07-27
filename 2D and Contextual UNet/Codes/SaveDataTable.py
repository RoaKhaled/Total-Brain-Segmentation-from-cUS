"""
Author: Roa'a Khaled
Affiliation: Department of Computer Engineering, Department of Condensed Matter Physics,
             University of Cádiz, , Puerto Real, 11519, Cádiz, Spain.
Email: roaa.khaled@gm.uca.es
Date: 2025-07-27
Project: This work is part of a predoctoral research and part of PARENT project that has received funding
         from the EU’s Horizon 2020 research and innovation program under the MSCA – ITN 2020, GA No 956394.

Description:
    This code defines functions for saving data dictionary in excel or csv files

Usage:
    import SaveDataTable

Notes:
    - for environment requirements check
"""

import csv
import os
import pandas as pd
########################################
#function to save data in csv file
def save_to_csv(filepath, dict_list, append=False):
    """Saves a list of dictionaries as a .csv file.

    :param str filepath: the output filepath
    :param List[Dict] dict_list: The data to store as a list of dictionaries.
        Each dictionary will correspond to a row of the .csv file with a column for each key in the dictionaries.
    :param bool append: If True, it will append the contents to an existing file.

    :Example:

    save_to_csv('data.csv', [{'id': '0', 'score': 0.5}, {'id': '1', 'score': 0.8}])
    """
    assert isinstance(dict_list, list) and all([isinstance(d, dict) for d in dict_list])
    open_mode = 'a' if append else 'w+'
    with open(filepath, mode=open_mode) as f:
        csv_writer = csv.DictWriter(f, dict_list[0].keys(), restval='', extrasaction='raise', dialect='unix')
        if not append or os.path.getsize(filepath) == 0:
            csv_writer.writeheader()
        csv_writer.writerows(dict_list)
########################################
#function to save data in xlsx file
def save_to_excel(file_path, data_to_append, sheet_name=None):
    """
    Appends data to an existing Excel file or creates a new one.

    Args:
        file_path (str): The path to the Excel file.
        data_to_append (dict): A dictionary where keys are column names and values are lists of data.
        sheet_name (str): Name of the sheet to append data. If None, data is appended to a new sheet.

    Returns:
        None
    """
    if os.path.exists(file_path):
        # Excel file exists, load it
        existing_excel_file = pd.ExcelFile(file_path)

        # Create a new Excel writer with the existing sheets
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            if sheet_name:
                # Append data to a new sheet
                writer.book = existing_excel_file.book
                writer.sheets = {ws.title: ws for ws in existing_excel_file.book.worksheets}
                df_to_append = pd.DataFrame(data_to_append)
                df_to_append.to_excel(writer, index=False, sheet_name='Sheet{}'.format(len(writer.sheets) + 1))
            else:
                # Append data to the same opened sheet
                df_to_append = pd.DataFrame(data_to_append)
                df_to_append.to_excel(writer, index=False)

            writer.save()
        print(f'Data has been appended to {file_path}')
    else:
        # Excel file doesn't exist, create it
        df = pd.DataFrame(data_to_append)
        df.to_excel(file_path, index=False, sheet_name=sheet_name)
        print(f'Excel file {file_path} has been created with the data')



