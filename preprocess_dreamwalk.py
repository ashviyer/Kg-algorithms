#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:13:28 2024

@author: alejandroadriaquelozano
"""

""" preprocess file after exporting csv in neo4j so it matches for dreamwalk """
# Set path
import os 
from pathlib import Path
path = Path(__file__).resolve().parent
os.chdir(path)


import pandas as pd
import requests



data = pd.read_csv('data.csv')


data = data[~((data['_labels'] == ':Disease') & (data['source'] == 'OpenTargets'))]
data['_labels'] = data['_labels'].replace(':drug', ':Drug')
data['_labels'] = data['_labels'].replace(':disease', ':Disease')
data['_labels'] = data['_labels'].replace(':gene', ':Protein')

data = data[(data['_labels'] == ':Drug') | (data['_labels'] == ':Disease') | (data['_labels'] == ':Protein') | (data['_start'].notna())]

data = data[~data['type'].isin(['part_of_go', 'localized_in','part_of_pathway'])]



# Drugbank ATC    
atc_hierarchy = pd.read_csv('atc_hierarchy.csv')
atc_hierarchy=atc_hierarchy.drop(columns=['Unnamed: 0','dbID'])
atc_hierarchy = atc_hierarchy.dropna(subset=['id'])


merged_df = pd.merge(data,atc_hierarchy,on='id',how='left') 

merged_df = merged_df[~((merged_df['_labels'] == ':Drug') & (merged_df['drugID'].isna()))] #delete drugs with no drugbank ID
merged_df = merged_df[~((merged_df['_labels'] == ':Drug') & (merged_df['atcClassification'].isna()))] #delete drugs with no ATC

merged_df['class']=None
merged_df['uniprotID']=None

merged_df.to_csv('preprocessed_graph.csv', index=False)


