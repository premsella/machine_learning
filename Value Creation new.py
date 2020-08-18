# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:05:10 2019

@author: prems
"""
import datetime
import pandas as pd
import itertools
import numpy as np

this_brand = ['W', 'Joint Store']
this_main_brand = 'W'
f_product_master_sku = {'W': '../BaseFiles/W_SS19_Product_Master.csv',
                        'Aurelia': '../BaseFiles/Aurelia_SS19_Product_Master.csv'}
stock_mov_f = '../RegularInputFiles/Inward/Stock_Transfer_Movement.parquet'
f_store_master = '../BaseFiles/Store master A+W (June 1st 19).xlsx'

############################################################################################################

f_product_master_sku = f_product_master_sku[this_main_brand]
f_product_master_sku
pm_raw = pd.read_csv(f_product_master_sku)
pm = pm_raw.copy()
pm = pm.loc[pm['AG'].notnull()].copy()
pm['Size'] = pm['Size'].astype(str)
pm['Material'] = pm['Style_Code'] + '-' + pm['Size'].astype(str)
pm.reset_index(drop=True, inplace=True)
pm.loc[(pm['Item Category']=='BOTTOMWEAR') & (pm['Product Group'].isin(['SLIM PANTS','PARALLEL PANTS','FLARED PANTS','TULIP PANTS'])),'Product Group']='PANTS'
pm.columns

###########################################################################################################

dfst = pd.read_excel(f_store_master)
dfst.columns
dfst.rename(columns = {'Cluster':'Region','Store Name':'Store_Name'},inplace = True)
#st_brand_map = pd.read_csv(f_brand_store_mapper)
#dfst = dfst.merge(st_brand_map, on=['Store_Code'], how='left')
dfst = dfst[dfst['Brand'].isin(this_brand)].reset_index(drop=True)
dfst.shape

stk = pd.read_parquet('../RegularInputFiles/Inward/Stock_Transfer_Movement.parquet')

all_stores = stk.To_Location_ID.append(stk.From_Location_ID).unique()
all_stores = all_stores[np.in1d(all_stores,dfst.Store_Code)]

all_material = (stk.Style_Code+'-'+stk.Size).unique()
all_material = all_material[np.in1d(all_material,pm.SKU_Code)]

start = pd.to_datetime('2019-02-01')
end = pd.to_datetime('today')

date_range = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
    
a = (list(itertools.product(all_stores,all_material,date_range)))