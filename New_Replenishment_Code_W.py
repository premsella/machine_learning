import pandas as pd
import numpy as np
import time
from datetime import timedelta
from itertools import product
from  joblib import Parallel, delayed

brand       = 'W'
if brand == 'W': inw_wh_name = 'GYANKUNJ' 
elif brand == 'AURELIA': inw_wh_name = 'MANDI'
#inw_wh_name = 'GYANKUNJ'
date        = '27-Jan-2020'
output      = '27Jan20'
prevdate    = '26-Jan-2020'
#1 - Base Files
sm     = pd.read_csv('../Processed/BaseFiles/TCNS_Live_StoreMaster.csv')
#pm     = pd.read_parquet('../BaseFiles/Product_Master_W_(23-Sep-2019).parquet')
pm     = pd.read_parquet('../Processed/BaseFiles/Product_Master_'+brand+'_Consolidated.parquet')  ######################################################
theta  = pd.read_excel('../Processed/BaseFiles/Core_Rep_Theta_20Nov19.xlsx')
#groups = pd.read_parquet('../BaseFiles/Groups_Aurelia_20-7-2019.parquet')
#2 - Input Files
pickuplist        = pd.read_excel('../InputFiles/W core process Status.xlsx', 'Item wise')
latest_inv        = pd.read_parquet('../InputFiles/STOREINV_('+prevdate+').parquet')
whinv             = pd.read_csv('../InputFiles/WHINV_('+prevdate+').csv')
wh_cur_inv        = pd.read_csv('../InputFiles/CORE_WHINV_('+date+').csv')  ##########################################################################
bsq_data          = pd.read_parquet('../InputFiles/BSQ_Data.parquet')
stock_transfer_df = pd.read_parquet('../InputFiles/Stock_Transfer_Movement.parquet')
ist_orders        = pd.read_csv('../InputFiles/W_IST_'+date+'_All_Clusters.csv')
#fofo_ist          = pd.read_csv('../InputFiles/W_IST_COCO_FOFO_'+date+'.csv')
Sales             = pd.read_parquet('../InputFiles/Sales_(01-Jan-2019 to '+prevdate+').parquet')
#fofo_inv	  = pd.read_csv('../InputFiles/FOFOINV_('+prevdate+').csv')
#fofo_bsq	  = pd.read_parquet('../InputFiles/W_FOFO_BSQ.parquet')
#Daily_inv 	  = pd.read_parquet('../InputFiles/Daily_Inventory.parquet')
#3 Checking Valid Ouput Directory
pd.DataFrame(columns = ['Test']).to_parquet('../Output/'+output+'/'+brand+'/'+'Test.parquet')

def Merge_Variants(data,mapper,mode = 'Material'):
    if mode == 'Material':
        mapper = mapper[[mode,'UniqueID']].drop_duplicates()
        data = data.loc[data[mode].isin(mapper[mode])]
        data = data.merge(mapper,on = mode,how = 'left')
        data[mode] = data['UniqueID'].copy()
        data = data.drop('UniqueID',axis=1)
        return data
    elif mode == 'Style_Code':
        mapper['UniqueID'] = mapper['UniqueID'].str.rpartition('-')[0]    
        mapper = mapper[[mode,'UniqueID']].drop_duplicates()
        data = data.loc[data[mode].isin(mapper[mode])]
        data = data.merge(mapper,on = mode,how = 'left')
        data[mode] = data['UniqueID'].copy()
        data = data.drop('UniqueID',axis=1)
        return data

#pm['Material'] = pm['Material'].apply(lambda x: '19'+x if x.startswith('CR') else x) #**$**
sm.rename(columns={'Store Code':'Store_Code'}, inplace=True)
sm1 = sm[(sm.W == 1)&(sm['Channel'] == 'OWN')&(sm.Rep_Status=='ACTIVE')].reset_index(drop=True) #**$**
sm2 = sm[(sm.W == 1)&(sm['Channel'] != 'OWN')&(sm.Rep_Status!='CLOSED')].reset_index(drop=True)
sm = pd.DataFrame()
sm = sm1.append(sm2 , sort = False, ignore_index = True)
#pm = pm.loc[pm.CORE_THETA == 1]
#removed_stores = ['W451','W632','W455','W627','W469', 'W485'] #**$**
#sm = sm.loc[~sm.Store_Code.isin(removed_stores)]#**$**
live_stores = sm.Store_Code.unique().tolist()
print('Brand          :',brand)
#print('Removed Stores :',removed_stores)
print('Live Stores    :',len(live_stores))
print('Products       :',len(pm))

# WareHouse Inventory Preprocess
whinv.rename(columns = {'ITEMID':'Style_Code', 'INVENTSIZEID':'Size', 'qty':'WH_Qty', 'INVENTLOCATIONID':'Store_Code'}, inplace = True)
whinv['Style_Code'] = whinv['Style_Code'].astype(str).str.upper().str.strip()
whinv['Size'] = whinv['Size'].astype(str).str.upper().str.strip()
whinv['Store_Code'] = whinv['Store_Code'].astype(str).str.upper().str.strip()
whinv['Material'] = whinv['Style_Code']+'-'+whinv['Size']
whinv = Merge_Variants(whinv.copy(), pm.copy())
whinv = whinv.groupby('Material')['WH_Qty'].sum().reset_index()

pickuplist['From_Location_ID'] = inw_wh_name ####
pickuplist.rename(columns = {'Store code':'To_Location_ID', 'Qty':'Picklist_Qty'}, inplace = True) ####
pickuplist['Size'] = pickuplist['Size'].astype(str).str.strip() ####

#get_inwards_here
inward = stock_transfer_df[(stock_transfer_df['From_Location_ID']==inw_wh_name) & (stock_transfer_df['Style_Code'].isin(pm['Style_Code'].unique()))].reset_index(drop=True)
inward = inward.merge(pickuplist[['From_Location_ID','To_Location_ID', 'Style_Code', 'Size', 'Picklist_Qty']], on = ['From_Location_ID', 'To_Location_ID', 'Style_Code', 'Size'], how = 'outer', indicator = True) ####
inward.loc[inward._merge == 'right_only', 'Transfer_Qty'] = inward.loc[inward._merge == 'right_only', 'Picklist_Qty'] ####
inward.loc[inward._merge == 'right_only', 'Shipment_Date'] = pd.to_datetime(date) ####
inward.drop(['From_Location_ID'], axis=1, inplace=True)
inward['Order Date'] = np.nan
inward.loc[inward['Received_Date'].notnull(), 'Status'] = 'Receive'
inward.loc[inward.Status.isna(),'Status'] = 'Intransit'
inward.rename(columns={'Stock_Transfer_ID':'TO number', 'Shipment_Date':'Dispatch Date', 'Transfer_Qty':'Quantity', 
                           'To_Location_ID':'Store_Code'}, inplace=True)
iw_cols=['Store_Code','Style_Code','Size','Quantity','Status','Order Date','Dispatch Date','Received_Date']
inward=inward[iw_cols]
inward=inward[inward['Store_Code'].isin(live_stores)]
inward.reset_index(drop=True, inplace=True)
inward = Merge_Variants(inward.copy(),pm.copy(),'Style_Code')
inward['Material']=inward['Style_Code']+'-'+inward['Size']
#inward['Material'] = inward['Material'].apply(lambda x: '19'+x if x.startswith('CR') else x) #**$**
#print('Inwards :',inward.shape,inward.Quantity.sum())
#print(inward.Received_Date.describe())
Sales.columns
Sales = Merge_Variants(Sales.copy(),pm.copy())
Sales['Style_Code'] = Sales['Material'].str.rpartition('-')[0]
Sales = Sales.loc[Sales.Material.isin(pm.Material)]
Sales = Sales.loc[Sales.Store_Code.isin(sm.Store_Code)]
Sales['Week'] = Sales.Date.dt.week
Sales.Quantity *=-1
Sales = Sales.loc[Sales.Quantity>0]
Sales = Sales.groupby(['Date','Style_Code','Store_Code','Material','Size'])['Quantity'].sum().reset_index()

#Creating Inv and BSQ from BSQ data and Inv 
#Creating Inv and BSQ from BSQ data and Inv 
#print('BSQ-Data :',bsq_data.shape)
bsq_data = bsq_data[['Material', 'Size', 'Style_Code', 'Region','Store_Code','ROS_THRES', 'BSQ']]
bsq_data = bsq_data.merge(sm[['Store_Code', 'LT']], on = 'Store_Code', how = 'inner')
print('BSQ_Styes:',bsq_data.Style_Code.nunique())
bsq_data = Merge_Variants(bsq_data.copy(),pm.copy())
bsq_data = Merge_Variants(bsq_data.copy(),pm.copy(),'Style_Code')
print('BSQ_Styes:',bsq_data.Style_Code.nunique())
bsq_data = bsq_data.groupby(['Store_Code','Material','Region', 'Style_Code', 'Size'])['ROS_THRES', 'BSQ'].sum().reset_index()
#Using New Theta for core rep
print("\nNAN's in BSQ Data\n",bsq_data.isna().sum())
#bsq_data['LT'] = bsq_data['LT'].astype('int64')
#bsq_data = pd.merge_asof(bsq_data.sort_values(by = 'ROS_THRES'),theta.sort_values(by = 'ROS_THRES'),by = ['LT'],on='ROS_THRES',direction = 'backward')
#bsq_data.isna().sum()
#bsq_data.groupby(['Material'])['BSQ'].mean().round().reset_index().rename(columns = {'BSQ':'Avg_BSQ'})
#bsq_data.FAQ.fillna(8,inplace = True)
#bsq_data['BSQ'] = bsq_data[['BSQ','FAQ']].max(axis=1)

comb = pd.DataFrame(product(bsq_data.Store_Code.unique(),pm.UniqueID.unique()),columns = ['Store_Code', 'Material'])
comb['Style_Code'] = comb.Material.str.rpartition('-')[0]
comb['Size'] = comb.Material.str.rpartition('-')[2]
comb = comb.merge(bsq_data[['Store_Code','Material','Style_Code', 'Size', 'ROS_THRES', 'BSQ']], on = ['Store_Code', 'Material', 'Style_Code', 'Size'], how = 'outer')
comb.isna().sum()
comb = comb.merge(sm[['Store_Code', 'LT', 'IST_Cluster']], on = 'Store_Code', how = 'inner').rename(columns = {'IST_Cluster':'Region'})

comb['ROS_THRES_new'] = comb.groupby(['Material','Region'])['ROS_THRES'].transform('mean').round()
comb['BSQ_new'] = comb.groupby(['Material','Region'])['BSQ'].transform('mean').round()
comb.loc[comb.ROS_THRES.isna(), 'ROS_THRES'] = comb.loc[comb.ROS_THRES.isna(), 'ROS_THRES_new']
comb.loc[comb.BSQ.isna(), 'BSQ'] = comb.loc[comb.BSQ.isna(), 'BSQ_new']
comb.isna().sum()

comb['ROS_THRES_new'] = comb.groupby(['Material'])['ROS_THRES'].transform('mean').round()
comb['BSQ_new'] = comb.groupby(['Material'])['BSQ'].transform('mean').round()
comb.loc[comb.ROS_THRES.isna(), 'ROS_THRES'] = comb.loc[comb.ROS_THRES.isna(), 'ROS_THRES_new']
comb.loc[comb.BSQ.isna(), 'BSQ'] = comb.loc[comb.BSQ.isna(), 'BSQ_new']
print("COCO BSQ Data \n",comb.isna().sum())
comb.fillna(1, inplace = True)
#comb = comb.loc[~comb.ROS_THRES.isna()]
comb = comb[bsq_data.columns]
comb[['Store_Code', 'Material']].drop_duplicates()
bsq_data = comb.copy()

latest_inv.rename(columns = {'StoreID':'Store_Code','Closing Inventory':'Sim_Inv','ItemID':'Style_Code','Size ID':'Size'},inplace = True)
latest_inv['Style_Code'] = latest_inv['Style_Code'].str.upper()
latest_inv['Size'] = latest_inv['Size'].str.upper()
latest_inv = Merge_Variants(latest_inv.copy(),pm.copy(),'Style_Code')
latest_inv['Material'] = latest_inv['Style_Code']+'-'+latest_inv['Size']
latest_inv = latest_inv.groupby(['Store_Code', 'Material'])['Sim_Inv'].sum().reset_index()
#latest_inv['Material'] = latest_inv['Material'].astype(str).str.upper()
#latest_inv['Material'] = latest_inv['Material'].apply(lambda x: '19'+x if x.startswith('CR') else x) #**$**
#bsq_data['Material'] = bsq_data['Material'].apply(lambda x: '19'+x if x.startswith('CR') else x) #**$**
bsq_data = bsq_data.merge(latest_inv[['Store_Code','Material','Sim_Inv']].groupby(['Store_Code','Material'])['Sim_Inv'].sum().reset_index(),on = ['Store_Code','Material'],how='left')
bsq_data['WC_IST_Date'] = pd.to_datetime(date)
bsq_data = bsq_data.merge(sm[['Store_Code','Zone']].drop_duplicates(),on='Store_Code',how='left')
bsq_data.rename(columns={'ROS_THRES':'ROS'},inplace=True)
csi = bsq_data.groupby(['WC_IST_Date', 'Zone', 'Region', 'Store_Code','Size', 'Material']).agg({'ROS':'sum','BSQ':'sum', 'Sim_Inv':'sum'}).reset_index()
csi.loc[csi['BSQ']>10, 'BSQ'] = 10
csi.isna().sum()
#print('BSQ-Inv :',csi.shape)

#fofo_inv.rename(columns = {'STOREID':'Store_Code', 'ITEMID':'Style_Code','QTY':'SOH', 'INVENTSIZEID':'Size'},inplace = True)
#fofo_inv = fofo_inv[['Store_Code', 'Style_Code','Size','SOH']]
#fofo_inv=  fofo_inv.astype(str)
#for col in fofo_inv: fofo_inv[col] = fofo_inv[col].str.upper()
#fofo_inv['Material'] = fofo_inv['Style_Code']+'-'+fofo_inv['Size']
#fofo_inv = fofo_inv.loc[(fofo_inv['SOH']!='QTY')]
#fofo_inv['SOH'] = fofo_inv['SOH'].astype(int)
##fofo_inv = fofo_inv.drop_duplicates(['Store_Code','Material'])  #Done this stuff
#fofo_inv = fofo_inv.groupby(['Store_Code','Material'])['SOH'].mean().reset_index()
#fofo_inv['SOH'] = pd.np.floor(fofo_inv['SOH'])
#fofo_inv = fofo_inv[['Store_Code','Material','SOH']]
##Variants
#fofo_inv = fofo_inv.loc[fofo_inv.Store_Code.isin(sm.Store_Code)]
#fofo_inv = fofo_inv.loc[fofo_inv.Material.isin(pm.Material)]
#fofo_inv = fofo_inv.merge(pm[['Material','UniqueID']],on = 'Material',how = 'left')
#fofo_inv['Material'] = fofo_inv['UniqueID'].copy()
#fofo_inv = fofo_inv.groupby(['Store_Code','Material'])['SOH'].sum().reset_index()
#
#fofo_bsq = fofo_bsq.merge(pm[['Material','UniqueID']],on = 'Material',how = 'left')
#fofo_bsq['Material'] = fofo_bsq['UniqueID'].copy()
#fofo_bsq = fofo_bsq.groupby(['Store_Code','Material'])['BSQ'].sum().reset_index()
#fofo_bsq.loc[fofo_bsq.BSQ>8,'BSQ'] = 8
#
#comb1 = pd.DataFrame(product(fofo_bsq.Store_Code.unique(),pm.UniqueID.unique()), columns = ['Store_Code', 'Material'])
#comb1 = comb1.merge(fofo_bsq[['Store_Code','Material','BSQ']], on = ['Store_Code', 'Material'], how = 'outer')
#comb1 = comb1.merge(sm[['Store_Code', 'LT', 'IST_Cluster']], on = 'Store_Code', how = 'inner').rename(columns = {'IST_Cluster':'Region'})
##comb1 = comb1.merge(pm[['Material', 'Item Category', 'Product Group' ]], on = 'Store_Code', how = 'inner').rename(columns = {'IST_Cluster':'Region'})
#comb1.isna().sum()
#
#comb1['BSQ_new'] = comb1.groupby(['Material','Region'])['BSQ'].transform('mean').round()
#comb1.loc[comb1.BSQ.isna(), 'BSQ'] = comb1.loc[comb1.BSQ.isna(), 'BSQ_new']
#comb1.isna().sum()
#comb1['BSQ_new'] = comb1.groupby(['Material', 'LT'])['BSQ'].transform('mean').round()
#comb1.loc[comb1.BSQ.isna(), 'BSQ'] = comb1.loc[comb1.BSQ.isna(), 'BSQ_new']
#comb1.isna().sum()
#comb1['BSQ_new'] = comb1.groupby(['Material'])['BSQ'].transform('mean').round()
#comb1.loc[comb1.BSQ.isna(), 'BSQ'] = comb1.loc[comb1.BSQ.isna(), 'BSQ_new']
#comb1.isna().sum()
#comb1 = comb1.merge(comb.groupby('Material')['BSQ'].mean().round().reset_index().rename(columns = {'BSQ':'BSQ1'}), on = 'Material', how = 'left')
#comb1.loc[comb1.BSQ1>3, 'BSQ1'] = 3
#comb1.loc[comb1.BSQ.isna(), 'BSQ'] = comb1.loc[comb1.BSQ.isna(), 'BSQ1']
#print("FOFO BSQ Data ",comb1.isna().sum())
#comb1.fillna(1,inplace = True)
#fofo_bsq = comb1[['Store_Code','Material', 'BSQ']]

#fcsi = fofo_inv.merge(fofo_bsq[['Store_Code','Material','BSQ']],on=['Store_Code','Material'],how = 'right').rename(columns = {'SOH':'Sim_Inv'})
#fcsi['Size'] = fcsi['Material'].str.rpartition('-')[2]
#fcsi['WC_IST_Date'] = pd.to_datetime(date)
#fcsi.isna().sum()
#fcsi.BSQ.value_counts()
#fcsi.fillna(0, inplace = True)
#csi = pd.concat([csi,fcsi],ignore_index = True, sort = False)
#csi.isna().sum()
#print('BSQ-Inv :',csi.shape)

#Reading Scheduled IST 
#ist_orders = ist_orders.append(fofo_ist)
ist_orders.rename(columns={'Style code':'Style_Code'}, inplace=True)
ist_orders = ist_orders[['Zone', 'Region', 'Store_From', 'Store_To', 'Style_Code', 'Size', 'Qty']]
ist_orders.rename(columns={'Qty':'IST_Qty','Style code':'Style_Code'},inplace=True)
ist_orders['Style_Code']=ist_orders['Style_Code'].str.upper()
ist_orders['Size'] = ist_orders['Size'].str.upper()
ist_orders = Merge_Variants(ist_orders.copy(),pm.copy(),'Style_Code')
ist_orders['Material']=ist_orders['Style_Code']+'-'+ist_orders['Size']
#ist_orders['Material'] = ist_orders['Material'].apply(lambda x: '19'+x if x.startswith('CR') else x) #**$**
ist_orders = ist_orders[(ist_orders['Store_From'].isin(live_stores)) & (ist_orders['Store_To'].isin(live_stores))]
ist_orders = ist_orders.groupby(['Zone', 'Region', 'Store_From', 'Store_To', 'Style_Code', 'Size', 'Material'])['IST_Qty'].sum().reset_index()
print('ISTs           :',ist_orders.IST_Qty.sum())

#==========================================================
#==========================================================

transit_inv=inward[(~inward['Status'].isin(['Receive'])) & (~(inward['Dispatch Date'].isna()))]
transit_inv = transit_inv.loc[transit_inv['Dispatch Date']> pd.to_datetime(date) - pd.offsets.timedelta(days = 30)]
transit_inv=transit_inv.groupby(['Store_Code','Material'])['Quantity'].sum().reset_index()
transit_inv.rename(columns={'Quantity':'Transit_Qty'},inplace=True)
print('In Transit Qty :',transit_inv.Transit_Qty.sum())

ist_transit=ist_orders.groupby(['Store_To','Material'])['IST_Qty'].sum().reset_index()
ist_transit.rename(columns={'Store_To':'Store_Code'},inplace=True)
ist_orders['IST_Qty']=-ist_orders['IST_Qty']
ist_transit=ist_transit.append(ist_orders.groupby(['Store_From','Material'])['IST_Qty'].sum().reset_index().rename(columns={'Store_From':'Store_Code'}))

rep_sku=pd.merge(csi,transit_inv,on=['Store_Code','Material'],how='left')
rep_sku=pd.merge(rep_sku,ist_transit,on=['Store_Code','Material'],how='left')
'''
#Overriding BSQ
rep_sku = rep_sku.merge(Sales.loc[Sales.Week>Sales.Week.max()-3].groupby(['Store_Code','Material'])['Quantity'].sum().reset_index().rename(columns = {'Quantity':'3W_Avg_Sales'}),on=['Store_Code','Material'],how = 'left')
rep_sku = rep_sku.merge(Sales.loc[Sales.Week>Sales.Week.max()-6].groupby(['Store_Code','Material'])['Quantity'].sum().reset_index().rename(columns = {'Quantity':'6W_Avg_Sales'}),on=['Store_Code','Material'],how = 'left')
rep_sku = rep_sku.merge(Sales.loc[Sales.Week>Sales.Week.max()-9].groupby(['Store_Code','Material'])['Quantity'].sum().reset_index().rename(columns = {'Quantity':'9W_Avg_Sales'}),on=['Store_Code','Material'],how = 'left')
rep_sku.fillna(0,inplace = True)
#rep_sku['3W_Avg_Sales'] = rep_sku['3W_Avg_Sales']/3
#rep_sku['6W_Avg_Sales'] = rep_sku['6W_Avg_Sales']/6
#rep_sku['9W_Avg_Sales'] = rep_sku['9W_Avg_Sales']/9
rep_sku['Old_BSQ'] = rep_sku['BSQ'].copy()
rep_sku.loc[(rep_sku.ROS<0.3)&(rep_sku['9W_Avg_Sales']==0),'BSQ'] = 0
rep_sku.loc[(rep_sku.ROS<0.2)&(rep_sku['6W_Avg_Sales']==0),'BSQ'] = 0
rep_sku.loc[(rep_sku.ROS<0.1)&(rep_sku['3W_Avg_Sales']==0),'BSQ'] = 0
rep_sku.loc[(rep_sku.ROS>=0.3)&(rep_sku.ROS<=0.6)&(rep_sku['9W_Avg_Sales']==0),'BSQ'] = 1
rep_sku.loc[(rep_sku.ROS>0.6)&(rep_sku['9W_Avg_Sales']==0),'BSQ'] = 2
rep_sku.loc[(rep_sku.ROS>=0.5)&(rep_sku.ROS<=0.6)&(rep_sku['6W_Avg_Sales']==0),'BSQ'] = 1
rep_sku.loc[(rep_sku.ROS>0.5)&(rep_sku['9W_Avg_Sales']==0),'BSQ'] = 2
rep_sku.loc[(rep_sku.Old_BSQ>3)&(rep_sku['3W_Avg_Sales']==0),'BSQ'] = 3
print('BSQ reduced by :',int((1 - rep_sku.BSQ.sum()/rep_sku.Old_BSQ.sum())*10000)/100,'%')
'''
rep_sku['Transit_Qty'].fillna(0,inplace=True)
rep_sku['IST_Qty'].fillna(0,inplace=True)

rep_sku.rename(columns={'Sim_Inv':'Inv_Qty'}, inplace=True)
rep_sku['Req_Qty']=rep_sku['BSQ']-(rep_sku['Inv_Qty']+rep_sku['Transit_Qty']+rep_sku['IST_Qty'])

#making rows with -ive req_qty as 0
rep_sku.loc[rep_sku['Req_Qty']<0,'Req_Qty']=0
rep_sku=rep_sku[rep_sku['Store_Code'].isin(live_stores)]
print('Total Req Qty  :',rep_sku.Req_Qty.sum())
'''
rep_sku.loc[rep_sku.Req_Qty>1,'Req_Qty'] = 1
rep_sku = rep_sku.merge(pm[['Material','AG','Product Group']],on = 'Material',how='left')
rep_sku['Agg_demand'] = rep_sku.groupby(['Store_Code','AG','Product Group','Size'])['3W_Avg_Sales'].transform('sum')
rep_sku['Agg_demand'] = np.ceil(rep_sku['Agg_demand']/3).astype(int)
rep_sku['Agg_Req_Qty'] = rep_sku.groupby(['Store_Code','AG','Product Group','Size'])['Req_Qty'].transform('sum')
rep_sku = rep_sku.sort_values(['Store_Code','AG','Product Group','Size','ROS'],ascending = False)
rep_sku['n'] = rep_sku.groupby(['Store_Code','AG','Product Group','Size'])['Req_Qty'].cumsum()
rep_sku['Old_Req_Qty'] = rep_sku['Req_Qty'].copy()
rep_sku.loc[rep_sku['n']>rep_sku['Agg_demand'],'Req_Qty'] = 0
rep_sku['Agg_Req_Qty'] = rep_sku.groupby(['Store_Code','AG','Product Group','Size'])['Req_Qty'].transform('sum')
print('Reduced Req Qty:',rep_sku.Req_Qty.sum())
'''
#***********************4. Using Priority Logic to determine Alloc Qty ********************
# ACT WH Inv
wh_cur_inv.rename(columns={'ITEMID':'Style_Code', 'INVENTLOCATIONID':'WH_Name', 'INVENTSIZEID':'Size', 'qty':'WH_Qty'}, inplace=True)
wh_cur_inv['WH_Name'] = wh_cur_inv['WH_Name'].astype(str)
wh_cur_inv['WH_Name'] = wh_cur_inv['WH_Name'].str.upper()
wh_cur_inv = wh_cur_inv[wh_cur_inv['WH_Name']==inw_wh_name].reset_index(drop=True)
wh_cur_inv = wh_cur_inv[['Style_Code', 'Size', 'WH_Qty']].copy()
wh_cur_inv['Style_Code'] = wh_cur_inv['Style_Code'].astype('str')
wh_cur_inv['Style_Code'] = wh_cur_inv['Style_Code'].str.upper()
wh_cur_inv['Size'] = wh_cur_inv['Size'].astype('str')
wh_cur_inv['Size'] = wh_cur_inv['Size'].str.upper()
wh_cur_inv = Merge_Variants(wh_cur_inv.copy(),pm.copy(),'Style_Code')
wh_cur_inv['Material'] = wh_cur_inv['Style_Code'] + '-' + wh_cur_inv['Size']
#wh_cur_inv['Material'] = wh_cur_inv['Material'].apply(lambda x: '19'+x if x.startswith('CR') else x) #**$**
wh_cur_inv = wh_cur_inv.groupby(['Material'])['WH_Qty'].sum().reset_index().rename(columns = {'WH_Qty':'Core_WH_Inv'})
wh_cur_inv = whinv.merge(wh_cur_inv, on = ['Material'], how = 'outer')
wh_cur_inv.loc[~wh_cur_inv.Core_WH_Inv.isna(), 'WH_Qty'] = wh_cur_inv.loc[~wh_cur_inv.Core_WH_Inv.isna(), 'Core_WH_Inv']
wh_cur_inv.drop(columns = 'Core_WH_Inv', inplace = True)
print('WH_Qty         :',wh_cur_inv.WH_Qty.sum())

df1=pd.merge(rep_sku.groupby(['Material'])['Req_Qty'].sum().reset_index(),wh_cur_inv,on=['Material'],how='left')
df1.loc[df1['WH_Qty'].isnull(),'WH_Qty']=0
df1.rename(columns={'Req_Qty':'Total_Req_Qty'},inplace=True)
#df1['Status'] = 'Shortage'
#df1.loc[df1.Total_Req_Qty < df1.WH_Qty, 'Status'] = 'Excess'
#df1.Status.value_counts()

pm['Core_Product'] = pm['Material'].apply(lambda x: 'CR' in x)
#pm['Classic_Product'] = pm['Story'].apply(lambda x: 'classic' in [i.lower() for i in x.split()])

pm.loc[pm['Core_Product']==True,'Priority_Product'] = True
pm.isna().sum()
#pm.loc[(pm['Classic_Product']==True) | (pm['Core_Product']==True), 'Priority_Product'] = True
pm['Priority_Product'].fillna(True, inplace=True)

#cc_sku = pm[pm['Priority_Product']==True]['Material'].unique()  ##################################################################################
cc_sku = pm.UniqueID.unique()

#a = rep_sku.merge(sm[['Store_Code', 'City', 'Channel', 'IST_Cluster']], on = 'Store_Code', how = 'left')
#a = a.groupby(['Material', 'City','Channel', 'IST_Cluster']).agg({ 'Inv_Qty':'sum','BSQ':'sum','Req_Qty':'sum'}).reset_index()
#a = a.merge(df1[['Material','WH_Qty', 'Status']], on = 'Material', how = 'left')
#a['SOH/BSQ'] = a['Inv_Qty']/a['BSQ']
#a['Flag'] = 0
#a.loc[(a['SOH/BSQ']>1) & (a.Channel == 'OWN') & (a.IST_Cluster != 'NO CLUSTER') & (a.WH_Qty > 0) & (a.Req_Qty > 0) & (a.Inv_Qty > 0),'Flag'] = 1
#a = a.loc[a.Flag == 1]
#a.Status.value_counts()

#rep_sku.merge(a, on = ['Material', 'City', 'Channel', 'IST_Cluster'], how = 'left')
#rep_sku.columns

def alloc_material(df_alloc, whinv, sku):
    print("SKU"  ,sku)
    pending_req_qty=df_alloc[df_alloc['Material']==sku]['Pending Req Qty'].sum()
    remaining_wh_qty=whinv[whinv['Material']==sku]['WH_Qty'].mean()
    print(pending_req_qty,remaining_wh_qty)
    if((remaining_wh_qty>0) & (pending_req_qty>0)):
        while remaining_wh_qty>0:
#                st1 = df_alloc[(df_alloc['Material']==sku) & (df_alloc['Pending Req Qty']>0)].copy() 
#                st1 = st1[st1['Req_by_BSQ']==st1['Req_by_BSQ'].max()]
#                st1 = st1[st1['ROS']==st1['ROS'].max()]
#                st1 = st1.Store_Code.unique()
            st1=df_alloc[(df_alloc['Material']==sku) & 
                         (df_alloc['Pending Req Qty']>0)].sort_values(['Req_by_BSQ','ROS'],ascending=[False,False])['Store_Code'].unique()
            idx = (df_alloc['Material']==sku) & (df_alloc['Store_Code']==st1[0])
            df_alloc.loc[idx,['Alloc Qty','Pending Req Qty']] += 1,-1
#                df_alloc['Req_by_BSQ'] = df_alloc['Pending Req Qty']/df_alloc['BSQ']
            df_alloc.loc[idx,'Req_by_BSQ']=df_alloc.loc[idx,'Pending Req Qty'] / df_alloc.loc[idx,'BSQ']
            
            remaining_wh_qty-=1
            pending_req_qty=df_alloc[df_alloc['Material']==sku]['Pending Req Qty'].sum()
            if pending_req_qty<=0:
                break
    return df_alloc

def alloc_rep_pri_qty(whinv ,df1 ,cc_sku, df_st ,debug):
#    from  joblib import Parallel, delayed

    #first of all see which SKUs can be fully fulfilled
    df_alloc=pd.merge(df_st,whinv,on=['Material'],how='left')
    df_alloc['Req_by_BSQ']=df_alloc['Req_Qty']/df_alloc['BSQ']
    df_alloc['Alloc Qty']=0
    df_alloc['Pending Req Qty']=df_alloc['Req_Qty']
    df_alloc.loc[df_alloc['WH_Qty'].isnull(),'WH_Qty']=0
    df_alloc['Total_Req_Qty']=df_alloc.groupby(['Material'])['Req_Qty'].transform('sum')
    
    df_alloc.loc[df_alloc['Material'].isin(cc_sku), 'Priority_Rep'] = True
    total_sku_wh_req_df = df_alloc[df_alloc['Material'].isin(cc_sku)].groupby('Material').agg({'Req_Qty':'sum', 'WH_Qty':'mean'}).reset_index()
    total_sku_wh_req_df.loc[total_sku_wh_req_df['WH_Qty']<total_sku_wh_req_df['Req_Qty'], 'Stocked_Out_WH_Qty'] = total_sku_wh_req_df['Req_Qty'] - total_sku_wh_req_df['WH_Qty']
    total_sku_wh_req_df['Stocked_Out_WH_Qty'].fillna(0, inplace=True)
#    df_alloc.loc[(df_alloc['Material'].isin(cc_sku)), 'Alloc Qty'] = df_alloc.loc[(df_alloc['Material'].isin(cc_sku)) ,'Req_Qty']
#    df_alloc.loc[(df_alloc['Material'].isin(cc_sku)),'Pending Req Qty'] = 0
#    return df_alloc #--------------------------------HERE--------------------------------

    df_alloc = df_alloc.merge(total_sku_wh_req_df[['Material', 'Stocked_Out_WH_Qty']].drop_duplicates(), on=['Material'], how='left').drop_duplicates().reset_index(drop=True)
   
#    df_alloc['Stocked_Out_WH_Qty'].fillna(0, inplace=True)
#    df_alloc['Priority_Rep'].fillna(False, inplace=True)
    
#    full_sku = df1[df1['WH_Qty']>=df1['Total_Req_Qty']]['Material'].unique()
#    full_sku = [x for x in full_sku if x not in cc_sku]
    
#    part_sku = df1[(df1['WH_Qty']<df1['Total_Req_Qty']) & (df1['WH_Qty']>0)]['Material'].unique()
#    part_sku = [x for x in part_sku if x not in cc_sku]
    
#    df_alloc.loc[(df_alloc['Material'].isin(full_sku)), 'Alloc Qty'] = df_alloc.loc[(df_alloc['Material'].isin(full_sku)) ,'Req_Qty']
#    df_alloc.loc[(df_alloc['Material'].isin(full_sku)),'Pending Req Qty'] = 0
     
#     for sku in full_sku:
#         total_req_qty=df_alloc[df_alloc['Material']==sku]['Req_Qty'].sum()
#         wh_qty=df_alloc[df_alloc['Material']==sku]['WH_Qty'].sum()
#         if wh_qty<total_req_qty:
#           print('Error! for sku - '+str(int(sku))+' in full_sku list : wh_qty < total_req_qty')
#         else:
#             df_alloc.loc[df_alloc['Material']==sku,'Alloc Qty']=df_alloc.loc[df_alloc['Material']==sku,'Req_Qty']
#             df_alloc.loc[(df_alloc['Material']==sku),'Pending Req Qty']=0
#             #Check - total alloc qty should be <= WH Inv
#     #now for partial fulfillable SKUs we will go by the priority list of channelsi
#    a=1
#    for sku in cc_sku: #changed from part_sku to cc_sku
#        a = a+1
#        print(a,sku)
#        print("Going inside Loop")
    mat = []
    mat.append(Parallel(n_jobs = -1)(delayed(alloc_material)(df_alloc.loc[df_alloc.Material == sku].copy(),whinv.copy(),sku) for sku in cc_sku))
    df = pd.DataFrame()
    for i in mat: df = df.append(i, sort = False, ignore_index = False)
#        pending_req_qty=df_alloc[df_alloc['Material']==sku]['Pending Req Qty'].sum()
#        remaining_wh_qty=whinv[whinv['Material']==sku]['WH_Qty'].mean()
#        print(remaining_wh_qty, pending_req_qty)
#        if((remaining_wh_qty>0) & (pending_req_qty>0)):
#            while remaining_wh_qty>0:
##                print(remaining_wh_qty, pending_req_qty)
##                st1 = df_alloc[(df_alloc['Material']==sku) & (df_alloc['Pending Req Qty']>0)].copy() 
##                st1 = st1[st1['Req_by_BSQ']==st1['Req_by_BSQ'].max()]
##                st1 = st1[st1['ROS']==st1['ROS'].max()]
##                st1 = st1.Store_Code.unique()
#                st1=df_alloc[(df_alloc['Material']==sku) & 
#                             (df_alloc['Pending Req Qty']>0)].sort_values(['Req_by_BSQ','ROS'],ascending=[False,False])['Store_Code'].unique()
#                idx = (df_alloc['Material']==sku) & (df_alloc['Store_Code']==st1[0])
#                df_alloc.loc[idx,['Alloc Qty','Pending Req Qty']] += 1,-1
##                df_alloc['Req_by_BSQ'] = df_alloc['Pending Req Qty']/df_alloc['BSQ']
#                df_alloc.loc[idx,'Req_by_BSQ']=df_alloc.loc[idx,'Pending Req Qty'] / df_alloc.loc[idx,'BSQ']
#                
#                remaining_wh_qty-=1
#                pending_req_qty=df_alloc[df_alloc['Material']==sku]['Pending Req Qty'].sum()
#                if pending_req_qty<=0:
#                    break
#    return df_alloc
    return df

def Check_MOQ(rep_alloc, moq = 5):
    st_moq = rep_alloc.groupby('Store_Code')['Alloc Qty'].sum().reset_index().rename(columns = {'Alloc Qty':'Store_Alloc_Qty'})
    rep_alloc = rep_alloc.merge(st_moq, on = 'Store_Code', how = 'left')
    rep_alloc['MOQ_Flag'] = 0
    rep_alloc.loc[rep_alloc.Store_Alloc_Qty < moq, 'MOQ_Flag'] = 1
    rep_alloc.MOQ_Flag.sum()
    return rep_alloc

def Update_alloc(rep, wh_cur_inv, df_st, cc_sku):
    rep.loc[rep.MOQ_Flag == 1, 'Alloc Qty'] = 0
    rep.loc[rep.MOQ_Flag == 1, 'Pending Req Qty'] = 0
    rep['Req_Qty'] = rep['Pending Req Qty']
    upd_wh = rep.groupby('Material')['Alloc Qty'].sum().reset_index()
    wh_cur_inv = wh_cur_inv.merge(upd_wh, on = 'Material', how = 'left')
    wh_cur_inv['WH_Qty'] = wh_cur_inv['WH_Qty'] - wh_cur_inv['Alloc Qty']
    wh_cur_inv.drop(columns = 'Alloc Qty', inplace = True)
    rep.drop(columns = ['WH_Qty','Req_by_BSQ', 'Alloc Qty', 'Pending Req Qty', 'Total_Req_Qty','Priority_Rep', 'Stocked_Out_WH_Qty', 'Store_Alloc_Qty', 'MOQ_Flag'],inplace = True)
    rep.columns
    return alloc_rep_pri_qty(wh_cur_inv ,df1 ,cc_sku, rep ,0)
    
print('\nCalculating Replenishment 1--')
t = time.time()
rep_alloc = alloc_rep_pri_qty(wh_cur_inv,df1,cc_sku, rep_sku,0)
#a = rep_alloc.copy()
#rep_alloc = a.copy()
#rep = rep_alloc.copy()
rep = Check_MOQ(rep_alloc)
updated_rep = Update_alloc(rep.copy(), wh_cur_inv.copy(), df1.copy(), cc_sku.copy())
print('Time Taken     :',time.time()-t)
rep_alloc.to_parquet('../Output/'+output+'/'+brand+'/'+'Rep1_Alloc.parquet')
print('Rep1_Alloc_Qty :',rep_alloc['Alloc Qty'].sum())
print('Pending_Qty    :',rep_alloc['Pending Req Qty'].sum())
rep_alloc['Style_Code'] = rep_alloc['Material'].str.rpartition('-')[0]
#print('Rep1_Alloc_Qty :',updated_rep['Alloc Qty'].sum())
#print('Pending_Qty    :',updated_rep['Pending Req Qty'].sum())


Sales1 = Sales.loc[Sales.Date >= pd.to_datetime('today') - timedelta(pd.to_datetime('today').weekday()) - timedelta(days = 22)]
Sales1.Date.describe()
Sku_sales1 = Sales1.groupby(['Store_Code', 'Style_Code', 'Size'])['Quantity'].sum().reset_index().rename(columns = {'Quantity':'Sales_3W'})
#Style_sales1 = Sales1.groupby(['Store_Code', 'Style_Code'])['Quantity'].sum().reset_index().rename(columns = {'Quantity':'Style_Sales_3W'})

Sales2 = Sales.loc[Sales.Date >= pd.to_datetime('today') - timedelta(pd.to_datetime('today').weekday()) - timedelta(days = 29)]
Sku_sales2 = Sales2.groupby(['Store_Code', 'Style_Code', 'Size'])['Quantity'].sum().reset_index().rename(columns = {'Quantity':'Sales_4W'})
Sales3 = Sales.loc[Sales.Date >= pd.to_datetime('today') - timedelta(pd.to_datetime('today').weekday()) - timedelta(days = 36)]
Sku_sales3 = Sales3.groupby(['Store_Code', 'Style_Code', 'Size'])['Quantity'].sum().reset_index().rename(columns = {'Quantity':'Sales_5W'})
Sales4 = Sales.loc[Sales.Date >= pd.to_datetime('today') - timedelta(pd.to_datetime('today').weekday()) - timedelta(days = 43)]
Sku_sales4 = Sales4.groupby(['Store_Code', 'Style_Code', 'Size'])['Quantity'].sum().reset_index().rename(columns = {'Quantity':'Sales_6W'})
Sales5 = Sales.loc[Sales.Date >= pd.to_datetime('today') - timedelta(pd.to_datetime('today').weekday()) - timedelta(days = 57)]
Sku_sales5 = Sales5.groupby(['Store_Code', 'Style_Code', 'Size'])['Quantity'].sum().reset_index().rename(columns = {'Quantity':'Sales_8W'})

Daily_inv 	  = pd.read_parquet('../InputFiles/Daily_Inventory.parquet')
Daily_inv = Merge_Variants(Daily_inv.copy(),pm.copy())
Daily_inv = Daily_inv.groupby(['Date', 'Store_Code', 'Material'])['CalcStockQty'].sum().reset_index()
Daily_inv_4W = Daily_inv.loc[Daily_inv.Date >= pd.to_datetime('today') - timedelta(pd.to_datetime('today').weekday()) - timedelta(days = 29)]
Daily_inv_4W['Days'] = 0
Daily_inv_4W.loc[Daily_inv_4W.CalcStockQty >0, 'Days'] = 1
Sku_inv_4W = Daily_inv_4W.groupby(['Store_Code', 'Material'])['Days'].sum().reset_index()
Sku_inv_4W = Sku_inv_4W.loc[Sku_inv_4W.Material.isin(pm.Material.unique())].reset_index(drop = True)
Sku_inv_4W['Availability'] = Sku_inv_4W['Days']/28
Sku_inv_4W['Style_Code'] = Sku_inv_4W.Material.str.rpartition('-')[0]
Sku_inv_4W['Size'] = Sku_inv_4W.Material.str.rpartition('-')[2]
Daily_inv_8W = Daily_inv.loc[Daily_inv.Date >= pd.to_datetime('today') - timedelta(pd.to_datetime('today').weekday()) - timedelta(days = 57)]
Daily_inv_8W['Days'] = 0
Daily_inv_8W.loc[Daily_inv_8W.CalcStockQty >0, 'Days'] = 1
Sku_inv_8W = Daily_inv_8W.groupby(['Store_Code', 'Material'])['Days'].sum().reset_index()
Sku_inv_8W = Sku_inv_8W.loc[Sku_inv_8W.Material.isin(pm.Material.unique())].reset_index(drop = True)
Sku_inv_8W['Availability'] = Sku_inv_8W['Days']/56
Sku_inv_8W['Style_Code'] = Sku_inv_8W.Material.str.rpartition('-')[0]
Sku_inv_8W['Size'] = Sku_inv_8W.Material.str.rpartition('-')[2]

#Style_inv = Sku_inv.groupby(['Store_Code', 'Style_Code'])['Days'].sum().reset_index()
#Style_inv = Style_inv.merge(pm[['Style_Code', 'Count_Sizes']].drop_duplicates(), on = 'Style_Code', how = 'inner')
#Style_inv['New_Size'] = 28 * Style_inv.Count_Sizes
#Style_inv['Flag'] = 0
#Style_inv.loc[Style_inv.New_Size < Style_inv.Days, 'Flag'] =1
#Style_inv['Availability'] = Style_inv['Days'] / Style_inv['New_Size']
#Style_inv.loc[Style_inv.Availability > 1, 'Availability'] = 1

rep_alloc = rep_alloc.merge(Sku_sales1, on = ['Store_Code', 'Style_Code', 'Size'], how = 'left')
#rep_alloc = rep_alloc.merge(Style_sales1, on = ['Store_Code', 'Style_Code'], how = 'left')
rep_alloc = rep_alloc.merge(Sku_sales2, on = ['Store_Code', 'Style_Code', 'Size'], how = 'left')
rep_alloc = rep_alloc.merge(Sku_sales3, on = ['Store_Code', 'Style_Code', 'Size'], how = 'left')
rep_alloc = rep_alloc.merge(Sku_sales4, on = ['Store_Code', 'Style_Code', 'Size'], how = 'left')
rep_alloc = rep_alloc.merge(Sku_sales5, on = ['Store_Code', 'Style_Code', 'Size'], how = 'left')

rep_alloc = rep_alloc.merge(Sku_inv_4W[['Store_Code', 'Style_Code', 'Size', 'Availability']], on = ['Store_Code', 'Style_Code', 'Size'], how = 'left').rename(columns = {'Availability':'SKU_Availability_4W'})
rep_alloc = rep_alloc.merge(Sku_inv_8W[['Store_Code', 'Style_Code', 'Size', 'Availability']], on = ['Store_Code', 'Style_Code', 'Size'], how = 'left').rename(columns = {'Availability':'SKU_Availability_8W'})
#rep_alloc = rep_alloc.merge(Style_inv[['Store_Code', 'Style_Code', 'Availability']], on = ['Store_Code', 'Style_Code'], how = 'left').rename(columns = {'Availability':'Style_Availability'})
rep_alloc.Sales_4W.fillna(0,inplace = True)
rep_alloc.Sales_3W.fillna(0,inplace = True)
rep_alloc.Sales_5W.fillna(0,inplace = True)
rep_alloc.Sales_6W.fillna(0,inplace = True)
rep_alloc.Sales_8W.fillna(0,inplace = True)
#rep_alloc.SKU_Availability_8W.fillna(0,inplace = True)
#rep_alloc.SKU_Availability_4W.fillna(0,inplace = True)
#rep_alloc.SKU_4W_Sales.fillna(0,inplace = True)
#rep_alloc.Style_4W_Sales.fillna(0,inplace = True)
#rep_alloc.SKU_Availability.fillna(0,inplace = True)
#rep_alloc.Style_Availability.fillna(0,inplace = True)
rep_alloc.isna().sum()
rep_alloc.to_parquet('../Output/'+output+'/'+brand+'/'+'New_Rep_Alloc.parquet')

'''
# Rep-2
pm = pm.merge(groups[['Material','Price_Band','Color_Group','Length_Group']],on='Material',how='left')
#product_subset = ['Item Category','AG','Product Group','Price_Band'] # + colour ...?
product_subset = ['Item Category','AG','Product Group'] # + colour ...?
wh_inv = wh_cur_inv.merge(rep_alloc.groupby(['Material'])['Alloc Qty'].sum().reset_index(),on='Material',how='left')
wh_inv.fillna(0,inplace = True)
wh_inv['WH_Qty'] = wh_inv['WH_Qty'] - wh_inv['Alloc Qty']
wh_inv = wh_inv.loc[wh_inv.WH_Qty>0]
wh_inv = wh_inv.merge(pm[['Material']+product_subset+['Size']],on='Material',how='left')

pending_rep = rep_alloc.loc[rep_alloc['Pending Req Qty']>0,['Store_Code','Material','ROS','Pending Req Qty']].copy()
pending_rep = pending_rep.merge(pm[['Material','Size']+product_subset],on=['Material'],how='left')
rep_matrix  = pending_rep.groupby(['Store_Code']+product_subset+['Size'])['Pending Req Qty'].sum().reset_index()
rep_matrix = rep_matrix.merge(wh_inv,on = product_subset+['Size'],how = 'left')
rep_alloc['New_Inv_Qty'] = rep_alloc['Inv_Qty'] + rep_alloc['Alloc Qty'] + rep_alloc['Transit_Qty'] + rep_alloc['IST_Qty']
rep_matrix = rep_matrix.merge(rep_alloc[['Store_Code','Material','ROS','BSQ','Inv_Qty','New_Inv_Qty']],on = ['Store_Code','Material'],how='left')
rep_matrix = rep_matrix.dropna()
rep_matrix = rep_matrix.sort_values('WH_Qty')
rep_matrix['Inv-BSQ'] = rep_matrix['New_Inv_Qty'] - rep_matrix['BSQ']
rep_matrix = rep_matrix.loc[rep_matrix['Inv-BSQ'].isin([0,1])]
rep_matrix['Key'] = rep_matrix['Store_Code'] + rep_matrix[product_subset[0]] + rep_matrix[product_subset[1]] + rep_matrix[product_subset[2]]  + rep_matrix['Size'] 
rep2 = pd.DataFrame()
print('\nCalculating Replenishment 2--')
t = time.time()
for mat in rep_matrix.Material.unique():
    idx = (rep_matrix.Material == mat) & (rep_matrix['Inv-BSQ']==0) & (rep_matrix['Pending Req Qty']>0)
    m = rep_matrix.loc[idx].sort_values('ROS',ascending = False)
    if len(m) == 0 : continue
    qty = int(m['WH_Qty'].iloc[0])
    m = m.iloc[:qty]
    rep2 = rep2.append(m[['Store_Code','Material']])
    qty = qty - len(m)
    rep_matrix.loc[idx,'WH_Qty'] = qty
    rep_matrix.loc[rep_matrix['Key'].isin(m.Key),'Pending Req Qty']-=1
    if qty > 0 :
        idx = (rep_matrix.Material == mat) & (rep_matrix['Inv-BSQ']==1) & (rep_matrix['Pending Req Qty']>0)
        m = rep_matrix.loc[idx].sort_values('ROS')
        if len(m) == 0 : continue
        m = m.iloc[:qty]
        rep2 = rep2.append(m[['Store_Code','Material']])
        qty = qty - len(m)
    rep_matrix.loc[idx,'WH_Qty'] = qty
    rep_matrix.loc[rep_matrix['Key'].isin(m.Key),'Pending Req Qty']-=1
print(time.time()-t)
rep2['Rep2_Alloc_Qty'] = 1
print('Rep2_Alloc_Qty :',rep2['Rep2_Alloc_Qty'].sum())
rep2.to_parquet('../Output/'+date+'/Rep2.parquet')

#=================================ANALYSIS=====================================
print('\nCreating Analysis File--')
check = rep_alloc[['Store_Code','Material','ROS','BSQ', 'Inv_Qty', 'Transit_Qty', 'IST_Qty', 'Req_Qty', 'Alloc Qty', 'Pending Req Qty', 'New_Inv_Qty']].rename(columns = {'New_Inv_Qty':'Inv_Qty_After_Rep1'})
check['Inv_Qty_After_Transit'] = check['Inv_Qty']+check['Transit_Qty']+check['IST_Qty']
check = check.merge(rep2,on=['Store_Code','Material'],how='left')
check.fillna(0,inplace = True)
check['Inv_Qty_After_Rep2'] = check['Inv_Qty_After_Rep1'] + check['Rep2_Alloc_Qty']
check = check[['Store_Code', 'Material','ROS', 'BSQ', 'Inv_Qty', 'Transit_Qty', 'IST_Qty',
               'Req_Qty', 'Alloc Qty', 'Pending Req Qty', 'Inv_Qty_After_Transit',
               'Inv_Qty_After_Rep1', 'Rep2_Alloc_Qty', 'Inv_Qty_After_Rep2']].rename(columns = {'Alloc Qty':'Rep1_Alloc_Qty'})
check['Avl0'],check['Avl1'],check['Avl2'],check['Avl3'] = 0,0,0,0
check.loc[check.Inv_Qty>0,'Avl0'] = 1
check.loc[check.Inv_Qty_After_Transit>0,'Avl1'] = 1
check.loc[check.Inv_Qty_After_Rep1>0,'Avl2'] = 1
check.loc[check.Inv_Qty_After_Rep2>0,'Avl3'] = 1
check['#Material_Initial'] = check.groupby('Store_Code')['Avl0'].transform('sum')
check['#Material_After_Transit'] = check.groupby('Store_Code')['Avl1'].transform('sum')
check['#Material_After_Rep1'] = check.groupby('Store_Code')['Avl2'].transform('sum')
check['#Material_After_Rep2'] = check.groupby('Store_Code')['Avl3'].transform('sum')            
check['Total_Rep_Qty'] = check['Rep1_Alloc_Qty'] + check['Rep2_Alloc_Qty']
check = check.drop(['Avl0','Avl1','Avl2','Avl3'],axis=1)
check = check.merge(Sales.loc[Sales.Week>Sales.Week.max()-3].groupby(['Store_Code','Material'])['Quantity'].sum().reset_index().rename(columns = {'Quantity':'3W_Sales'}),on=['Store_Code','Material'],how = 'left')
check = check.merge(Sales.loc[Sales.Week>Sales.Week.max()-6].groupby(['Store_Code','Material'])['Quantity'].sum().reset_index().rename(columns = {'Quantity':'6W_Sales'}),on=['Store_Code','Material'],how = 'left')
check = check.merge(Sales.loc[Sales.Week>Sales.Week.max()-9].groupby(['Store_Code','Material'])['Quantity'].sum().reset_index().rename(columns = {'Quantity':'9W_Sales'}),on=['Store_Code','Material'],how = 'left')
check.fillna(0,inplace = True)
check = check[['Store_Code', 'Material','3W_Sales','6W_Sales','9W_Sales','ROS','BSQ', 'Inv_Qty', '#Material_Initial','Transit_Qty', 'IST_Qty','Inv_Qty_After_Transit','#Material_After_Transit',
               'Req_Qty', 'Rep1_Alloc_Qty', 'Pending Req Qty', 'Inv_Qty_After_Rep1', '#Material_After_Rep1','Rep2_Alloc_Qty', 'Inv_Qty_After_Rep2',
               '#Material_After_Rep2','Total_Rep_Qty']]
check.to_csv('../Output/'+date+'/Analysis.csv')


rep = check[['Store_Code','Material','Inv_Qty','Total_Rep_Qty']].rename(columns = {'Total_Rep_Qty':'Rep_Qty'})
print('Total_Rep_Qty  :',rep['Rep_Qty'].sum())
rep = rep.merge(pm[['Material','Style_Code','Size']],on = 'Material',how = 'left')
rep['From'] = 'WareHouse'
rep['WC_Rep_Date'] = pd.to_datetime(date)
rep = rep[['From','Store_Code','Style_Code','Size','Rep_Qty','Inv_Qty','WC_Rep_Date']]
rep = rep.loc[rep.Rep_Qty>0]
rep.to_csv('../Output/'+date+'/Replenishment_W_('+date+').csv',index= False)
'''