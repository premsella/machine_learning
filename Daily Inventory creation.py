# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:48:18 2019

@author: prems
"""
import pandas as pd

directory = 'C:\\Users\\prems\\Downloads\\Wyng\\Replenishment(13-05-19)\\Replenishment\\Processed\\BaseFiles\\'

Product_Master = pd.read_parquet(directory + 'Product_Master_Aurelia_updated.parquet')
Store_Master = pd.read_parquet('file:///C:/Users/prems/Downloads/Wyng/Replenishment(13-05-19)/Replenishment/Processed/BaseFiles/Store_Master_Aurelia.parquet')
Size_Ratio = pd.read_parquet('file:///C:/Users/prems/Downloads/Wyng/Replenishment(13-05-19)/Replenishment/Processed/BaseFiles/Size_Ratio_Aurelia_17Jun19.parquet')
Size_Ratio.Store_Code.nunique()
Size_Ratio.AG.unique()


#Store_Master = pd.read_parquet(directory + 'Store_Master_W.parquet')
Daily_Inventory_Material = pd.read_parquet('file:///C:/Users/prems/Downloads/Wyng/Replenishment(13-05-19)/Replenishment/CalcValues/AureliaCalcInv-SS19.parquet')
Daily_Inventory_Material.Date.describe()
extra = pd.read_parquet('file:///C:/Users/prems/Downloads/Wyng/Replenishment(13-05-19)/Replenishment/CalcValues/AureliaCalcInv-SS19_17Jun19.parquet')
extra.Date.describe()
Daily_Inventory_Material = Daily_Inventory_Material.loc[Daily_Inventory_Material.Date<pd.to_datetime('27-May-2019')]
Daily_Inventory_Material.shape[0]+extra.shape[0]
Daily_Inventory_Material = Daily_Inventory_Material.append(extra,sort = False,ignore_index = True)
#Daily_Inventory_Material = Daily_Inventory_Material.loc[~Daily_Inventory_Material.Store_Code.isin(['W451','W632'])]
Daily_Inventory_Material.Date.describe()
(Daily_Inventory_Material.Date.max() - Daily_Inventory_Material.Date.min()).days
Daily_Inventory_Material.Store_Code.nunique()
Daily_Inventory_Material.Material.nunique()
Daily_Inventory_Material.groupby('Date')['CalcStockQty'].sum().plot()
Daily_Inventory_Material.to_parquet('../CalcValues/Aurelia_Daily_Inventory_Material_(01-Jan-2019 to 16-Jun-2019).parquet')
Daily_Inventory_Material = Daily_Inventory_Material.loc[Daily_Inventory_Material.Material.isin(Product_Master.Material.unique())]

#Daily_Inventory_Material = Daily_Inventory_Material.drop('Actual_StockQty',axis=1)
#Daily_Inventory_Material = Daily_Inventory_Material.drop('delta',axis=1)
Daily_Inventory_Material = Daily_Inventory_Material.merge(Product_Master[['Material','AG','Sizing','Product Group','Count_Sizes','Size','Style_Code']],on = ['Material'],how = 'left')
Daily_Inventory_Material = Daily_Inventory_Material.merge(Size_Ratio[['Store_Code','AG','Sizing','Product Group','Size','Size_Ratio']],on = ['Store_Code','AG','Sizing','Product Group','Size'],how = 'left')
Daily_Inventory_Material = Daily_Inventory_Material.dropna() #5.7% dropped, maybe some new sizes that were not there in historical data
Daily_Inventory_Material['Avl'] = 1
Daily_Inventory_Material.loc[Daily_Inventory_Material['CalcStockQty'] == 0,'Avl'] = 0
Daily_Inventory_StyleCode = Daily_Inventory_Material.groupby(['Date','Style_Code','Store_Code','AG','Sizing','Count_Sizes'])['Avl'].sum().reset_index()
Daily_Inventory_StyleCode['SAVL'] = Daily_Inventory_StyleCode['Avl']/Daily_Inventory_StyleCode['Count_Sizes']
Daily_Inventory_Material.loc[Daily_Inventory_Material.Avl == 0 , 'Size_Ratio'] = 0
Daily_Inventory_StyleCode['WAVL'] = Daily_Inventory_Material.groupby(['Date','Style_Code','Store_Code','AG','Sizing'])['Size_Ratio'].unique().apply(lambda x : x.sum()).values
(Daily_Inventory_StyleCode.WAVL > 1.001).sum()

Daily_Inventory_StyleCode.groupby(['Date'])[['SAVL', 'WAVL']].sum().plot(figsize=(12, 6))
Daily_Inventory_StyleCode.shape
Daily_Inventory_StyleCode.nunique()
Daily_Inventory_StyleCode.to_parquet('C:/Users/prems/Downloads/Wyng/Replenishment(13-05-19)/Replenishment/CalcValues/Aurelia_Daily_Inventory_StyleCode_(01-Jan-2019 to 16-Jun-2019).parquet')