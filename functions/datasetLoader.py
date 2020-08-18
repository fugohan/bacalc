
# coding: utf-8

# # datasetLoader 
# 
# Diese Funktion ist dazu da eine Dataset auszuwählen und ein pandas Dataframe zurück liefert. 

# packages: 

# In[9]:


import pandas as pd 
import numpy as np
from datetime import datetime 
import requests as r

# In[40]:

def datasetLoader(name=""):
    """ Gibt Daten aus dem Dataset zurück. 
    Vorhandene Datensets: 
    - Litecoin
    - OMX
    - btc_usd
    """
    if "litecoin.csv" in name: 
        def parser(x):
            return datetime.strptime(x,'%Y-%m-%d %H:%M:%S %Z')
        try:
            data = pd.read_csv(name,index_col=0, parse_dates=[0], usecols=["snapped_at","price"] ,date_parser=parser)
            return data
        except:
            print("Path Error")
            return

#     elif "OMX" in name:
#         def parser(x):
#             return datetime.strptime(x,'%Y-%m-%d')
#         try:
#             omx = pd.read_csv("dataset/omxs30.csv",index_col=0, parse_dates=[0], usecols=["Date","Closingprice"] 
#             ,date_parser=parser, sep=";", skiprows=1)
# #             yahoo = pd.read_csv("dataset/OMXyahoo.csv",index_col=0, parse_dates=[0], usecols=["Date","Close"],date_parser=parser, sep=",")
#             now = int(pd.Timestamp.now("utc").value/10**9)
#             old_data = int(pd.Timestamp("1986-09-30").value/10**9) 
#             link = "https://query1.finance.yahoo.com/v7/finance/download/%5EOMX?period1=" + str(old_data) + "&period2="+ str(now) +"&interval=1d&events=history"
#             f = open("dataset/omx30.yahoo.csv", "w")
#             try:
#                 data = r.get(link)
#             except:
#                 for i in range(60):
#                     data = r.get(link)
#                     time.sleep(1)
#                     if data.status_code is 200:
#                         break
#                     else: 
#                         continue
                    
#             f.write(data.text)
#             f.close()
#             yahoo = pd.read_csv("dataset/omx30.yahoo.csv", index_col=0, parse_dates=[0], usecols=["Date","Close"],date_parser=parser, sep=",")
#             omx.Closingprice = omx.Closingprice.replace({',':''}, regex=True)
#             omx.Closingprice = omx.Closingprice.astype(float)
#             omx = omx.loc[(omx!=0).any(axis=1)]
#             compare = omx.append(yahoo, ignore_index=False)
# #             omx.plot()
# #             yahoo.plot()
#         except:
#             print("DatasetErrors: \n check if dataset is loaded")
#             return
# # #         omx = omx.iloc[::-1]
# #         print(omx)
# #         print(yahoo)
# #         #compare = omx 
# #         compare.columns = ['Close', "Closingprice"]
# #         print(compare.columns)
# #         #test = pd.concat(omx.index, yahoo)
# #         print(compare)
        
# #         compare["price"] = compare.pop("Close").fillna(compare.pop("Closingprice")).astype(float)
# # #         compare = compare.iloc[::-1]
# # #         return compare.price.values[::-1]
# #         print(compare)
# #         compare["price"] = compare["price"].iloc[::-1]
#         compare = yahoo.join(omx, on=["Date"])
#         compare = compare.append(omx, ignore_index=False)
#         print(compare)
#         return compare
    elif "omx-test2" in name: 
        def parser(x):      
            return datetime.strptime(x,'%Y-%m-%d')
        yahoo = pd.read_csv("dataset/omx30.yahoo.csv", index_col=0, parse_dates=[0], usecols=["Date","Close"],date_parser=parser, sep=",")
        return yahoo
    elif "omx_newer" in name: 
        def parser(x):      
            return datetime.strptime(x,'%Y-%m-%d')
        now = int(pd.Timestamp.now("utc").value/10**9)
        old_data = int(pd.Timestamp("2020-07-21").value/10**9) 
        link = "https://query1.finance.yahoo.com/v7/finance/download/%5EOMX?period1=" + str(old_data) + "&period2="+ str(now) +"&interval=1d&events=history"
        f = open("dataset/omx30.yahoo.csv", "w")
        try:
            data = r.get(link)
        except:
            for i in range(60):
                data = r.get(link)
                time.sleep(1)
                if data.status_code is 200:
                    break
                else: 
                    continue
        f.write(data.text)
        f.close()
        yahoo = pd.read_csv("dataset/omx30.yahoo.csv", index_col=0, parse_dates=[0], usecols=["Date","Close"],date_parser=parser, sep=",")
        return yahoo
    elif "omx_old" in name:
        def parser(x):
            return datetime.strptime(x,'%Y-%m-%d')
        omx = pd.read_csv("dataset/omxs30.csv",index_col=0, parse_dates=[0], usecols=["Date","Closingprice"] ,date_parser=parser, sep=";", skiprows=1)
        omx.Closingprice = omx.Closingprice.replace({',':''}, regex=True)
        omx.Closingprice = omx.Closingprice.astype(float)
        omx = omx.loc[(omx!=0).any(axis=1)]
        omx = omx.iloc[::-1]
        return omx    
    elif "OMX" in name:
        a =datasetLoader("omx_old")
        b =datasetLoader("omx_newer")
        a["price"] = a.Closingprice
        b["price"] = b.Close
        d = a.append(b)
        d = d.drop(d.columns[[0,2]], 1)
#         d= d.drop('Closeingprice ',1)
#         d = d.price
        return d
    elif "btc_usd" in name:
        url = "https://www.coingecko.com/price_charts/export/1/usd.csv"
        btc_csv=r.get(url)
        f = open("dataset/btc_usd.csv", "w")
        f.write(btc_csv.text)
        f.close()
        def parser(x):
            return datetime.strptime(x,'%Y-%m-%d %H:%M:%S %Z')
        try:
            data = pd.read_csv("dataset/btc_usd.csv",index_col=0, parse_dates=[0], usecols=["snapped_at","price"] ,date_parser=parser)
            return data
        except:
            print("Path Error")
            return
        
    elif "EUR_USD---" in name:
        try:  
            df = pd.read_csv("dataset/EUR_USD.csv", decimal=",")
            return df.reindex(index=df.index[::-1]).Zuletzt.values
        except:
            print("Path Error")
            return
  
    ### Laststatement 
    elif "" in name:
        print("Select a correct Dataset")


##############################################################################        
def minmaxindex(data):
    """ returnvalue = dataminindex, datamin, datamaxindex, datamax"""
    datamaxindex = np.where(data ==np.amax(data))[0][0]
    dataminindex = np.where(data ==np.amin(data))[0][0]
    datamin = np.amin(data)
    datamax = np.amax(data)
    return dataminindex, datamin,datamaxindex, datamax
