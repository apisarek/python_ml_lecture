
# coding: utf-8

# In[1]:

import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

from glob import glob

pd.set_option('display.max_columns', 50)
get_ipython().magic('matplotlib inline')


# In[2]:

paths = glob('../data/*.csv')


# In[4]:

data = pd.concat([pd.read_csv(path) for path in paths])


# In[5]:

data


# In[6]:

def extract_price(text):
    price, price_per_sqm, *_ = text.split('zł')
    price = re.sub(r',', '.', price)
    price = re.sub(r'[^0-9.]', '', price)
    
    price_per_sqm = re.sub(r',', '.', price_per_sqm)
    price_per_sqm = re.sub(r'[^0-9.]', '', price_per_sqm)
    return float(price), float(price_per_sqm)

def extract_area(text):
    area = re.sub(r',', '.', text)
    area = re.sub(r'[^0-9.]', '', area)
    return float(area)

def extract_rooms(text):
    rooms = re.sub(r',', '.', text)
    rooms = re.sub(r'[^0-9.]', '', rooms)
    return int(rooms)

def extract_floor(text):
    if not re.search(r'\d', text):
        return None, None
    text = re.sub(r'parter', '1', text)
    text = re.sub(r'poddasze', '0', text)
    floor, number_of_floors = None, None
    if re.search(r'piętro', text):
        if re.search(r'\(z \d+\)', text):
#             try:
            floor, number_of_floors = text.split('z')
#             except:
#                 print("ERROR", text)
            number_of_floors = re.sub(r'[^0-9.]', '', number_of_floors)            
        else:
            floor = text        
        floor = re.sub(r'[^0-9.]', '', floor)
    else:
        number_of_floors = re.sub(r'[^0-9.]', '', text)

    
    floor = int(number_of_floors) if floor == '0' and number_of_floors is not None else floor
    floor = int(floor) if floor is not None else None
    number_of_floors = int(number_of_floors) if number_of_floors is not None else None
        
    return floor, number_of_floors

def extract_main(main_list):
    raw_price = main_list[0]
    raw_area = main_list[1]
    raw_rooms = main_list[2]
    raw_floor = main_list[3]
    
    return [
        *extract_price(raw_price),
        extract_area(raw_area),
        extract_rooms(raw_rooms),
        *extract_floor(raw_floor)
    ]

def extract_sub(text):
    def split_and_strip(text):
        key, value = text.split(':')
        return key.strip(), value.strip()
    return dict(split_and_strip(x) for x in text)

def extract_dotted(text):
    return {x.strip(): True for x in text}


# In[7]:

def prepare_lat(data):
    result = pd.DataFrame(data['lat']).reset_index(drop=True)
    result.columns = ['lat']
    return result

def prepare_lon(data):
    result = pd.DataFrame(data['lon']).reset_index(drop=True)
    result.columns = ['lon']
    return result

def prepare_main(data):
    main_columns = ['price', 'price_per_sqm', 'area', 'rooms', 'floor', 'number_of_floors']
    result = pd.DataFrame(data['main'].map(extract_main).tolist(), columns=main_columns).reset_index(drop=True)
    result.columns = main_columns
    return result

def prepare_sub(data):
    pre_data = pd.DataFrame(data['sub'].map(eval).map(extract_sub).tolist())
    columns = pre_data.columns
    result = pre_data.reset_index(drop=True)
    result.columns = columns
    return result

def prepare_dotted(data):
    pre_data = pd.DataFrame(data['dotted'].map(eval).map(extract_dotted).tolist())
    columns = pre_data.columns
    result = pre_data.reset_index(drop=True)
    result.columns = columns
    return result


# In[8]:

def prepare_data(data):
    data.coords = data.coords.map(eval)
    data.place = data.place.map(eval)
    data.main = data.main.map(eval)
    data['lat'] = data.coords.map(lambda x: float(x[0][0]))
    data['lon'] = data.coords.map(lambda x: float(x[1][0]))

    main_columns = ['price', 'price_per_sqm', 'area', 'rooms', 'floor', 'number_of_floors']
    dfs = [
        prepare_lat(data),
        prepare_lon(data),
        prepare_sub(data),
        prepare_dotted(data),
        prepare_main(data)
    ]
    final_data = pd.concat(dfs, axis=1)
    final_data['rok budowy']  = final_data['rok budowy'].apply(np.float64)
    final_data['rooms']  = final_data['rooms'].apply(np.float64)
    return final_data


# In[9]:

data = prepare_data(data)


# In[10]:

data.columns


# In[19]:

data


# # Simple analysis

# In[11]:

plt.scatter(data.lat.values, data.lon.values)
plt.show()


# In[12]:

plt.hist(data.price.values, bins=50)
plt.show()


# In[13]:

plt.hist(data.area.values, bins=100)
plt.show()


# In[14]:

plt.hist(data.rooms.values, bins=100)
plt.show()


# In[15]:

plt.scatter(data.area.values, data.price.values)
plt.xlim(0, 300)
plt.ylim(0, 2250000)
plt.show()


# # Simple analysis

# In[16]:

import xgboost as xgb
# from sklearn.linear_model import LinearRegression


# In[17]:

model = xgb.XGBRegressor()
# model = LinearRegression()


# In[18]:

mask = np.random.random(len(data)) < 0.8


# In[19]:

columns_of_interest = ['rynek', 'area', 'lat', 'lon', 'rok budowy', 'rooms', 'floor', 'number_of_floors']


# In[20]:

train_data = pd.get_dummies(data[columns_of_interest + ['price']].loc[mask]).fillna(0)
test_data = pd.get_dummies(data[columns_of_interest + ['price']].loc[~mask]).fillna(0)


# In[22]:

len(train_data)


# In[23]:

len(test_data)


# In[26]:

train_X = train_data[['rynek_pierwotny', 'area', 'lat', 'lon', 'rok budowy', 'rooms', 'floor', 'number_of_floors']]
train_Y = train_data['price']


# In[27]:

model.fit(train_X, train_Y)


# In[28]:

test_X = test_data[['rynek_pierwotny', 'area', 'lat', 'lon', 'rok budowy', 'rooms', 'floor', 'number_of_floors']]
test_Y = test_data['price']


# In[32]:

error = model.predict(test_X) - test_Y 


# In[39]:

np.abs(error.values).mean()


# In[33]:

test_data['price_predicted'] = model.predict(test_data[['rynek_pierwotny', 'area', 'lat', 'lon', 'rok budowy', 'rooms', 'floor', 'number_of_floors']])


# In[ ]:

test_data.price_predicted = test_data.price_predicted.map(np.round)


# In[40]:

test_data['price_diff'] = test_data.price_predicted - test_data.price


# In[42]:

test_data['price_diff_rel'] = test_data.price_diff / test_data.price


# In[43]:

test_data[['area', 'price', 'price_predicted', 'price_diff', 'price_diff_rel']]


# In[ ]:

np.abs(test_data.price_diff_rel)


# In[56]:

plt.hist(error, bins=100)
plt.show()


# In[57]:

plt.scatter(train_data.area.values, train_data.price.values)
plt.scatter(test_data.area.values, test_data.price.values)
plt.xlim(0, 300)
plt.ylim(0, 2250000)
plt.show()


# In[30]:

data


# In[1]:

get_ipython().system('jupyter nbconvert --to script preprocessing.ipynb')


# # Histogram, binned groupby

# In[19]:

plt.hist(data.area, 20)


# In[28]:

bins = np.array([20, 30, 50, 80, 110])
groups = data.groupby(pd.cut(data.area, bins))
hist = groups.count().area.values 
hist = hist / len(data) * 100
hist


# In[33]:

bins = np.linspace(data.area.min(), 200, 6)
groups = data.groupby(pd.cut(data.area, bins))
hist = groups.count().area.values 
hist = hist / len(data) * 100
bins = np.around(bins, decimals=2)
hist = np.around(hist, decimals=2)
result = dict(zip(bins, np.append(hist, None)))


# In[34]:

result


# In[26]:

a = dict(zip(bins, np.append(hist, None)))


# In[19]:

def get_hist_price(nb_bins):
    bins = np.linspace(data.price_per_sqm.min(), 10000, nb_bins)
    groups = data.groupby(pd.cut(data.price_per_sqm, bins))
    hist = groups.count().area.values
    hist = hist / len(data) * 100
    bins = np.around(bins, decimals=2)
    hist = np.around(hist, decimals=2)
    result = dict(zip(bins, np.append(hist, None)))
    return result


# In[20]:

get_hist_price(5)


# # Putting into database

# In[20]:

import json
from pymongo import MongoClient


host = 'localhost'
port = 30017
db_name = 'test3'
collection_name = 'otodom_offers'

client = MongoClient(host, port)
db = client[db_name]
collection = db[collection_name]


# In[31]:

data.columns = [x.replace('.', '_') for x in data.columns]
records = json.loads(data.T.to_json()).values()


# In[32]:

collection.insert(records)


# In[34]:

len(list(collection.find()))


# In[18]:

def get_nearest_neighbors_mean_price_per_sqm(data=data, lat, lon, nb_nearest=20):
    distances = (data.lat - current_lat)**2 + (data.lon - current_lon)**2
    sorted_distances_top = distances.sort_values()[:nb_nearest]
    mean_result = data.ix[sorted_result_top.index]['price_per_sqm'].mean()
    return mean_result


# In[20]:

current_lon = data.lon[300]
current_lat = data.lat[300]


# In[21]:

current_lon, current_lat


# In[23]:




# In[36]:




# In[37]:




# In[38]:

mean_result


# In[ ]:



