# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd


# %%
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os


# %%
load_dotenv()
db = os.getenv('DATABASE_URL')


# %%
df3 = df3.rename({'Crime Rate (per 1000 residents)': 'Crime Rate per 1000'}, axis=1)
df3.columns


# %%
engine = create_engine(db)


# %%
df3.to_sql('data', con=db, if_exists='replace')


# %%
df3['Good Days']


# %%
livability = df3[['City', 'State', 'Rent', 'Good Days', 'Crime Rate per 1000']]


# %%
livability['Rent'] = livability['Rent'] * -1


# %%
livability['Crime Rate per 1000'] = livability['Crime Rate per 1000'] * -1


# %%
livability


# %%
from sklearn.preprocessing import MinMaxScaler


# %%
scaler = MinMaxScaler()


# %%
scaler.fit(livability.drop(['City', 'State'], 1))


# %%
scaler.transform(livability.drop(['City', 'State'], 1))


# %%
import pickle


# %%
with open('../app/livability_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# %%
with open('../app/livability_scaler.pkl', 'rb') as f:
    s = pickle.load(f)


# %%
s.transform(livability.drop(['City', 'State'], 1).iloc[0,:].to_numpy().reshape(1, -1))


# %%



