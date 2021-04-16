
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
db = os.getenv('DATABASE_URL')

# line was deleted? maybe Erik put the DB_URL directly .. maybe similar to df3 = pd.read_sql()... or
# df3 = pd.read_sql(‘data’, con=engine)

df3 = df3.rename({'Crime Rate (per 1000 residents)': 'Crime Rate per 1000'}, axis=1)
df3.columns

engine = create_engine(db)

df3.to_sql('data', con=db, if_exists='replace') #last param if loading to the DB, not if trying to pull form it

df3['Good Days']

livability = df3[['City', 'State', 'Rent', 'Good Days', 'Crime Rate per 1000']] # new df called livability

livability['Rent'] = livability['Rent'] * -1

livability['Crime Rate per 1000'] = livability['Crime Rate per 1000'] * -1

livability


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() # scales the livability features info so data exists btwn range of 0-1

scaler.fit(livability.drop(['City', 'State'], 1))

scaler.transform(livability.drop(['City', 'State'], 1))


import pickle

with open('../app/livability_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('../app/livability_scaler.pkl', 'rb') as f:
    s = pickle.load(f)

s.transform(livability.drop(['City', 'State'], 1).iloc[0,:].to_numpy().reshape(1, -1))

'''
From Ike Mar 22, 2021 9:45PM in Lambda School Workspace labspt_ds channel, pinned by Jeffrey Asuncion:

Come up with a way to calculate a (composite score) livability score for each locale : the codebase already has a livability score and you're expected to either recalculate based on the new features you'll add or come with a completely new way to calculate it (livability score is a composite score)
'''

# import math, numpy

# new_feats_liv_score = (weathr * weathr_wt) + (avail_jobs * avail_jobs_wt)  (+...) # new livability Score includes legacy features + new

# rankd_liv_score = 

# pd.read_sql('data', con=engine)

# import sqlalchemy

# engine = sqlalchemy.create_engine(DATABASE_URL)

# connection = engine.connect()