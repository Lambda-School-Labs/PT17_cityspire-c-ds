"""Database functions"""

import pandas as pd
import os
from fastapi import APIRouter, Depends
import sqlalchemy
from dotenv import load_dotenv
import databases
import asyncio
from typing import Union, Iterable
from pypika import Query, Table, CustomFunction
from pypika.terms import Field

Field_ = Union[Field, str]

load_dotenv()
database_url = os.getenv("DATABASE_URL")
database = databases.Database(database_url)

# database_url = "postgresql://DBCITYG:FLc3wX793XwzdEK@cityspire-g.c2uishzxxikl.us-east-1.rds.amazonaws.com/postgres"
# database = databases.Database(database_url)

router = APIRouter()

@router.get("/info")
async def get_url():
    """Verify we can connect to the database,
    and return the database URL in this format:

    dialect://user:password@host/dbname

    The password will be hidden with ***
    """

    url_without_password = repr(database.url)
    return {"database_url": url_without_password}


async def select(columns: Union[Iterable[Field_], Field_], city):
    data = Table("data")
    if type(columns) == str or type(columns) == Field:
        q = Query.from_(data).select(columns)
    else:
        cols = [data[x] for x in columns]
        q = Query.from_(data).select(*cols)

    q = q.where(data.City == city.city).where(data.State == city.state)

    value = await database.fetch_one(str(q))
    return value


async def select_all(city):
    """Fetch all data at once

    Fetch data from DB

    args:
        city: selected city

    returns:
        Dictionary that contains the requested data, which is converted
            by fastAPI to a json object.
    """
    data = Table("data")
    di_fn = CustomFunction("ROUND", ["number"])
    columns = (
        # 'lat', 'lon'
        data["lat"].as_("latitude"),
        data["lon"].as_("longitude"),
        data["Crime Rating"].as_("crime"),
        data["Rent"].as_("rental_price"),
        data["Air Quality Index"].as_("air_quality_index"),
        data["Population"].as_("population"),
        data["Nearest"].as_("nearest_string"),
        data["Good Days"].as_("good_days"),
        data["Crime Rate per 1000"].as_("crime_rate_ppt"),
        di_fn(data["Diversity Index"] * 100).as_("diversity_index"),
    )

    q = (
        Query.from_(data)
        .select(*columns)
        .where(data.City == city.city)
        .where(data.State == city.state)
    )
    value = await database.fetch_one(str(q))
    return value


sql = "SELECT * FROM master_jobs_table"
jobs_df = pd.read_sql(sql, database_url)
columns = ["index", "city_state", "title", "company", "salary", "summary"]
jobs_df['metadata'] = jobs_df[columns].to_dict(orient='records')

@router.get("/get_jobs")
async def get_available_jobs_dict(city_state):
    cols = ['title','company','salary','summary', 'metadata']
    avail_jobs = jobs_df.loc[jobs_df['city_state'] == city_state, cols].head(10).to_dict(orient='records')
    return avail_jobs

@router.get("/get_jobs_count")
async def get_jobs_count_dict(city_state):
    cols = ['index']
    jobs_count = jobs_df.loc[jobs_df['city_state'] == city_state, cols].count().to_dict()
    return jobs_count