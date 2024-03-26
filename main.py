from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
from routers.predict_route import pred_route
from config_db.database import Base, engine


app = FastAPI()
app.title = 'Credit Machine Learning'
app.version = '0.0.1'


Base.metadata.create_all(bind=engine)
app.include_router(pred_route)

@app.get('/', tags=['Home'])
def Home():
    return  HTMLResponse('<h1>Welcome to the Credit ML API</h1')


