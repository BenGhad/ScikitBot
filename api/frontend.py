# API/frontend.py
from fastapi import FastAPI, HTTPException


app = FastAPI(title="Scikit Stock Bot API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Scikit Stock Bot API"}
