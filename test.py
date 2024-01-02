from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

# run this file using uvicorn main:app --reload

# how to run it?
# uvicorn main:app --reload
# where should I run uvicorn main:app --reload

