from fastapi import FastAPI
from handler import (Handler, Request, Response)

app = FastAPI()
handler = Handler()


@ app.post("/model/")
async def inference(request: Request):
    response = handler(request)
    return response
