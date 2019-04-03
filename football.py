from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from starlette.datastructures import URL, Address, FormData, Headers, QueryParams
from starlette.formparsers import FormParser, MultiPartParser
from starlette.types import Message, Receive, Scope

from starlette.requests import Request
from starlette.responses import Response
from datetime import datetime


from fastai.vision import (
    ImageDataBunch,
    create_cnn,
    open_image,
    imagenet_stats,
    get_transforms,
    models,
    defaults,
    load_learner
)
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio

#binary sync read
def get_bytes_bin(url):
    import requests
    # some web servers will ignore empty user agent.
    h = {  'User-Agent': 'Neo' }
    r = requests.get(url, headers=h)
    return r.content
    #r.text for text



async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        # some web servers will ignore empty user agent.
        h = {  'User-Agent': 'Neo' }
        try:
            async with session.get(url, headers=h) as response:
                return await response.read()    
        except:
            print("Except get_bytes")
            return b"" #bytes literal of length 0 in case of not 200 reurn 
        


defaults.device = torch.device('cpu')

app = Starlette(debug=False)#True) #set to False later on

learn = load_learner(".", fname="export.pkl")



# Classes
classes = ["american football", "foodball"]



@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
#    print(request.query_params["url"])
    try:

        f=open("the.log", "a")       
        f.write(str(datetime.now()) + "\t" +  request.query_params["url"] + "\n" )
        f.close()
    except:
        return HTMLResponse( "Bad URL." )


    bytes = await get_bytes(request.query_params["url"])
    # bytes = get_bytes_bin(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes): 
    if(len(bytes) == 0):
        return HTMLResponse("Bad image URL")

    buffer = BytesIO()
    buffer.write(bytes)
    buffer.seek(0)
    try:
        img = open_image(buffer)
    except:
        return HTMLResponse("No image")

    pred_class,pred_idx,probab = learn.predict(img)
    # print(pred_class,pred_idx,probab) #f tensor(1) tensor([3.1544e-05, 9.9997e-01])
    return JSONResponse({
        "predictions": sorted(
            zip(classes, map(float, probab)),
            key=lambda p: p[1],
            reverse=True
        )
    })


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <h1>Americaon vs. European football classification</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == '__main__':
    
    uvicorn.run(app, host='0.0.0.0', port=8888 , log_level="info", reload=False )

