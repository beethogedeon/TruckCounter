from os.path import join
import json
import aiofiles.tempfile
import os
from loguru import logger
from sys import stderr
from cv2 import IMREAD_UNCHANGED, imdecode, imencode
from numpy import frombuffer, uint8
from fastapi import FastAPI, Request, UploadFile, File, WebSocket, status, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from util import bytes_to_image, TruckDetector

# logger

logger.remove()
logger.add(
    stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

# FastAPI Setup

# title
app = FastAPI(
    title="Truck Detection and Counting",
    version="0.0.1",
)


@app.on_event("startup")
def save_openapi_json():
    """This function is used to save the OpenAPI documentation
    data of the FastAPI application to a JSON file.
    The purpose of saving the OpenAPI documentation data is to have
    a permanent and offline record of the API specification,
    which can be used for documentation purposes or
    to generate client libraries. It is not necessarily needed,
    but can be helpful in certain scenarios."""
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)


# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    """
    It basically sends a GET request to the route & hopes to get a "200"
    response code. Failing to return a 200 response code just enables
    the GitHub Actions to roll back to the last version the project was
    found in a "working condition". It acts as a last line of defense in
    case something goes south.
    Additionally, it also returns a JSON response in the form of:
    {
        'healtcheck': 'Everything OK!'
    }
    """
    return {'healthcheck': 'Everything OK!'}


class DetectRequest(BaseModel):
    source: UploadFile


@app.get("/", include_in_schema=False, tags=["General"])
async def index(request: Request):
    return RedirectResponse(join(request.url.path, "docs"))


@app.post("/detect_from_image", tags=["Detection"])
async def run(image: bytes = File(...)):
    request_image = bytes_to_image(image)
    request_image.save("request_image.jpg")
    truck_detector = TruckDetector("request_image.jpg")

    detected_image, nb_truck = truck_detector()

    return {
        "nb_truck": nb_truck
    }


@app.post("detect_from_video", tags=["Detections"])
async def run(video: UploadFile):
    try:
        async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as temp:
            try:
                contents = await video.read()
                await temp.write(contents)
            except Exception as e:
                return {"message": f"Error about uploaded video because {e}"}
            finally:
                await video.close()

        truck_detector = TruckDetector(temp.name)
        nb_trucks = await run_in_threadpool(truck_detector)
    except Exception as e:
        return {"message": f"Error while detecting because {e}"}
    finally:
        os.remove(temp.name)

    return {"nb_trucks": nb_trucks}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # listen for connections
    await websocket.accept()
    # count = 1
    try:

        truck_detector = TruckDetector()
        while True:
            contents = await websocket.receive_bytes()
            arr = frombuffer(contents, uint8)
            frame = imdecode(arr, IMREAD_UNCHANGED)

            annotated_frame = truck_detector.callback(frame)

            ret, buffer = imencode('.jpg', annotated_frame)

            await websocket.send_json(
                json.dumps({"truck_number": truck_detector.line_zone.in_count + truck_detector.line_zone.out_count}))
            await websocket.send_bytes(buffer.tobytes())
    except WebSocketDisconnect:
        print("Client disconnected")
