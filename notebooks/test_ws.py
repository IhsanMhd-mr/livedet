import asyncio
import websockets
import base64
import json
import cv2
import numpy as np

async def test_ws():
    # create a dummy black image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    
    async with websockets.connect("ws://localhost:8765") as ws:
        print("Connected.")
        await ws.send(b64_str)
        response = await ws.recv()
        print("Response:", response)

asyncio.run(test_ws())
