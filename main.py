from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from cvzone.PoseModule import PoseDetector
from starlette.websockets import WebSocketState
import cv2
import numpy as np
import time
import uvicorn
import base64
import logging
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Pushup Counter API is running!"}

class PushupTracker:
    def __init__(self):
        self.detector = PoseDetector()
        self.pushup_count = 0
        self.set_count = 0
        self.pushups_in_current_set = 0
        self.pushup_position = "up"
        self.threshold_angle = 90
        self.min_time_between_pushups = 1.5
        self.last_pushup_time = 0
        self.feedback = ""
        # Calibration timer variables
        self.start_time = time.time()
        self.calibration_done = False

    def calculate_angle(self, a, b, c):
        try:
            angle = np.arctan2(c[2] - b[2], c[1] - b[1]) - np.arctan2(a[2] - b[2], a[1] - b[1])
            angle = np.abs(angle * 180.0 / np.pi)
            return angle if angle <= 180 else 360 - angle
        except Exception as e:
            logger.error(f"Angle error: {str(e)}")
            return 90

    async def process_frame(self, img):
        try:
            img = self.detector.findPose(img, draw=False)
            lmList, _ = self.detector.findPosition(img, bboxWithHands=False)
            self.feedback = ""

            result = {
                "pushup_count": self.pushup_count,
                "set_count": self.set_count,
                "feedback": self.feedback,
                "landmarks": [],
                "calibration_remaining": 0
            }

            current_time = time.time()

            if not self.calibration_done:
                elapsed = current_time - self.start_time
                if elapsed < 5:
                    remaining = 5 - elapsed
                    result["calibration_remaining"] = int(remaining)
                    result["feedback"] = f"Adjust position: {int(remaining)}s"
                    if lmList:
                        result["landmarks"] = [[lm[1], lm[2]] for lm in
                            [lmList[11], lmList[12], lmList[13], lmList[14], lmList[23], lmList[24]]]
                    return result
                else:
                    self.calibration_done = True
                    result["calibration_remaining"] = 0
                    result["feedback"] = "Start doing push-ups!"

            if lmList:
                try:
                    shoulder_left = lmList[11]
                    shoulder_right = lmList[12]
                    elbow_left = lmList[13]
                    elbow_right = lmList[14]
                    hip_left = lmList[23]
                    hip_right = lmList[24]

                    left_angle = self.calculate_angle(shoulder_left, elbow_left, hip_left)
                    right_angle = self.calculate_angle(shoulder_right, elbow_right, hip_right)

                    if left_angle < self.threshold_angle and right_angle < self.threshold_angle:
                        if self.pushup_position == "up" and (current_time - self.last_pushup_time) > self.min_time_between_pushups:
                            self.pushup_count += 1
                            self.pushups_in_current_set += 1
                            self.pushup_position = "down"
                            self.last_pushup_time = current_time
                            self.feedback = "Good pushup!"
                            if self.pushups_in_current_set >= 12:
                                self.set_count += 1
                                self.pushups_in_current_set = 0
                                self.feedback = "Set complete! Rest now."
                    elif left_angle > self.threshold_angle and right_angle > self.threshold_angle:
                        self.pushup_position = "up"

                    result["landmarks"] = [[lm[1], lm[2]] for lm in
                        [shoulder_left, shoulder_right, elbow_left, elbow_right, hip_left, hip_right]]
                    result["feedback"] = self.feedback

                except IndexError:
                    logger.warning("Missing pose landmarks")

            return result

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    tracker = PushupTracker()
    last_processed = 0

    try:
        while True:
            try:
                data = await websocket.receive_text()

                if not data.startswith("data:image/"):
                    continue

                header, encoded = data.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

                if img is None or time.time() - last_processed < 0.1:
                    continue

                result = await tracker.process_frame(img)
                await websocket.send_text(json.dumps(result))
                last_processed = time.time()

            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps({"error": "Processing failed"}))

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
            logger.info("Connection closed cleanly")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)