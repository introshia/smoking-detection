"""Compatibility shim providing `get_model()` expected by some packages.

This wraps the installed `inference_sdk.InferenceHTTPClient` so imports
like `from inference import get_model` resolve for static analysis and at
runtime.
"""
from inference_sdk import InferenceHTTPClient
import base64
import cv2

class _ModelWrapper:
    def __init__(self, client: InferenceHTTPClient, model_id: str):
        self._client = client
        self._model_id = model_id

    def infer(self, frame):
        """Accepts a BGR numpy frame or a base64-encoded JPEG string.

        Returns the raw dict result from `InferenceHTTPClient.infer()`.
        """
        if isinstance(frame, str):
            b64 = frame
        else:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf).decode('utf-8')
        return self._client.infer(b64, model_id=self._model_id)

def get_model(model_id: str, api_key: str = None, api_url: str = "https://serverless.roboflow.com"):
    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    return _ModelWrapper(client, model_id)
