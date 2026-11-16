#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pillow",
#   "torch",
#   "torchvision",  # Sam3ImageProcessor needs it
#   "transformers>=5.3",
# ]
# ///
"""SAM3 object detector for the VLM brain, as a small HTTP server.

The brain (wojtek_nav/vlm_brain_node.py, `detector_url`) sends one colour
JPEG and a short English concept phrase ("purple pillar"); SAM3 returns
every matching instance. Each instance comes back as its score, box and
one *interior* point -- the pixel deepest inside the mask, so the pixel
resolver's depth patch lands on the object and not past its edge (a
mask's centroid can fall outside a thin or curved object).

Runs on the model host, next to Ollama, not in the ROS container (which
has no torch). The weights are gated: request access to facebook/sam3 on
Hugging Face and log in (`hf auth login`) or set HF_TOKEN.

  uv run ros/src/wojtek_nav/scripts/sam3_server.py [--host 127.0.0.1] [--port 8100]
      [--model facebook/sam3] [--device auto|cuda|cpu]

`auto` takes CUDA in bfloat16 when there is a GPU, else the CPU in
float32 (correct, but seconds per image). Then set DETECTOR_URL in the
repo-root .env to http://127.0.0.1:8100 (the same Linux/macOS container
rules as VLM_URL apply; see .env.example).

  POST /detect  {"image": <base64 JPEG/PNG>, "prompt": "purple pillar",
                 "threshold": 0.5}
             -> {"detections": [{"score", "box": [x0, y0, x1, y1],
                 "point": [u, v], "area"}], "width", "height", "latency_s"}
                 box/point normalised to the image size, area the mask's
                 share of the image; most confident first.
  GET  /health -> {"model", "device", "dtype"}

One request at a time (a single model on one device). Plain HTTP with no
authentication: keep it on loopback or behind an SSH tunnel.
"""

import argparse
import base64
import io
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

DEFAULT_MODEL = "facebook/sam3"
DEFAULT_PORT = 8100
MAX_BODY = 20 * 1024 * 1024  # a camera JPEG is ~100 KB; refuse anything absurd
# Erosion passes for the interior point. Each peels one pixel ring; after
# this many the remaining core is deep inside any object worth walking to.
INTERIOR_PASSES = 64


def interior_point(mask, passes=INTERIOR_PASSES):
    """(u, v) normalised: the mask pixel nearest the centroid of its most
    eroded non-empty core. None for an empty mask. Pixels outside the image
    count as background, so an object cut by the frame edge gets a point
    away from that edge."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        return None
    h, w = m.shape
    core = m
    for _ in range(passes):
        e = core.copy()
        e[1:, :] &= core[:-1, :]
        e[:-1, :] &= core[1:, :]
        e[:, 1:] &= core[:, :-1]
        e[:, :-1] &= core[:, 1:]
        e[0, :] = e[-1, :] = False
        e[:, 0] = e[:, -1] = False
        if not e.any():
            break
        core = e
    ys, xs = np.nonzero(core)
    i = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
    return (float(xs[i]) + 0.5) / w, (float(ys[i]) + 0.5) / h


def to_detections(masks, boxes, scores, width, height):
    """Post-processed SAM3 instances (numpy: masks N x H x W, boxes N x 4
    absolute xyxy, scores N) as the JSON-ready list, most confident first.
    An instance with an empty mask has no point and is dropped."""
    out = []
    for mask, box, score in zip(masks, boxes, scores):
        point = interior_point(mask)
        if point is None:
            continue
        x0, y0, x1, y1 = (float(c) for c in box)
        out.append({
            "score": round(float(score), 4),
            "box": [round(x0 / width, 4), round(y0 / height, 4),
                    round(x1 / width, 4), round(y1 / height, 4)],
            "point": [round(point[0], 4), round(point[1], 4)],
            "area": round(float(np.count_nonzero(mask)) / (width * height), 5),
        })
    out.sort(key=lambda d: d["score"], reverse=True)
    return out


def parse_request(body):
    """-> (image bytes, prompt, threshold) or raises ValueError."""
    try:
        req = json.loads(body)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"body is not JSON: {exc}") from exc
    if not isinstance(req, dict):
        raise ValueError("body must be a JSON object")
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    try:
        image = base64.b64decode(req.get("image") or "", validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"image is not base64: {exc}") from exc
    if not image:
        raise ValueError("image is empty")
    threshold = req.get("threshold", 0.5)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a number in [0, 1]")
    return image, prompt.strip(), float(threshold)


class Sam3:
    """The model, loaded once."""

    def __init__(self, name, device):
        import torch
        from transformers import Sam3Model, Sam3Processor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = device
        # bf16 on the GPU (GB10 and any recent NVIDIA card); the CPU's bf16
        # kernels are slow or missing, so fp32 there.
        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.name = name
        self.processor = Sam3Processor.from_pretrained(name)
        self.model = Sam3Model.from_pretrained(name, dtype=self.dtype).to(device).eval()

    def detect(self, image_bytes, prompt, threshold):
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        res = self.processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist())[0]
        masks = res["masks"].to("cpu").numpy().astype(bool)
        boxes = res["boxes"].float().to("cpu").numpy()
        scores = res["scores"].float().to("cpu").numpy()
        return to_detections(masks, boxes, scores, image.width, image.height), image.width, image.height


def make_handler(sam):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 -- http.server's naming
            if self.path.rstrip("/") == "/health":
                self._reply(200, {"model": sam.name, "device": sam.device, "dtype": str(sam.dtype)})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/detect":
                self._reply(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                self._reply(413 if length else 411, {"error": "bad Content-Length"})
                return
            try:
                image, prompt, threshold = parse_request(self.rfile.read(length))
            except ValueError as exc:
                self._reply(400, {"error": str(exc)})
                return
            t0 = time.perf_counter()
            try:
                dets, w, h = sam.detect(image, prompt, threshold)
            except Exception as exc:  # noqa: BLE001 -- a bad image must not kill the server
                self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._reply(200, {"detections": dets, "width": w, "height": h,
                              "latency_s": round(time.perf_counter() - t0, 3)})

        def log_message(self, fmt, *args):
            sys.stderr.write(f"sam3_server: {fmt % args}\n")

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", help="auto, cuda, cuda:N or cpu")
    args = ap.parse_args(argv)
    sam = Sam3(args.model, args.device)
    print(f"sam3_server: {sam.name} on {sam.device} ({sam.dtype}), "
          f"http://{args.host}:{args.port}/detect", flush=True)
    HTTPServer((args.host, args.port), make_handler(sam)).serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
