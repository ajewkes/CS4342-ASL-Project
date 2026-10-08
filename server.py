import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

import imageImport
import train

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "index.html")

DEFAULTS = {"mode": "train", "lr": 0.001, "epochs": train.EPOCHS, "batchSize": 64,
            "datasetPower": imageImport.DATASET_POWER, "evalEvery": 500, "k": 3, "workers": 8}
LIMITS = {"lr": (1e-6, 1), "epochs": (1, 100), "batchSize": (8, 512), "datasetPower": (0, 1),
          "evalEvery": (10, 5000), "k": (2, 10), "workers": (0, 16)}
WHOLE = {"epochs", "batchSize", "evalEvery", "k", "workers"}

# request body limits, images get more room than settings
MAX_BODY = 10000
MAX_IMAGE_BODY = 3000000

# one training job at a time, shared with the request threads
lock = threading.RLock()
stopEvent = threading.Event()
job = {"state": "idle", "params": None, "log": [], "history": [], "error": None, "started": None, "finished": None,
       "progress": None, "trainStarted": None}
result = {"scores": []}

# model.pt contents, reloaded only when the file changes
modelLock = threading.RLock()
cache = {"mtime": None, "saved": None, "model": None}


def parse(body):
    # checks the request body and fills in defaults
    params = dict(DEFAULTS)
    if body.get("mode", "train") not in ("train", "kfold"):
        raise ValueError("mode must be 'train' or 'kfold'")
    params["mode"] = body.get("mode", "train")
    for key, (low, high) in LIMITS.items():
        if key not in body:
            continue
        value = body[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")
        if key in WHOLE:
            if value != int(value):
                raise ValueError(f"{key} must be a whole number")
            value = int(value)
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        params[key] = value
    return params


def log(line):
    with lock:
        job["log"].append(str(line))
        del job["log"][:-500]


def onEval(entry):
    with lock:
        job["history"].append(entry)


def onProgress(step, total, fold=0):
    with lock:
        if job["trainStarted"] is None:
            job["trainStarted"] = time.time()
        job["progress"] = {"step": step, "total": total, "fold": fold}


def progress(params, now):
    # how far the run is, from 0 to 1 over all folds, and a rough estimate of the seconds left
    p = job["progress"]
    if not p:
        return None
    folds = params["k"] if params["mode"] == "kfold" else 1
    fraction = (p["fold"] + p["step"] / p["total"]) / folds
    perEpoch = p["total"] / params["epochs"]
    spent = now - job["trainStarted"]
    return {"fraction": fraction, "epoch": min(params["epochs"], p["step"] // perEpoch + 1), "epochs": params["epochs"],
            "fold": p["fold"] + 1, "folds": folds,
            # wait for a little progress before guessing, the first steps include loader start-up
            "remaining": spent * (1 - fraction) / fraction if fraction > 0.01 else None}


def run(params):
    state, error = "done", None
    try:
        if params["mode"] == "train":
            trainLoader, validLoader, _ = imageImport.load_data(params["batchSize"], params["workers"], params["datasetPower"])
            train.train(trainLoader, validLoader, lr=params["lr"], epochs=params["epochs"],
                        evalEvery=params["evalEvery"], log=log, stop=stopEvent, onEval=onEval, onProgress=onProgress)
        else:
            scores = train.kfold(k=params["k"], lr=params["lr"], epochs=params["epochs"],
                                 batchSize=params["batchSize"], datasetPower=params["datasetPower"],
                                 workers=params["workers"], log=log, stop=stopEvent, onEval=onEval, onProgress=onProgress)
            with lock:
                result["scores"] = scores
        if stopEvent.is_set():
            state = "stopped"
    except Exception as e:
        state, error = "error", f"{type(e).__name__}: {e}"
        log(f"Error: {error}")
    with lock:
        job.update(state=state, error=error, finished=time.time())


def start(params):
    with lock:
        if job["state"] in ("running", "stopping"):
            return False
        stopEvent.clear()
        result["scores"] = []
        job.update(state="running", params=params, log=[], history=[], error=None, started=time.time(), finished=None,
                   progress=None, trainStarted=None)
    threading.Thread(target=run, args=(params,), daemon=True).start()
    return True


def status():
    with lock:
        params = job["params"]
        end = job["finished"] or time.time()
        return {"state": job["state"], "params": params, "error": job["error"],
                "elapsed": (end - job["started"]) if job["started"] else 0,
                "progress": progress(params, end) if params else None,
                "log": job["log"][-200:], "history": list(job["history"]), "scores": list(result["scores"])}


def savedCheckpoint():
    # model.pt contents, or None when it is missing or unreadable
    with modelLock:
        if not os.path.isfile(train.CHECKPOINT):
            return None
        mtime = os.path.getmtime(train.CHECKPOINT)
        if cache["mtime"] != mtime:
            try:
                saved = train.readCheckpoint(train.CHECKPOINT)
            except Exception:
                # can be half written while a run is saving, try again next time
                return None
            cache.update(mtime=mtime, saved=saved, model=None)
        return cache["saved"]


def predictBlocker():
    # why predictions can't run yet, or None once a finished training run has saved a model
    with lock:
        if job["state"] in ("running", "stopping") and job["params"]["mode"] == "train":
            return "Training is still running. Predictions open once it finishes."
    if not os.path.isfile(train.CHECKPOINT):
        return "No trained model yet. Train one first, predictions open once it finishes."
    saved = savedCheckpoint()
    if saved is None:
        return "The saved model can't be read right now. Train again if this doesn't clear up."
    return train.notUsable(saved)


def readyModel():
    # (model, None) when predictions are allowed, otherwise (None, reason)
    with modelLock:
        reason = predictBlocker()
        if reason:
            return None, reason
        if cache["model"] is None:
            cache["model"] = train.fromCheckpoint(cache["saved"], torch.device("cpu"))
        return cache["model"], None


def decodeImage(dataUrl):
    # "data:image/...;base64,..." -> PIL image
    if not isinstance(dataUrl, str) or not dataUrl.startswith("data:image/"):
        raise ValueError("image must be a data:image URL")
    try:
        image = Image.open(io.BytesIO(base64.b64decode(dataUrl.split(",", 1)[1], validate=True)))
        image.load()
    except Exception:
        raise ValueError("that file isn't a readable image")
    return image


def checkpointInfo():
    blocker = predictBlocker()
    if not os.path.isfile(train.CHECKPOINT):
        return {"exists": False, "ready": False, "reason": blocker}
    info = {"exists": True, "ready": blocker is None, "reason": blocker}
    saved = savedCheckpoint()
    if saved is not None:
        info.update(valAcc=saved.get("valAcc"), step=saved.get("step"), lr=saved.get("lr"),
                    finished=saved.get("finished", True), modified=cache["mtime"])
    return info


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, payload, contentType="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", contentType)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def hostOk(self):
        # only answer requests addressed to localhost
        host = self.headers.get("Host", "").rsplit(":", 1)[0]
        if host in ("localhost", "127.0.0.1"):
            return True
        self.reply(403, {"error": "forbidden host"})
        return False

    def do_GET(self):
        if not self.hostOk():
            return
        if self.path in ("/", "/index.html"):
            with open(PAGE, "rb") as f:
                self.reply(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self.reply(200, status())
        elif self.path == "/api/config":
            self.reply(200, {"defaults": DEFAULTS, "limits": LIMITS})
        elif self.path == "/api/checkpoint":
            self.reply(200, checkpointInfo())
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        if not self.hostOk():
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self.reply(415, {"error": "send application/json"})
        limit = MAX_IMAGE_BODY if self.path == "/api/predict" else MAX_BODY
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > limit:
                return self.reply(413, {"error": f"request is too large, the limit is {limit} bytes"})
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except ValueError as e:
            return self.reply(400, {"error": f"bad request body: {e}"})

        if self.path == "/api/train":
            try:
                params = parse(body)
            except ValueError as e:
                return self.reply(400, {"error": str(e)})
            if not start(params):
                return self.reply(409, {"error": "a job is already running"})
            self.reply(202, status())
        elif self.path == "/api/stop":
            with lock:
                if job["state"] != "running":
                    return self.reply(409, {"error": "nothing is running"})
                job["state"] = "stopping"
            stopEvent.set()
            self.reply(202, status())
        elif self.path == "/api/predict":
            model, reason = readyModel()
            if model is None:
                return self.reply(409, {"error": reason})
            try:
                image = decodeImage(body.get("image"))
            except ValueError as e:
                return self.reply(400, {"error": str(e)})
            top = train.predictImage(model, image, torch.device("cpu"))
            self.reply(200, {"predictions": [{"character": c, "confidence": p} for c, p in top]})
        else:
            self.reply(404, {"error": "not found"})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"UI running at http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down")
        stopEvent.set()


if __name__ == "__main__":
    main()
