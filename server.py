import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

import imageImport
import train

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "index.html")

DEFAULTS = {"mode": "train", "lr": 0.001, "minutes": 10, "batchSize": 64,
            "datasetPower": imageImport.DATASET_POWER, "evalEvery": 500, "k": 3, "workers": 8}
LIMITS = {"lr": (1e-6, 1), "minutes": (0.1, 600), "batchSize": (8, 512), "datasetPower": (0, 1),
          "evalEvery": (10, 5000), "k": (2, 10), "workers": (0, 16)}
WHOLE = {"batchSize", "evalEvery", "k", "workers"}

# one training job at a time, shared with the request threads
lock = threading.RLock()
stopEvent = threading.Event()
job = {"state": "idle", "params": None, "log": [], "history": [], "error": None, "started": None, "finished": None}
result = {"scores": []}


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


def run(params):
    state, error = "done", None
    try:
        if params["mode"] == "train":
            with lock:
                started = job["started"]
            trainLoader, validLoader, _ = imageImport.load_data(params["batchSize"], params["workers"], params["datasetPower"])
            train.train(trainLoader, validLoader, lr=params["lr"], minutes=params["minutes"],
                        evalEvery=params["evalEvery"], log=log, stop=stopEvent, onEval=onEval, startTime=started)
        else:
            scores = train.kfold(k=params["k"], lr=params["lr"], minutes=params["minutes"],
                                 batchSize=params["batchSize"], datasetPower=params["datasetPower"],
                                 workers=params["workers"], log=log, stop=stopEvent, onEval=onEval)
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
        job.update(state="running", params=params, log=[], history=[], error=None, started=time.time(), finished=None)
    threading.Thread(target=run, args=(params,), daemon=True).start()
    return True


def status():
    with lock:
        params = job["params"]
        end = job["finished"] or time.time()
        folds = params["k"] if params and params["mode"] == "kfold" else 1
        return {"state": job["state"], "params": params, "error": job["error"],
                "elapsed": (end - job["started"]) if job["started"] else 0,
                "budgetSeconds": params["minutes"] * 60 * folds if params else 0,
                "log": job["log"][-200:], "history": list(job["history"]), "scores": list(result["scores"])}


def checkpointInfo():
    if not os.path.isfile(train.CHECKPOINT):
        return {"exists": False}
    try:
        saved = torch.load(train.CHECKPOINT, map_location="cpu")
    except Exception:
        # can be half written while a run is saving
        return {"exists": True}
    return {"exists": True, "valAcc": saved.get("valAcc"), "step": saved.get("step"), "lr": saved.get("lr"),
            "modified": os.path.getmtime(train.CHECKPOINT)}


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
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(min(length, 10000)) or b"{}")
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
