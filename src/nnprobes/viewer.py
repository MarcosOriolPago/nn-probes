from __future__ import annotations

import argparse
import csv
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def serve(path: str | Path, port: int = 8050) -> ThreadingHTTPServer:
    """Serve the live viewer for a run folder in a background thread and print its address.

    It stops when the Python process ends; `nnprobes view <path>` keeps it open afterwards.
    """
    server = _start(path, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nnprobes", description="Browse a run folder written by nnprobes.Probes.")
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("view", help="open the live viewer")
    view.add_argument("path", help="the run folder, e.g. runs/exp1")
    view.add_argument("--port", type=int, default=8050)
    args = parser.parse_args(argv)
    server = _start(args.path, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


def _start(path: str | Path, port: int) -> ThreadingHTTPServer:
    for candidate in range(port, port + 20):
        try:
            server = _Server(("127.0.0.1", candidate), _Handler)
            break
        except OSError:
            continue
    else:
        raise OSError(f"no free port in {port}-{port + 19}")
    server.root = Path(path).resolve()
    print(f"nn-probes viewer: http://127.0.0.1:{server.server_address[1]}", flush=True)
    return server


def run_summary(root: Path) -> dict:
    """Everything the viewer lists: scalar history and the steps saved for each figure."""
    scalars: dict[str, dict[str, dict]] = {}
    csv_path = root / "probes.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    value = float(row["value"])
                except (TypeError, ValueError, KeyError):
                    continue
                entry = scalars.setdefault(row["probe"], {}).setdefault(row["metric"], {"steps": {}, "marks": {}})
                step = row["step"]
                if re.fullmatch(r"-?\d+", step):
                    entry["steps"][int(step)] = value
                else:
                    entry["marks"][step] = value
    for metrics in scalars.values():
        for entry in metrics.values():
            steps = sorted(entry.pop("steps").items())
            entry["x"] = [s for s, _ in steps]
            entry["y"] = [v if v == v else None for _, v in steps]

    figures: dict[str, dict[str, list]] = {}
    if root.is_dir():
        for path in root.glob("*/*/*.json"):
            probe, metric = path.parent.parent.name, path.parent.name
            figures.setdefault(probe, {}).setdefault(metric, []).append([path.stem, int(path.stat().st_mtime_ns)])
    for metrics in figures.values():
        for steps in metrics.values():
            steps.sort(key=_step_order)
    return {"root": str(root), "scalars": scalars, "figures": figures}


def _step_order(entry: list) -> tuple:
    label, mtime = entry
    match = re.fullmatch(r"epoch(-?\d+)", label)
    return (0, int(match.group(1)), 0) if match else (1, 0, mtime)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    root: Path

    def handle_error(self, request, client_address):
        pass


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def do_GET(self):
        url = urlparse(self.path)
        root: Path = self.server.root
        if url.path == "/":
            self._send(files("nnprobes").joinpath("viewer.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/run":
            self._send(json.dumps(run_summary(root)).encode(), "application/json")
        elif url.path == "/api/figure":
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            parts = [query.get(k, "") for k in ("probe", "metric", "step")]
            path = (root / parts[0] / parts[1] / f"{parts[2]}.json").resolve()
            if all(p and p != ".." and "/" not in p and "\\" not in p for p in parts) and path.is_file():
                self._send(path.read_bytes(), "application/json")
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass
