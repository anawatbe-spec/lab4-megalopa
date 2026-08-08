#!/usr/bin/env python3
"""Run YOLO detection on validation images, validation videos, or a USB webcam."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

SUPPORTED_MODELS = {".pt", ".onnx"}
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEOS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

DEFAULT_DISPLAY_WIDTH = 760
DEFAULT_DISPLAY_HEIGHT = 420


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else root / path


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def safe_stem(text: str, limit: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return (cleaned or "result")[:limit]


def deployment_paths() -> tuple[Path, Path, Path, Path]:
    root = repository_root()
    baseline_model = root / "shared/models/yolo26n.pt"
    custom_model_dir = root / "student_work/models"
    result_root = root / "student_work/results/pi5"
    result_root.mkdir(parents=True, exist_ok=True)
    return root, baseline_model, custom_model_dir, result_root


def discover_models(model_dir: Path) -> list[Path]:
    if not model_dir.is_dir():
        return []

    candidates: list[Path] = []
    for path in sorted(model_dir.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_MODELS:
            candidates.append(path)
        elif path.is_dir() and path.name.endswith("_ncnn_model"):
            candidates.append(path)
    return candidates


def choose_model(
    root: Path,
    baseline_model: Path,
    custom_model_dir: Path,
    explicit: str | None,
    baseline: bool,
) -> tuple[Path, str]:
    if baseline and explicit:
        raise ValueError("Use either --baseline or --model, not both.")

    if baseline:
        if not baseline_model.is_file():
            raise FileNotFoundError(
                f"Baseline model not found: {baseline_model}\n"
                "Synchronize the group fork and confirm shared/models/yolo26n.pt exists."
            )
        return baseline_model, "baseline"

    if explicit:
        path = resolve_path(root, explicit)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return path, "custom"

    models = discover_models(custom_model_dir)
    if not models:
        raise FileNotFoundError(
            f"No .pt, .onnx, or NCNN model found in {custom_model_dir}. "
            "Complete Session 1 and push student_work/models/ to the group fork."
        )

    print("\nAvailable custom models")
    for index, path in enumerate(models, start=1):
        print(f"[{index}] {path.name}")

    while True:
        raw = input(f"Select model [1-{len(models)}]: ").strip()
        try:
            selected = int(raw)
        except ValueError:
            selected = -1

        if 1 <= selected <= len(models):
            return models[selected - 1], "custom"
        print("Invalid selection.")


def choose_indexed_file(
    root: Path,
    source: str | None,
    selected_index: int | None,
    directory: str,
    supported_extensions: set[str],
    media_name: str,
) -> Path:
    if source and selected_index is not None:
        raise ValueError(f"Use either --source or --{media_name}-index, not both.")

    if source:
        path = resolve_path(root, source)
        if not path.is_file():
            raise FileNotFoundError(f"{media_name.title()} not found: {path}")
        if path.suffix.lower() not in supported_extensions:
            raise ValueError(
                f"Unsupported {media_name} extension: {path.suffix}. "
                f"Supported: {sorted(supported_extensions)}"
            )
        return path

    folder = resolve_path(root, directory)
    if not folder.is_dir():
        raise FileNotFoundError(f"Validation {media_name} folder not found: {folder}")

    files = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in supported_extensions
        ),
        key=lambda path: path.name.lower(),
    )

    if not files:
        raise FileNotFoundError(
            f"No supported validation {media_name} files found in {folder}. "
            f"Supported: {sorted(supported_extensions)}"
        )

    print(f"\nAvailable validation {media_name}s")
    for index, path in enumerate(files, start=1):
        print(f"[{index}] {path.name}")

    if selected_index is None:
        while True:
            raw = input(f"Select {media_name} [1-{len(files)}]: ").strip()
            try:
                selected_index = int(raw)
            except ValueError:
                selected_index = -1

            if 1 <= selected_index <= len(files):
                break
            print("Invalid selection.")

    if not 1 <= selected_index <= len(files):
        raise ValueError(f"--{media_name}-index must be between 1 and {len(files)}")

    selected = files[selected_index - 1]
    print(f"Selected {media_name}:", selected.name)
    return selected


def class_counts(result) -> Counter[str]:
    counts: Counter[str] = Counter()
    if result.boxes is None or result.boxes.cls is None:
        return counts

    names = result.names
    for raw_class_id in result.boxes.cls.detach().cpu().tolist():
        class_id = int(raw_class_id)
        if isinstance(names, dict):
            name = str(names.get(class_id, f"class_{class_id}"))
        else:
            name = (
                str(names[class_id]) if class_id < len(names) else f"class_{class_id}"
            )
        counts[name] += 1
    return counts


def annotate(
    result,
    model_name: str,
    model_role: str,
    confidence: float,
    latency_ms: float,
    fps: float,
):
    frame = result.plot()
    counts = class_counts(result)
    total = sum(counts.values())

    count_text = ", ".join(f"{name}={count}" for name, count in counts.items())
    count_lines: list[str] = []
    if count_text:
        while len(count_text) > 58:
            split_at = count_text.rfind(", ", 0, 58)
            split_at = split_at if split_at > 0 else 58
            count_lines.append(count_text[:split_at])
            count_text = count_text[split_at:].lstrip(", ")
        count_lines.append(count_text)
    else:
        count_lines.append("none")

    lines = [
        f"Mode: {model_role}",
        f"Model: {model_name}",
        f"Total detections: {total}",
        *[
            f"Classes: {line}" if index == 0 else f"         {line}"
            for index, line in enumerate(count_lines)
        ],
        f"Confidence: {confidence:.2f}",
        f"Latency: {latency_ms:.1f} ms",
        f"Inference FPS: {fps:.2f}",
        "Q/Esc: quit    S: save frame",
    ]

    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 25

    return frame, total, dict(counts)


def fit_for_display(frame, max_width: int, max_height: int):
    if max_width <= 0 or max_height <= 0:
        raise ValueError("Display width and height must be positive.")

    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)

    if scale >= 1.0:
        return frame

    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


class PreviewWindow:
    """A small, movable OpenCV window that keeps GUI events responsive."""

    def __init__(
        self,
        title: str,
        enabled: bool,
        max_width: int,
        max_height: int,
    ) -> None:
        self.title = title
        self.enabled = enabled
        self.max_width = max_width
        self.max_height = max_height
        self.initialized = False

    def show(self, frame, delay_ms: int = 1) -> int:
        if not self.enabled:
            return -1

        preview = fit_for_display(frame, self.max_width, self.max_height)
        preview_height, preview_width = preview.shape[:2]

        if not self.initialized:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, preview_width, preview_height)
            cv2.moveWindow(self.title, 10, 10)
            self.initialized = True

        cv2.imshow(self.title, preview)
        key = cv2.waitKey(max(1, delay_ms)) & 0xFF

        try:
            if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                return ord("q")
        except cv2.error:
            pass

        return key

    def close(self) -> None:
        if self.enabled and self.initialized:
            try:
                cv2.destroyWindow(self.title)
            except cv2.error:
                pass
        cv2.waitKey(1)


def save_summary(result_dir: Path, name: str, summary: dict) -> Path:
    path = result_dir / name
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved summary:", path)
    return path


def open_camera(index: int):
    capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open USB webcam index {index}")

    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def candidate_camera_indexes(scan_max: int) -> list[int]:
    indexes = set(range(max(0, scan_max) + 1))
    for path in Path("/dev").glob("video*"):
        suffix = path.name.removeprefix("video")
        if suffix.isdigit():
            indexes.add(int(suffix))
    return sorted(indexes)


def scan_cameras(scan_max: int) -> int:
    working: list[int] = []
    print("Scanning camera indexes. This may take several seconds.\n")

    for index in candidate_camera_indexes(scan_max):
        capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            continue

        ok = False
        frame = None
        for _ in range(12):
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                break

        if ok and frame is not None:
            height, width = frame.shape[:2]
            print(f"Camera index {index}: WORKING ({width}x{height})")
            working.append(index)

        capture.release()

    print("\nWorking camera indexes:", working or "NONE")
    if not working:
        print(
            "No camera returned a frame. Check the USB connection, power, "
            "video-group permission, and webcam compatibility."
        )
        return 1
    return 0


def run_image(
    root: Path,
    model: YOLO,
    model_path: Path,
    model_role: str,
    source: Path,
    result_dir: Path,
    conf: float,
    imgsz: int,
    display: bool,
    display_width: int,
    display_height: int,
) -> int:
    started = time.perf_counter()
    result = model.predict(
        source=str(source),
        conf=conf,
        imgsz=imgsz,
        device="cpu",
        verbose=False,
    )[0]

    latency_ms = (time.perf_counter() - started) * 1000.0
    fps = 1000.0 / max(latency_ms, 1e-9)
    frame, total, counts = annotate(
        result,
        model_path.name,
        model_role,
        conf,
        latency_ms,
        fps,
    )

    image_tag = safe_stem(source.stem)
    model_tag = safe_stem(model_path.stem if model_path.is_file() else model_path.name)
    model_format = (
        model_path.suffix.lower().lstrip(".") if model_path.is_file() else "ncnn"
    )

    output = result_dir / (
        f"static_{image_tag}_{model_tag}_{model_format}_detection.jpg"
    )
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Unable to save result image: {output}")

    print("Total detections:", total)
    print("Class counts:", counts)
    print("Latency (ms):", round(latency_ms, 2))
    print("Saved full-resolution result:", output)

    preview = PreviewWindow(
        f"Lab 4 - {model_role.title()} Static Detection",
        enabled=display,
        max_width=display_width,
        max_height=display_height,
    )

    if display:
        print(
            "Preview fitted to the Pi display. "
            "Press Q, Esc, Enter, or Space to close."
        )
        try:
            while True:
                key = preview.show(frame, delay_ms=50)
                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                    13,
                    10,
                    32,
                ):
                    break
        finally:
            preview.close()
    else:
        print("Display disabled; inspect the saved output image.")

    save_summary(
        result_dir,
        f"static_{image_tag}_{model_tag}_{model_format}_summary.json",
        {
            "mode": "image",
            "model_role": model_role,
            "model": relative_or_absolute(model_path, root),
            "model_format": model_format,
            "source": relative_or_absolute(source, root),
            "confidence": conf,
            "imgsz": imgsz,
            "total_detections": total,
            "class_counts": counts,
            "latency_ms": latency_ms,
            "inference_fps": fps,
            "output_image": relative_or_absolute(output, root),
            "preview_enabled": display,
            "preview_max_width": display_width,
            "preview_max_height": display_height,
        },
    )
    return 0


def create_video_writer(
    preferred_path: Path,
    fps: float,
    frame_size: tuple[int, int],
):
    fps = fps if fps > 0 else 25.0
    writer = cv2.VideoWriter(
        str(preferred_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if writer.isOpened():
        return writer, preferred_path

    writer.release()
    fallback_path = preferred_path.with_suffix(".avi")
    writer = cv2.VideoWriter(
        str(fallback_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("Unable to create MP4 or AVI output video.")
    return writer, fallback_path


def run_video(
    root: Path,
    model: YOLO,
    model_path: Path,
    model_role: str,
    source: Path,
    result_dir: Path,
    conf: float,
    imgsz: int,
    display: bool,
    display_width: int,
    display_height: int,
    max_frames: int,
) -> int:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open validation video: {source}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_fps = source_fps if source_fps > 0 else 25.0

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_width <= 0 or source_height <= 0:
        capture.release()
        raise RuntimeError("Validation video reported an invalid frame size.")

    stem = safe_stem(source.stem)
    preferred_output = result_dir / f"video_{stem}_annotated.mp4"
    writer, output_path = create_video_writer(
        preferred_output,
        source_fps,
        (source_width, source_height),
    )

    preview = PreviewWindow(
        f"Lab 4 - {model_role.title()} Validation Video",
        enabled=display,
        max_width=display_width,
        max_height=display_height,
    )

    latencies: deque[float] = deque(maxlen=300)
    frame_metrics: list[dict] = []
    detection_totals: list[int] = []
    frame_count = 0
    saved_count = 0
    last_total = 0
    last_counts: dict[str, int] = {}
    started = time.perf_counter()
    stopped_early = False

    print("Processing validation video.")
    if display:
        print("Press Q/Esc to stop early or S to save the current annotated frame.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            inference_started = time.perf_counter()
            result = model.predict(
                source=frame,
                conf=conf,
                imgsz=imgsz,
                device="cpu",
                verbose=False,
            )[0]

            latency_ms = (time.perf_counter() - inference_started) * 1000.0
            latencies.append(latency_ms)
            average_latency = sum(latencies) / len(latencies)
            inference_fps = 1000.0 / max(average_latency, 1e-9)

            annotated, last_total, last_counts = annotate(
                result,
                model_path.name,
                model_role,
                conf,
                average_latency,
                inference_fps,
            )

            writer.write(annotated)
            frame_count += 1
            detection_totals.append(last_total)
            frame_metrics.append(
                {
                    "frame_index": frame_count,
                    "timestamp_seconds": (frame_count - 1) / source_fps,
                    "total_detections": last_total,
                    "latency_ms": latency_ms,
                    "rolling_inference_fps": inference_fps,
                    "class_counts": json.dumps(last_counts, sort_keys=True),
                }
            )

            key = preview.show(annotated, delay_ms=1)
            if key in (ord("q"), ord("Q"), 27):
                stopped_early = True
                break

            if key in (ord("s"), ord("S")):
                saved_count += 1
                snapshot = result_dir / f"video_{stem}_frame_{frame_count:06d}.jpg"
                if cv2.imwrite(str(snapshot), annotated):
                    print("Saved:", snapshot)

            if max_frames > 0 and frame_count >= max_frames:
                stopped_early = True
                print(f"Stopped at --max-frames={max_frames}.")
                break
    finally:
        capture.release()
        writer.release()
        preview.close()

    elapsed = time.perf_counter() - started
    average_latency = sum(latencies) / len(latencies) if latencies else 0.0
    inference_fps = 1000.0 / average_latency if average_latency else 0.0
    end_to_end_fps = frame_count / elapsed if elapsed > 0 else 0.0

    metrics_path = result_dir / f"video_{stem}_frame_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "frame_index",
            "timestamp_seconds",
            "total_detections",
            "latency_ms",
            "rolling_inference_fps",
            "class_counts",
        ]
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(frame_metrics)

    mean_detections = (
        sum(detection_totals) / len(detection_totals) if detection_totals else 0.0
    )
    min_detections = min(detection_totals) if detection_totals else 0
    max_detections = max(detection_totals) if detection_totals else 0

    summary = {
        "mode": "video",
        "model_role": model_role,
        "model": relative_or_absolute(model_path, root),
        "source": relative_or_absolute(source, root),
        "confidence": conf,
        "imgsz": imgsz,
        "source_fps": source_fps,
        "source_width": source_width,
        "source_height": source_height,
        "frames_processed": frame_count,
        "stopped_early": stopped_early,
        "elapsed_seconds": elapsed,
        "average_latency_ms": average_latency,
        "inference_fps": inference_fps,
        "end_to_end_processing_fps": end_to_end_fps,
        "mean_detections_per_frame": mean_detections,
        "minimum_detections_per_frame": min_detections,
        "maximum_detections_per_frame": max_detections,
        "last_total_detections": last_total,
        "last_class_counts": last_counts,
        "saved_frames": saved_count,
        "output_video": relative_or_absolute(output_path, root),
        "frame_metrics_csv": relative_or_absolute(metrics_path, root),
        "preview_enabled": display,
    }

    save_summary(result_dir, f"video_{stem}_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def run_camera(
    root: Path,
    model: YOLO,
    model_path: Path,
    model_role: str,
    result_dir: Path,
    camera_index: int,
    conf: float,
    imgsz: int,
    display: bool,
    display_width: int,
    display_height: int,
    max_frames: int,
) -> int:
    capture = open_camera(camera_index)
    preview = PreviewWindow(
        f"Lab 4 - {model_role.title()} USB Webcam Detection",
        enabled=display,
        max_width=display_width,
        max_height=display_height,
    )

    latencies: deque[float] = deque(maxlen=60)
    frame_count = 0
    saved_count = 0
    last_total = 0
    last_counts: dict[str, int] = {}
    started = time.perf_counter()

    if display:
        print("USB webcam opened. Press Q/Esc to quit or S to save a frame.")
    else:
        print(
            "USB webcam opened with display disabled. "
            "Use --max-frames to stop automatically."
        )
        if max_frames <= 0:
            raise ValueError(
                "--no-display with --mode camera requires --max-frames greater than 0."
            )

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("USB webcam did not return a frame")

            inference_started = time.perf_counter()
            result = model.predict(
                source=frame,
                conf=conf,
                imgsz=imgsz,
                device="cpu",
                verbose=False,
            )[0]

            latency_ms = (time.perf_counter() - inference_started) * 1000.0
            latencies.append(latency_ms)
            average_latency = sum(latencies) / len(latencies)
            inference_fps = 1000.0 / max(average_latency, 1e-9)

            annotated, last_total, last_counts = annotate(
                result,
                model_path.name,
                model_role,
                conf,
                average_latency,
                inference_fps,
            )

            frame_count += 1
            key = preview.show(annotated, delay_ms=1)

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("s"), ord("S")):
                saved_count += 1
                output = result_dir / f"live_detection_{saved_count:02d}.jpg"
                if cv2.imwrite(str(output), annotated):
                    print("Saved:", output)
                else:
                    print("Unable to save:", output)

            if max_frames > 0 and frame_count >= max_frames:
                print(f"Stopped at --max-frames={max_frames}.")
                break
    finally:
        capture.release()
        preview.close()

    elapsed = time.perf_counter() - started
    average_latency = sum(latencies) / len(latencies) if latencies else 0.0
    inference_fps = 1000.0 / average_latency if average_latency else 0.0
    end_to_end_fps = frame_count / elapsed if elapsed > 0 else 0.0

    summary = {
        "mode": "camera",
        "model_role": model_role,
        "model": relative_or_absolute(model_path, root),
        "camera_index": camera_index,
        "confidence": conf,
        "imgsz": imgsz,
        "frames_processed": frame_count,
        "elapsed_seconds": elapsed,
        "average_latency_ms": average_latency,
        "inference_fps": inference_fps,
        "end_to_end_processing_fps": end_to_end_fps,
        "last_total_detections": last_total,
        "last_class_counts": last_counts,
        "saved_frames": saved_count,
        "result_directory": relative_or_absolute(result_dir, root),
        "preview_enabled": display,
    }

    save_summary(result_dir, "live_performance_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("image", "video", "camera"),
        default="camera",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use shared/models/yolo26n.pt instead of a student custom model",
    )
    parser.add_argument(
        "--model",
        help="Custom model path relative to the repository root or an absolute path",
    )
    parser.add_argument(
        "--source",
        help="Explicit image or video path for --mode image or --mode video",
    )
    parser.add_argument(
        "--image-index",
        type=int,
        help="1-based index in the sorted validation-image list",
    )
    parser.add_argument(
        "--image-dir",
        default="shared/validation/images",
        help="Validation-image folder relative to the repository root",
    )
    parser.add_argument(
        "--video-index",
        type=int,
        help="1-based index in the sorted validation-video list",
    )
    parser.add_argument(
        "--video-dir",
        default="shared/validation/videos",
        help="Validation-video folder relative to the repository root",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--scan-cameras",
        action="store_true",
        help="Scan available /dev/video* indexes and exit without loading a model",
    )
    parser.add_argument(
        "--camera-scan-max",
        type=int,
        default=10,
        help="Also scan integer camera indexes from 0 through this value",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--display-width",
        type=int,
        default=DEFAULT_DISPLAY_WIDTH,
        help="Maximum preview width; saved outputs retain full resolution",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=DEFAULT_DISPLAY_HEIGHT,
        help="Maximum preview height; saved outputs retain full resolution",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open an OpenCV window; outputs are still saved",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop video or camera mode after this many frames; 0 means no limit",
    )
    args = parser.parse_args()

    if args.scan_cameras:
        return scan_cameras(args.camera_scan_max)

    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be in (0, 1]")

    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")

    if args.display_width <= 0 or args.display_height <= 0:
        parser.error("--display-width and --display-height must be positive")

    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")

    if args.baseline and args.model:
        parser.error("Use either --baseline or --model, not both.")

    if args.mode == "camera" and args.source:
        parser.error("--source is only valid for image or video mode")

    if args.mode != "image" and args.image_index is not None:
        parser.error("--image-index is only valid with --mode image")

    if args.mode != "video" and args.video_index is not None:
        parser.error("--video-index is only valid with --mode video")

    root, baseline_model, custom_model_dir, result_root = deployment_paths()

    try:
        model_path, model_role = choose_model(
            root=root,
            baseline_model=baseline_model,
            custom_model_dir=custom_model_dir,
            explicit=args.model,
            baseline=args.baseline,
        )

        result_dir = result_root / model_role
        result_dir.mkdir(parents=True, exist_ok=True)

        print("Repository:", root)
        print("Model role:", model_role)
        print("Selected model:", model_path)
        print("Result folder:", result_dir)

        model = YOLO(str(model_path))
        display = not args.no_display

        if args.mode == "image":
            source = choose_indexed_file(
                root=root,
                source=args.source,
                selected_index=args.image_index,
                directory=args.image_dir,
                supported_extensions=SUPPORTED_IMAGES,
                media_name="image",
            )
            return run_image(
                root=root,
                model=model,
                model_path=model_path,
                model_role=model_role,
                source=source,
                result_dir=result_dir,
                conf=args.confidence,
                imgsz=args.imgsz,
                display=display,
                display_width=args.display_width,
                display_height=args.display_height,
            )

        if args.mode == "video":
            source = choose_indexed_file(
                root=root,
                source=args.source,
                selected_index=args.video_index,
                directory=args.video_dir,
                supported_extensions=SUPPORTED_VIDEOS,
                media_name="video",
            )
            return run_video(
                root=root,
                model=model,
                model_path=model_path,
                model_role=model_role,
                source=source,
                result_dir=result_dir,
                conf=args.confidence,
                imgsz=args.imgsz,
                display=display,
                display_width=args.display_width,
                display_height=args.display_height,
                max_frames=args.max_frames,
            )

        return run_camera(
            root=root,
            model=model,
            model_path=model_path,
            model_role=model_role,
            result_dir=result_dir,
            camera_index=args.camera_index,
            conf=args.confidence,
            imgsz=args.imgsz,
            display=display,
            display_width=args.display_width,
            display_height=args.display_height,
            max_frames=args.max_frames,
        )

    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
