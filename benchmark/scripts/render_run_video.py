"""Render AndroidWorld run trajectories into a single MP4 video.

This script reads a run directory containing .pkl.gz episode files created by
AndroidWorld's IncrementalCheckpointer, extracts per-step raw screenshots, and
writes an annotated MP4 for quick visual inspection.
"""

from __future__ import annotations

import argparse
import os
import textwrap

import cv2
import numpy as np

from android_world import checkpointer
from android_world import constants


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render AndroidWorld run directory to MP4."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to run directory containing task .pkl.gz files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output MP4 file path.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second for output video.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Maximum number of frames to write (0 means all).",
    )
    parser.add_argument(
        "--task_contains",
        default="",
        help="Only include tasks whose name contains this substring.",
    )
    return parser.parse_args()


def _iter_frames(
    episodes: list[dict], task_contains: str
) -> list[tuple[np.ndarray, str, int, float]]:
    frames: list[tuple[np.ndarray, str, int, float]] = []
    filt = task_contains.lower().strip()

    for episode in episodes:
        task_name = str(episode.get(constants.EpisodeConstants.TASK_TEMPLATE, ""))
        if filt and filt not in task_name.lower():
            continue

        success = float(episode.get(constants.EpisodeConstants.IS_SUCCESSFUL, 0.0))
        step_data = episode.get(constants.EpisodeConstants.EPISODE_DATA, {})
        if not isinstance(step_data, dict):
            exc = str(episode.get(constants.EpisodeConstants.EXCEPTION_INFO, ""))
            frames.append((_make_failure_card(task_name, exc), task_name, 1, success))
            continue
        screenshots = step_data.get("raw_screenshot", [])

        for i, img in enumerate(screenshots, start=1):
            if not isinstance(img, np.ndarray):
                continue
            frames.append((img, task_name, i, success))

    return frames


def _make_failure_card(task_name: str, exception_text: str) -> np.ndarray:
    card = np.zeros((2400, 1080, 3), dtype=np.uint8)
    cv2.rectangle(card, (0, 0), (1080, 2400), (20, 20, 20), thickness=-1)

    cv2.putText(
        card,
        "NO SCREENSHOT FRAMES",
        (40, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 180, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        card,
        f"task: {task_name}",
        (40, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    summary = exception_text.replace("\n", " ").strip()
    if not summary:
        summary = "No exception text available."

    y = 280
    for line in textwrap.wrap(summary, width=62)[:18]:
        cv2.putText(
            card,
            line,
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        y += 56

    return card


def _annotate_frame(frame: np.ndarray, task_name: str, step_idx: int, success: float) -> np.ndarray:
    annotated = frame.copy()
    status = "SUCCESS" if success > 0.5 else "FAIL"
    text = f"{task_name} | step={step_idx} | {status}"

    cv2.rectangle(annotated, (16, 16), (1060, 110), (0, 0, 0), thickness=-1)
    cv2.putText(
        annotated,
        text,
        (32, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def main() -> int:
    args = _parse_args()

    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    out_path = os.path.abspath(os.path.expanduser(args.output))

    if not os.path.isdir(run_dir):
        print(f"Run directory not found: {run_dir}")
        return 2

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cp = checkpointer.IncrementalCheckpointer(run_dir)
    episodes = cp.load()
    if not episodes:
        print(f"No episodes found in: {run_dir}")
        return 3

    frames = _iter_frames(episodes, task_contains=args.task_contains)
    if not frames:
        print("No screenshot frames found with the current filters.")
        return 4

    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    h, w, _ = frames[0][0].shape
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (w, h),
    )

    for img, task_name, step_idx, success in frames:
        frame = _annotate_frame(img, task_name, step_idx, success)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Wrote {out_path} with {len(frames)} frames at {args.fps} FPS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
