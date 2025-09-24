# CourtSight

## Overview

CourtSight is an end-to-end computer-vision toolkit that takes a single video of a 5-on-5 basketball game and produces player bounding boxes, team-aware tracking, event counts (passes, steals), per-player metrics (distance, speed, ball acquisition percentage), and a 2D top-down map of player movement and heatmaps. The project combines a fine-tuned YOLO detector, a multi-object tracker, and zero-shot classification/embedding-based retrieval for event detection.

This repository is ideal for researchers and engineers who want to prototype sports-analytics features from broadcast or court-level video without expensive manual annotation for every event type.

---

## Key features

* **Player detection** — YOLO model fine-tuned on labeled basketball player frames.
* **Robust multi-object tracking** — Re-identify players across frames (DeepSort / ByteTrack / other trackers supported).
* **Zero-shot event detection** — Count passes and steals using embedding-based classification (e.g., CLIP-like embeddings) and rule-based heuristics on top of tracked object interactions.
* **Ball acquisition percentage** — Percent of possessions where each player touches or acquires the ball.
* **Distance & speed** — Per-player distance traveled and instantaneous/average speed (in court units) computed from tracked trajectories.
* **2D movement map** — Homography to court coordinates producing trajectory plots, density heatmaps, and team tactical visualizations.

---

## Architecture / Pipeline

1. **Preprocessing**

   * Video decoding & frame sampling (configurable FPS).
   * Optional camera calibration / court template selection for homography.
2. **Detection**

   * Fine-tuned YOLO model produces bounding boxes for players and ball (if available).
3. **Tracking**

   * Online multi-object tracker assigns persistent IDs and produces trajectories.
4. **Event extraction (zero-shot / heuristic hybrid)**

   * For each short clip or tracked-interaction window, compute embeddings (visual + spatial features) and compare against textual event prompts (e.g., "pass", "steal").
   * Combine with interaction rules (proximity + ball possession changes) to increase precision.
5. **Metrics & visualizations**

   * Compute distance, speed, ball acquisition percentage.
   * Map trajectories to a normalized 2D court and generate heatmaps and animated replay visualizations.
6. **Output**

   * CSV/JSON with per-player stats, annotated video or frames, and visualization images (trajectory plots, heatmaps).

---

## Models & Methods (high level)

* **YOLO (fine-tuning)**: Use Ultralytics YOLOv8 (or preferred YOLO repo) fine-tuned on basketball frames labeled for `player` and optionally `ball`.
* **Tracker**: DeepSort / ByteTrack / FairMOT or similar for ID persistence.
* **Zero-shot classification**: Use an embedding model (CLIP image encoder or video CLIP variants) to embed short clips or frame stacks and compute similarity to textual prompts like `"a player passes the ball"` or `"steal"`. Augment with geometry and possession change heuristics.
* **Ball possession heuristics**: Determine possession by proximity between player bbox and ball bbox + motion/velocity cues. When ball detection is noisy or absent, estimate possession via relative motion of a player’s hand/arm region and the ball trajectory.
* **Homography & court mapping**: Detect court lines / known court landmarks or use manual calibration to compute homography (frame → court coordinates). Map bounding-box feet points to court plane to compute distances in meters/feet.

---

## Inputs & Outputs

**Input:** MP4 / AVI video of a 5v5 basketball game (single static camera recommended; broadcast works but may need calibration)


---

## Quickstart (local)

```bash
# 1) Clone
git clone https://github.com/natek-1/CourtSight.git
cd CourtSight

# 2) Create env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Recommended dependencies

* Python 3.10+
* PyTorch
* Ultralytics YOLO (or YOLOv8)
* OpenCV
* NumPy, pandas, matplotlib
* CLIP or a video embedding model (for zero-shot event detection)
* DeepSort / ByteTrack tracker implementation
* scikit-learn (for evaluation helpers)

See `requirements.txt` for pinned versions.

---

## How pass/steal detection works (short)

1. Track players and (if present) the ball across frames to detect possession changes.
2. For each candidate interaction window where ownership changes or two players interact near the ball, extract a short clip.
3. Compute joint embedding of the clip (visual sequence + simple motion features) and compare to textual prompts using cosine similarity.
4. Combine the embedding score with rule-based checks (e.g., did the ball travel from player A to B in short time window? Did player B make contact?) to produce a final event label and confidence.

This hybrid approach reduces reliance on large supervised action datasets while retaining reasonable accuracy.

---

## Evaluation & Metrics

* **Detection**: Precision / Recall / mAP on player detection.
* **Tracking**: MOTA, IDF1, ID switches.
* **Events**: Precision / Recall / F1 for pass and steal detection against annotated ground truth.
* **Kinematics**: Validate distance/speed by comparing a small annotated subset with court-measured distances (if available).

---

## Outputs & Example

When you run the pipeline the repository will produce images and CSVs in the `outputs/` folder. Example artifacts include `game1_trajectories.png`, `team1_heatmap.png`, `annotated_game1.mp4`.

---

## Roadmap / TODO

* Improve zero-shot detection by integrating a small supervised action recognizer for common actions.
* Improve ball detection using a separate ball-focused detector or optical-flow-based tracker.
* Add support for multi-camera synchronization for full-court 3D reconstruction.
* Add a web UI for interactive playback and stat exploration.

---

## Contributing

Contributions welcome! Please open issues for feature requests or bugs, and submit PRs for fixes and improvements. See `CONTRIBUTING.md` for guidelines.

---

## License

MIT License — see `LICENSE`.

---

## Contact

Project lead / maintainer: *Your Name* — open an issue or PR on GitHub.
