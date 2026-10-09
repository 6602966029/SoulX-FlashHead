# Full-body talking Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task by task. Preserve the existing local Windows setup. User authorized implementation, actual video testing, a new branch and pushing it; unresolved upstream blockers may be filed as issues.

**Goal:** Preserve the entire reference photograph while animating only the mouth region and exporting synchronized MP4.

**Architecture:** Prepare a fixed portrait canvas and reversible square face crop. Generate head chunks with the existing 512×512 pipeline, align their facial anchors, and composite a feathered mouth patch into the canvas. Encode incrementally so full-size frames do not accumulate.

**Tech Stack:** Python 3.11, NumPy, Pillow, OpenCV 4.11, MediaPipe 0.10.9, existing FlashHead / Gradio, FFmpeg.

**Spec:** `docs/superpowers/specs/2026-10-09-full-body-talking-design.md`

## Global Constraints

- Windows single GPU; output 1080×1920, 25 fps, model input 512×512.
- Preserve complete source bounds, static body/background and legacy head API.
- One GPU job per process; new mode initially only on the ordinary interface.
- No additional model downloads or changes to model/VAE internals.
- RGB uint8 frame contracts; report preparation/alignment/export failures.
- No commit of model weights, generated media, local API secrets or user photos.

## Review Focus

- Border faces and EXIF rotation must have reversible geometry, without stretching.
- Missing/multiple faces and missing landmarks must fail before generating the wrong region.
- Chunk transitions must preserve alignment state and must not duplicate frames/audio.
- Alignment/export failure must not publish a partial successful MP4.
- Existing head generation and setup must remain usable; bound full-size frame memory.

## Task 1: Reversible preparation

Files: `flash_head/portrait/{__init__,types,preparation}.py`; `tests/test_portrait.py`.

Interface: `fit_canvas(image, output_size) -> (RGBFrame, affine)`; `square_crop(image, bbox, size=512) -> (RGBFrame, affine)`; `prepare_portrait(image_path, work_dir, options) -> PortraitContext`.

- [x] Write tests asserting fit keeps both top/bottom markers, crop inverse retains reference coordinates, and invalid face counts fail.
- [x] Run `python -m unittest discover -s tests -v`; observe missing behavior fail.
- [x] Implement metadata, source orientation, strict face validation and resource ownership.
- [x] Run tests; commit task files only.

## Task 2: Stable local composition

Files: `flash_head/portrait/{alignment,compositor}.py`; `tests/test_portrait.py`.

Interface: `estimate_alignment(source_anchors, target_anchors, face_width) -> affine`; `MouthCompositor(context).render(head_rgb) -> RGBFrame`; compositor owns per-task mesh/tracking state and `close()`.

- [x] Add tests for known translation, excessive motion rejection, unchanged outside-mask pixels, and three-frame hold/fourth-frame failure.
- [x] Observe RED, implement anchors, similarity transform, fixed safe mask, feathering and bounded local color correction.
- [x] Verify tests and commit.

## Task 3: Incremental audio/video export

Files: `flash_head/services/{__init__,video_export}.py`; `tests/test_video_export.py`.

Interface: `export_video(frames, audio_path, output_path, fps) -> output_path`; consume iterator once, encode no more than ceil(original_duration*fps), fail on too few frames; atomic final publish.

- [x] Test a real tiny video with longer/shorter audio, ffprobe duration and expected failure cleanup.
- [x] Observe RED; implement incremental writer, FFmpeg AAC/faststart and cleanup.
- [x] Verify tests and commit.

## Task 4: Model and UI integration

Files: `flash_head/services/portrait_generation.py`, `gradio_app.py`, README.

Interface: `generate_portrait(..., progress_callback) -> str`; use a task-local context and composer with the existing model and chunk slicing; use shared generation lock around both old and new entry points.

- [x] Add boundary tests that short audio is padded for inference but export uses original duration; old API input count remains unchanged.
- [x] Implement new ordinary single-GPU endpoint/button and CPU preparation previews; preserve `dispatch_inference`.
- [x] Run the full suite and syntax checks; commit after actual smoke below.

## Task 5: Actual effect and final review

- [x] Generate 5–10 seconds from the user's existing full-body photograph with Lite and supplied example audio; run in a separate temporary localhost test service if current UI cannot be restarted safely.
- [x] Inspect exported frames and ffprobe; compare uncompressed pixels outside the mask, report actual alignment metrics and performance.
- [x] Run old head endpoint regression; test longer generation if short-video quality passes.
- [x] Obtain fresh code review; fix important findings with reproduction tests; verify suite.
- [x] Commit only feature/setup sources necessary to reproduce this Windows project, then push `full-body-talking`. If upstream permissions prevent pushing, report concrete blocker and use authorized fork when possible.

## Execution rulings

- Execute inline in the current checkout/new feature branch, as explicitly selected by the user. Reuse installed dependencies and ignored model/runtime files.
- User explicitly requested design followed by implementation and actual tests; proceed continuously without another approval round for this plan.
- Use a Windows-compatible local ledger under ignored `runtime/portrait-development/`, instead of invoking plugin Bash bookkeeping scripts.
- Keep ordinary head generation internals unchanged in the first patch; share the existing audio/model helpers in the new service rather than broad refactoring. This avoids changing legacy chunk timing while testing full-body composition.

Execution note: task changes are grouped into one reviewed commit after the complete suite and real video checks, rather than separate per-task commits. All implementation and verification steps above are complete; branch publication is the final step.
