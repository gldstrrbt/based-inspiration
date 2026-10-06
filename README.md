# Basedspiration

An experimental archive of Lil B's inspirational, funny, strange, and unexpectedly sincere language.

The long-term idea was larger than a quote website: **build a corpus of Lil B's public quotes across as many sources as possible, then explore training a language model on that body of text.** A lot of his most memorable messages were never collected in searchable text, especially the captions and statements embedded directly in his music videos.

This repository preserves the computer-vision side of that experiment.

## Recovered prototype

A separate browser prototype survives with **328 quote entries** extracted from / associated with Lil B video footage. It randomly surfaces passages from the archive as a tiny on-demand "Based inspiration" machine.

That prototype represents one slice of the intended dataset: rare video-only material. The broader all-source quote corpus and model-training phase remained a future direction rather than a completed system.

## Video archive workflow

The original workflow was built to locate text-heavy moments across a large collection of Lil B videos:

1. Download the Lil B video archive from YouTube.
2. Scan video frames with OpenCV and an EAST text-detection model.
3. Use frame changes and stricter confidence thresholds to reduce false positives.
4. Export candidate frames containing on-screen text.
5. Preserve the resulting passages as source material for the quote archive.

The point was less "OCR demo" and more **media archaeology**: recovering language that otherwise existed only for a few seconds inside hundreds of videos.

## Requirements

- `youtube-dl`
- `Pillow`
- `opencv-python`
- EAST text-detection model

## Run

```bash
youtube-dl -ciw -f "mp4" https://www.youtube.com/user/lilbpack1
python3 text_detection_video.py
```

Text detection across the full video collection is intentionally slow and produces false positives from shapes, lighting, and other visual forms that resemble text.

## What I changed from the reference implementation

The video text detector began from Adrian Rosebrock's excellent PyImageSearch EAST text-detection tutorial. I modified the workflow for this archive, including frame-change handling, higher prediction-confidence thresholds to reduce false positives, and exporting candidate frames for later review.

## Broader concept

The intended next step was to expand beyond video overlays into other documented Lil B quotes and statements, normalize them into a larger dataset, preserve source/provenance where possible, and eventually experiment with a model trained on the resulting corpus.

That broader model was **the direction, not something I completed or shipped**.

## Credits

Credit to [Adrian Rosebrock](https://www.pyimagesearch.com/author/adrian/) / [PyImageSearch](https://www.pyimagesearch.com/) for the original EAST text-detection tutorial that the detection code was adapted from.
