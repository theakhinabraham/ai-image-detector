# App smoke test (Phase 6)

**When and how:** run on 2026-11-21 against the real server (`python -I -m app.main`, bound to `127.0.0.1:8000`), with uploads sent by `curl` to `POST /predict`. The probabilities below are copied from the server's JSON responses.

## Valid uploads

| Upload | Kind | resnet18_mix | smallcnn |
|---|---|---|---|
| CIFAKE test fake, 32x32 JPEG | in-distribution fake | 0.965, likely AI-generated | 1.000, likely AI-generated |
| COCOAI real photo, 368x640 JPEG | real photo, large, non-square | 0.009, likely real | 0.224, likely real |
| COCOAI Midjourney 6 fake, re-saved as JPEG q25 | newer generator, heavily compressed | 0.927, likely AI-generated | 0.003, **likely real (wrong)** |
| COCOAI DALL·E 3 fake, saved as WebP | newer generator, WebP format | 0.892, likely AI-generated | 0.045, **likely real (wrong)** |

What these show:

- **Matches the findings:** SmallCNN, trained on CIFAKE only, misses both newer-generator fakes. That is the cross-generator failure measured in `findings.md`, section 4.
- **Warnings are correct:**
  - The large images get the "downscaled" warning.
  - The non-square photo gets the "central square" warning.
  - The q25 image gets "heavily JPEG-compressed (estimated quality 25)".
  - Every result names that model's training generators.
- **Latency:** 15 to 64 ms per request on CPU, including Grad-CAM.

**Bug found and fixed:** the first time this test ran, the q25 image did not get the compression warning. `ImageOps.exif_transpose` returns a copy without the JPEG quantisation tables. Quality is now read before the transpose, and a regression test covers it (`tests/test_infer.py::test_heavy_compression_warning_through_full_pipeline`).

## Error handling

| Upload | HTTP status | Message shown to the user |
|---|---|---|
| `bad.txt` (text/plain) | 415 | Please upload a JPEG, PNG or WebP image. |
| 6 MB file | 413 | The image is larger than 5 MB. Please upload a smaller file. |
| Corrupt PNG (valid signature, random bytes) | 422 | The file could not be read as an image. It may be corrupt. |
| GIF sent as `image/png` | 422 | Unsupported image format (GIF). Use JPEG, PNG or WebP. |

## Other checks

- **Network binding:** `lsof` shows the server listening on `127.0.0.1:8000` only.
- **No storage:** no file was written to disk during the test. The repo's git status and `data/` were unchanged.
- **UI:** checked in headless Chrome at desktop width and at 500 px width (see `docs/app_screenshot.png`, which uses an MIT-licensed CIFAKE image).
