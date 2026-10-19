# TrackID3x3 Baseline Review: Gaps on the Challenge Videos and Improvement Plan

> **TL;DR.** The tutorial baseline (YOLO11x + ByteTrack + EasyOCR + HSV K-Means) is
> tuned for a 4K clip from the *training* camera (TI-HOTA 0.635 there). It will break
> on the Challenge videos (IMG_0114/0115: 720p, a different low oblique camera).
> The biggest risk is the **team-anchor rule**: if team A/B comes out flipped, the score
> is **0**. The fix with the highest gain is a bib-colour team classifier trained on the
> Training set. Each team wears a fixed bib colour in every video.

**Deadline:** results due **Nov 3, 2026, 23:59 AoE**. Paper + code due Nov 10.
**Source analysis:** `CHALLENGE_ANALYSIS.md`, `analysis/DATA_ANALYSIS.md`, and `analysis/analyze_training.py` (repo `challenge2`, branch `claude/epic-curie-zcog0t`).

---

## 1. Context: Training data vs Challenge videos

| | Training (IMG_0104–0111) | Challenge (IMG_0114, IMG_0115) |
|---|---|---|
| Resolution | 3840×2160, 29.97 fps | **1280×720**, 29.97 fps |
| Frames | ~10.7k–13.2k per video | 11,536 / 11,823 |
| Camera | 5.2 m high, 17 m away, 17.7° down | **4.3 m high, 21 m away, 11.8° down, off to the side** |
| Player height in image | ~272 px | **~87 px** |
| Court error per 1 px foot error | ~2.2 cm | **~9.8 cm** (p90 12.5 cm) |
| Labels | bbox + court position + team + jersey | none (4 court corners + roster only) |

Challenge rosters:
- **IMG_0114:** A = {6, 9, 10}, B = {6, 9, 11}. **#6 and #9 exist in both teams.**
- **IMG_0115:** A = {4, 10, 12}, B = {1, 3, 5}.

---

## 2. Data findings that change the design

1. **GT court position = homography(bbox bottom-centre)** from the training camera. This is
   exact on all 8 training videos (max error 0.001 cm). The challenge GT comes from the same
   top-view camera, so the GT convention is "bottom-centre as seen by the training camera".
2. **4 teams, fixed bib colours, the same 12 players every game:**

   | Team (jerseys) | Bib | Challenge |
   |---|---|---|
   | {4, 10, 12} | purple | IMG_0115 → A |
   | {6, 9, 11} | green | IMG_0114 → B |
   | {6, 9, 10} | yellow | IMG_0114 → A |
   | {1, 3, 5} | pink | IMG_0115 → B |

3. **Off-court people wear bibs too.** Waiting teams stand next to the court in the
   non-playing colours.
4. **All frames are scored, including pre-game** (game starts at frame 90–232). Before the
   game starts, players stand in 3 separate groups.
5. **No clamping to the court.** 1.2 % of GT positions are outside the court, with extremes
   up to ~12 m outside.
6. **Proximity:** in 39 % of in-game frames an *opponent* pair is within 1 m (man-to-man
   defence). Same-team pairs within 1 m occur in only 3 % of frames.
7. **GT noise:** occasional ~38 cm spikes along the training camera's viewing ray (jumps).
   A perfectly smoothed tracker still scores 0.995, so **smoothing is safe**.

---

## 3. Metric sensitivity (TI-HOTA, τ = 100 cm)

Measured by perturbing the IMG_0104 ground truth and scoring it with the official logic.

| Perturbation | TI-HOTA |
|---|---|
| **Team A ↔ B flipped** | **0.000** |
| Foot-point noise in the 720p challenge view: 1 / 2 / 3 / 5 px | 0.973 / 0.864 / 0.731 / 0.511 |
| Court noise: 10 / 20 / 30 cm | 0.943 / 0.755 / 0.554 |
| Rows missing: 10 % / 20 % / 30 % | 0.900 / 0.804 / 0.700 |
| Same-team ID swap in 10 % / 20 % of frames | 0.937 / 0.882 |

> **Rules of thumb:** a wrong identity costs FP + FN, while a missing one costs only FN.
> Never output the same (team, jersey) twice in a frame. Output at most 6 rows per frame.

---

## 4. Module-by-module review

| Module | What the tutorial does | Risk on the Challenge videos | Severity | Fix |
|---|---|---|---|---|
| **A1 Detection** | COCO YOLO11x, person class | Small players (~87 px); spectators and waiting teams detected | Medium | Fine-tune on training boxes downscaled to 720p, with perspective augmentation |
| **A2 Tracking** | ByteTrack, frame-by-frame | Opponents are often within 1 m, which causes ID switches and fragmentation | Medium | Offline tracklet stitching. Constraint: exactly 6 players. |
| **A3 Localization** | 4-corner homography, bbox bottom-centre | 1 px = ~10 cm; 4 far-away corners give an imprecise H | **High** | Add manual landmarks (FT line, arc, lane, hoop base); check lens distortion |
| **A4 Filtering** | ≥ 10 consecutive in-court frames, side-zone rules, 300 cm buffer | Waiting teams in bibs stand inside the buffer | Medium | Also reject bibs that are not in the two roster colours |
| **B1 Sampling** | 40 crops per track, torso crop | Torso is only ~25×40 px at 720p | Low | Sample more frames; use the tracklet-level median |
| **B2 Jersey OCR** | EasyOCR on 2× upscaled crops | Digits are a few px tall, so OCR mostly fails. "1" vs "10/12" confusion in IMG_0115. | **High** | Use a ReID / person classifier (12 people, trained on the Training set) instead of, or alongside, OCR |
| **B3 Team ID** | K-Means (K = 2) on HSV; A = player nearest (752, 1105) in the **first frame** | Frame 1 is pre-game, so the offense anchor is unreliable and **a flip scores 0** | **Critical** | Bib-colour classifier trained on the Training set + official roster → A/B |
| **B4 Identity** | jersey + team, roster check | Assumes the 6 jerseys are unique; false for IMG_0114 (#6, #9 in both teams) | **High** | Decide the team first, then choose the jersey only from that team's roster |
| **C1 Merging** | Group by identity, keep the highest-conf row; **raise an error if an identity is missing** | If OCR misses one player, there is no submission at all | **Critical** | Never crash: output the identities you have; use a global Hungarian assignment of tracklets to the 6 identities |
| **C2 Reconstruction** | Linear interpolation between the first and last observation only | No rows in pre-game or end frames, so FN | High | Hold the first/last position up to frame 1 / the last frame; add temporal smoothing |
| **C3 Output** | 5-column CSV | Header and frame indexing must match the official format | Low | Headerless file `IMG_0114.txt`, original frame IDs starting at 1, cm |

---

## 5. Discrepancy catalogue (updated)

Items marked **(new)** are additions to the original list.

1. **Player detection:** missed detection, false detection, inaccurate bbox, occlusion
   failure. **(new)** Small-object detection at 720p. **(new)** Bib-wearing spectators and
   waiting teams.
2. **Multi-object tracking:** fragmentation, ID switch, contamination, tracking loss, wrong
   association. **(new)** Frequent close opponent pairs in man-to-man defence.
3. **Court localization:** homography error, perspective distortion, inaccurate foot
   position. **(new)** ~10 cm per pixel in the challenge view. **(new)** GT convention
   mismatch: the GT comes from a different (training) camera, with radial jump spikes.
4. **Court and track filtering:** false track removal, invalid detection retention, short
   track rejection. **(new)** Clamping to the court would hurt, because players do leave it.
5. **Jersey number recognition:** OCR failure, incorrect prediction, partial digits
   ("1" from "10/12"), high-confidence false reads. **(new)** Digits are too small to read
   at 720p.
6. **Team identification:** colour contamination, background, lighting. **(new)** The
   first-frame offense anchor is wrong in pre-game frames, and a flip scores 0.
7. **Player identity assignment:** missing identity, conflicting identity (1, 11, A11, B11).
   **(new)** Duplicate jerseys across teams (IMG_0114). **(new)** The code crashes when an
   identity is missing.
8. **(new) Trajectory coverage:** pre-game and end frames are not predicted, but they are scored.
9. **(new) Domain shift:** no training data is used anywhere in the baseline.

---

## 6. Prioritised action plan

| # | Task | Fixes | Expected gain | Owner | Status |
|---|---|---|---|---|---|
| 1 | Bib-colour team classifier (trained on Training set) + roster → A/B | B3, B4 | Removes the 0-score risk | | ☐ |
| 2 | Robust output: no crash, hold at start/end, ≤ 1 row per identity per frame | C1, C2 | Avoids FN and a missing submission | | ☐ |
| 3 | Refine challenge-camera calibration with extra manual landmarks | A3 | Lower localisation error everywhere | | ☐ |
| 4 | Fine-tune the detector on 720p-downscaled training data; foot-point refinement; temporal smoothing | A1, A3 | Each px of foot error ≈ 10 cm | | ☐ |
| 5 | 12-person ReID / classifier from Training crops; global tracklet → identity assignment | A2, B2 | Fewer ID swaps, no OCR dependency | | ☐ |
| 6 | Local validation: run on training videos downscaled to 720p with the official TI-HOTA | all | Measurable progress | | ☐ |

---

## 7. Rules compliance (from the organisers' clarification email)

- Train and fine-tune **only on the official Training set** (IMG_0104–0111). Other
  TrackID3x3 data (Indoor, Drone, and public GT of the challenge games) is **not allowed**.
- General pretrained models (COCO, OCR, ReID) are OK. **Disclose the checkpoints and their
  sources.**
- No self-training, pseudo-labels, or fine-tuning on the Challenge videos.
- Manual court-landmark selection is OK. **No manual player annotation or identity
  assignment.** The colour → team mapping must therefore be *learned from the Training set*,
  not hand-coded per video.
- Preserve the exact code, weights, and config used for the Nov 3 submission.

---

## 8. Open questions for the organisers

Contact: trackid3x3-challenge-2026@g.sp.m.is.nagoya-u.ac.jp

- Are the two videos' scores averaged, or are the frames pooled?
- How is the top-view GT time-aligned to the challenge video frames? Does the GT cover every
  frame, and do predictions on frames without GT count as FP?
- Is a bib-colour / person classifier trained on the Training set and applied automatically
  allowed? (We read the rules as yes.)

---

## 9. References

- Yamada et al., *TrackID3x3*, MMSports '25, DOI 10.1145/3728423.3759400 (must be cited)
- Challenge site: https://sites.google.com/view/abc2027-trackid3x3-challenge/the-challenge
- Official code and TI-HOTA: https://github.com/open-starlab/TrackID3x3
- Figures (heatmaps, cm-per-pixel, metric sensitivity, bib colours): `analysis/figures/` in the repo
