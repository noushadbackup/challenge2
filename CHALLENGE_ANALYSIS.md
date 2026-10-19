# 9th ABC Challenge (ABC 2027): TrackID3x3, in-depth analysis

Source: https://sites.google.com/view/abc2027-trackid3x3-challenge (all pages), the
official Drive folder, the tutorial Colab, and https://github.com/open-starlab/TrackID3x3.
Facts marked **(verified)** were checked directly against the downloaded data.

## 1. The task in one line

Given a fixed-camera **3x3 basketball** video, output for **every frame** the
**court position (cm)** and the **identity (team A/B + jersey number)** of each of the
**6 on-court players**.

This is a simplified "Game State Reconstruction": detect, then track, then project onto the court, then identify.

## 2. Data

| Split | Videos | Camera | Resolution / fps | Labels given |
|---|---|---|---|---|
| Training | IMG_0104 … IMG_0111 (8 videos, .MOV) | RKU outdoor court, high "top-ish" fixed view | 3840x2160 HEVC, 29.97 fps, ~11.4k frames each **(verified)** | bbox per frame (MOT), court position + team + jersey per frame, court registration, roster |
| Final challenge | IMG_0114, IMG_0115 (.MP4) | Same RKU court, **different, low oblique viewpoint** | **1280x720** H.264, 29.97 fps, 11 536 / 11 823 frames **(verified)** | court registration (4 corners) and roster only |

The site's "Dataset Structure" section also mentions a separate `test/` split. The site
says elsewhere that there is **no** intermediate Test set, and the Drive folder only has
`Training/` and `Final challenge/`. **One submission, no leaderboard feedback.**

### File formats (verified)

- `Image-space bounding boxes/IMG_xxxx.txt` (MOT): `frame, track_id, x, y, w, h, 1.00, -1, -1`
- `court position/IMG_xxxx.txt`: `frame, court_x, court_y, team(A|B), jersey`
- `court_registration.csv`: `point_id, image_x, image_y, court_x, court_y`, with 4 court corners
  RKU0 (0,0), RKU1 (0,1105), RKU10 (1505,0), RKU11 (1505,1105). The court is **1505 x 1105 cm**
  (15 m x 11 m, the 3x3 half-court). `x` runs along the 15 m side (the hoop is at x ≈ 752).
- `games_roster.csv` (training): `video, game_start_frame, track_id: jersey, offense_track_ids,
  defense_track_ids, team: jersey`
- Final rosters:
  - IMG_0114 → `A: 6, 9, 10; B: 6, 9, 11`. **Both teams have #6 and #9.**
  - IMG_0115 → `A: 4, 10, 12; B: 1, 3, 5`

### Key facts discovered in the data

1. **GT court position = homography(bbox bottom-centre)** using the 4-point registration.
   Reprojecting bbox `(x + w/2, y + h)` with the training homography reproduces the GT
   court position to **0.000 cm median / 0.00 cm max** **(verified, IMG_0104)**. Moving the
   foot point up by just 5 % of the box height already gives ~29 cm error.
2. **Team A = the team on offense at game start** (true for all 8 training rosters; the
   tutorial also says "A = initial Offense"). That is how you decide which jersey colour is A
   in the challenge videos.
3. GT positions are **not clipped** to the court (x from −142 to 1628, y from −474 to 1260
   cm). Do not clamp predictions to the court rectangle.
4. Some training frames have only 5 GT players (80 frames in IMG_0104, a player out of
   view). The GT file has slightly fewer frames than the video (11 376 vs 11 400).
5. All frames are evaluated, **including pre-game frames** (game start ≈ frame 90–232).
   Never trim or renumber frames, and keep the original numbering starting at 1.
6. The final view has very poor ground resolution. At 720p, **1 px vertical error at
   the foot equals ~9 cm at mid-court and up to ~17 cm at the far corner** (the training 4K
   view is ~1–4 cm/px) **(verified via homography Jacobian)**. Foot-point accuracy is the
   main bottleneck.
7. The challenge GT reportedly comes from the **top-view annotations of the same games**
   (the public repo has `IMG_0114/0115` GT, which is **forbidden to use**). So the GT is
   "bottom-centre in another camera". Expect some irreducible cross-view offset, and
   occluded players in the oblique view still need a prediction.

## 3. Submission format

One file per video, `IMG_0114.txt` and `IMG_0115.txt`. CSV, **no header**:

```
frame_id,court_x,court_y,team_id,jersey_number
1,75.48,614.47,A,4
```

- frame_id starts at 1 and follows the original video numbering.
- court_x and court_y are in cm, in the same coordinate system as the training GT.
- team_id is `A` or `B`. (team, jersey) is the identity, so there is no track-id column.
- Expect at most 6 rows per frame (one per roster identity).

Deadlines (AoE, UTC−12):
- **Results**: Nov 3, 2026, via Google Form (link to be announced).
- **Paper (8–12 pages) + code ZIP + weights**: Nov 10, 2026, via CMT.

The code must be the exact version that produced the Nov 3 predictions. At least one
member presents at the workshop. Accepted papers go to IJABC (J-STAGE).
Prize: 100 000 JPY + wearables.

## 4. Metric: TI-HOTA (τ = 100 cm)

Per frame, per GT–pred pair:

- `LocSim = exp(ln(0.05) · d² / τ²)` (1 at d = 0, 0.05 at d = 100 cm)
- `IdSim = 1` only if team **and** jersey both match, else 0
- `Sim = LocSim · IdSim`

For each α ∈ {0.05, 0.10, …, 0.95} (19 values):

1. Run Hungarian matching on `Sim ≥ α`, which gives TP/FP/FN.
2. `DetA = TP / (TP + FP + FN)`.
3. `AssA` = mean over GT identities of `TPA / (TPA + FNA + FPA)`. FPA counts unmatched
   predictions carrying that identity label.
4. `TI-HOTA_α = sqrt(DetA · AssA)`.

The final score is the mean over α.

Max distance that still counts as a match at a given α: `d ≤ 100·sqrt(ln α / ln 0.05)`

| α | 0.05 | 0.25 | 0.50 | 0.75 | 0.95 |
|---|---|---|---|---|---|
| d_max | 100 cm | 68 cm | 48 cm | 31 cm | **13 cm** |

### What the metric rewards

- **A wrong identity is worse than no prediction.** A wrong row costs an FP *and* leaves an
  FN. A missing row costs only an FN.
- Back-of-envelope rule: output a row when P(correct and within threshold) > DetA / (1 + DetA).
  With DetA ≈ 0.6 that is about 0.37. So output reasonably confident guesses, including
  interpolated ones through occlusion.
- Never output the same (team, jersey) twice in a frame. Output at most 6 rows per frame.
- The tracker's own track IDs don't matter. Only a correct identity label per frame counts,
  so ID switches between same-team players are the real failure mode.
- Localisation precision drives the high-α half of the score (≤ 13–31 cm). Smoothing the
  trajectories helps.
- Swapping the team colours, or swapping #6 and #9 between teams in IMG_0114, gives
  catastrophic zeros for those identities.

Tutorial baseline (YOLO11x + ByteTrack + EasyOCR + HSV colour, on a **training** 4K clip):
TI-HOTA 0.635, DetA 0.636, AssA 0.634, median position error 14 cm, team/jersey accuracy ≈ 0.92.
Expect noticeably lower results on the 720p oblique challenge view.

## 5. Rules that constrain the method

- **Training/fine-tuning only on the official Training set (IMG_0104–0111).** Other
  TrackID3x3 data (Indoor, Drone, and the public Outdoor GT for 0112–0115) is **not allowed**.
- General-purpose public pretrained models are OK: COCO detectors, ReID, pose, OCR,
  PARSeq, and so on. You must disclose the checkpoints, their sources, and any data used
  to fine-tune them.
- Challenge videos are for **inference only**. No self-training, no pseudo-labels, no
  fine-tuning on them. Normal online tracker state updates are fine.
- **Manual court landmark selection is allowed** (describe it in the paper). You can add
  more landmarks than the 4 corners, such as free-throw line, arc, or hoop base, for a
  better homography. **No manual per-frame annotation or correction** of boxes, tracks,
  teams, or jerseys.
- Only one prediction submission. Code must reproduce it.

## 6. Pipeline sketch and where the points are

1. **Calibration**: refine the homography from extra manually clicked landmarks, and check
   lens distortion. Every pixel matters at 720p.
2. **Detection**: a strong person detector (e.g. YOLO11x / RT-DETR) fine-tuned on the
   training bboxes. Use resolution/viewpoint augmentation: downscale 4K to 720p, plus
   perspective warps that mimic the oblique view. Filter out spectators, referees, and
   bench players with a court mask.
3. **Foot point**: bbox bottom-centre (the GT convention). Consider pose keypoints (ankles)
   or a regressor trained to predict bottom-centre from training data. Then smooth
   temporally (Kalman / RTS smoother).
4. **Tracking**: ByteTrack/BoT-SORT/OC-SORT, with the knowledge that there are always
   exactly 6 players. Use global offline association (tracklet stitching), because the
   whole video is available.
5. **Team classification**: cluster torso colour into 2 teams per video, then name them
   A/B by "offense at game start".
6. **Jersey OCR**: 720p crops are tiny. Use tracklet-level voting, the roster constraint
   (only 3 valid numbers per team), and a Hungarian assignment of tracklets to the 6
   identities with a temporal non-overlap constraint.
7. **Gap filling**: interpolate positions through occlusions and missed detections, and
   always emit the 6 identities when confident.
8. **Local validation**: run on training videos downscaled to 720p with the official
   TI-HOTA code. Note that this still differs in viewpoint from the final videos.

## 7. Open questions worth emailing the organisers about (abc2027@autocare.ai, subject "ABC Challenge")

- How are the per-video scores aggregated: mean of the two videos, or pooled frames?
- How is the challenge GT time-aligned to the oblique video frames, and does it cover every
  frame of the MP4 (11 536 / 11 823)?
- Is the challenge GT position the top-view bbox bottom-centre (as in training)?
- When the official form is published, does it expect the exact 5-column format above? The
  public eval notebook reads `frame, id, x, y, attr`.
