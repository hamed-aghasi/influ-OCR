# Ground truth: which frames show which metric

Lets `training/eval_selection.py` measure what any frame-selection strategy
**loses**, not only how many frames it sends. No OCR is involved.

One folder per video. Open `frames.html` in a browser (every extracted frame,
with its frame number in yellow; click for full size) and fill in `truth.json`
next to it.

## What to fill in

For each of the 19 metrics that appears on an **Insights screen** in the video:

- `value`: the number as shown, e.g. `21756`. For `followers` and
  `non_followers`, the percentage, e.g. `99.8`. Expand K/M: `8.6K` → `8600`.
  Leave `null` if the screen shows `--`.
- `frames`: the frame numbers where that metric is **fully readable**, meaning
  the label (or the icon, for the Reel icon strip) *and* the complete number are
  on screen: not cut off at the top or bottom edge, not a loading placeholder,
  not blurred mid-scroll. Write them as ranges: `"066-105"`, or
  `"066-105, 213-228"` if the metric comes back later. The ranges don't have to be
  pixel-perfect; one frame too many or too few at an edge is fine.

Leave metrics that never appear as `{"value": null, "frames": ""}`.

Also set `insights_type` to `"reel"`, `"story"` or `"post"`, and when the video
is finished, set `"status": "done"`. Anything else goes in `notes`.

**`c70ca2f3…` is already filled in** as a worked example, drafted from the
frames. Check it, fix anything wrong, then set its status to `"done"`.

## Which on-screen label is which metric

| key | on screen |
|---|---|
| `views` | Views / بازدیدها: Summary tile, Views donut, Overview row |
| `accounts_reached` | Accounts reached / Viewers / بینندگان / بازدیدکنندگان |
| `followers`, `non_followers` | the % split under the Views donut |
| `interactions` | Interactions / تعاملات |
| `likes` | Likes, or ♥ in the Reel icon strip / پسندها |
| `replies` | Replies, Comments, or 💬 in the icon strip / پاسخ‌ها |
| `shares` | Shares, or ✈ paper plane in the icon strip / اشتراک‌گذاری‌ها |
| `links_clicks` | Link clicks / Bio link clicks |
| `sticker_taps` | Sticker taps |
| `navigation`, `forward`, `next_story`, `back`, `exited` | the Navigation block (Stories) |
| `profile_activity`, `profile_visits`, `external_link_taps`, `follows` | the Profile activity block |
| `gender_men`, `gender_women` | Audience > **Gender** view: Men % / Women % (مردان / زنان). Not the Age or Country view |
| `source_feed`, `source_reels_tab`, `source_profile`, `source_explore`, `source_stories`, `source_search`, `source_other` | Reel **Top sources of views** (منابع برتر بازدید): one % per row. Label and value both visible |

The icon strip also shows reposts (⟳) and saves (🔖). They are not in the
schema, so skip them.

## Then

    .venv-train/bin/python training/eval_selection.py
