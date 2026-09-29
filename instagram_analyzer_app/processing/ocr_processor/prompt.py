"""The extraction prompt sent with every batch. Static text; no templating."""

EXTRACTION_PROMPT = """\
You are reading Instagram Insights screenshots. For each frame, extract any
metric values visible. Numbers may be in English digits or Persian digits
(۰-۹ map to 0-9). If a metric is not visible, omit it (leave it null).

Metric keys to extract (omit any not present):
- views, followers, non_followers, accounts_reached
- interactions, likes, replies, shares
- links_clicks, sticker_taps, navigation
- forward, next_story, back, exited
- profile_activity, profile_visits, external_link_taps, follows
- gender_men, gender_women
- source_feed, source_reels_tab, source_profile, source_explore,
  source_stories, source_search, source_other

On-screen label mapping (English / Persian):
- "Viewers" / بینندگان -> accounts_reached (NOT views; the "Views" donut or
  Summary row is views)
- "Comments" / نظرها -> replies (the JSON key for the comment count)
- "Bio link clicks" / کلیک‌های پیوند بیو -> links_clicks
- "Likes" / پسندها -> likes
- "Navigation" / پیمایش -> navigation
- "Forward" / بازارسال -> forward
- followers / non_followers are the percentage split under the Views donut —
  report the percentage number (e.g. 62.2)
- "Men" / مردان / مرد -> gender_men, "Women" / زنان / زن -> gender_women: the
  Gender split in the Audience section, as percentages (e.g. 95.9 and 4.1).
  Only from the Gender view — not from the Age or Country views.
- "Top sources of views" / منابع برتر بازدید (Reels): one percentage per row.
  Feed / فید -> source_feed, Reels tab / برگه Reels -> source_reels_tab,
  Profile / نمایه -> source_profile, Explore / کاوش -> source_explore,
  Stories -> source_stories, Search / جستجو -> source_search, Other / سایر ->
  source_other. Only from that list — "Profile" elsewhere is not a source.

Also read the icon strip at the top of Reel insights screens (see "Reels
engagement icon row" below) and the Engagement tab rows.
Expand K/M abbreviations to full numbers: 8.6K -> 8600, 1.2M -> 1200000.
Read carefully digit by digit; do not merge a number with a neighboring
number, and never repeat digits beyond what is on screen.

Respond with one entry in `frames` per input frame, in input order, using
each frame's zero-based position as `frame_index`.

Metric-key disambiguation (read carefully before extracting):

- "Views", "Viewers", and "Profile visits" sound similar in Persian
  (بازدیدها / بازدیدکنندگان / بازدیدهای نمایه) but are three separate
  metrics. Never conflate them:
    - "بازدیدها" / "Views"          -> views
    - "بازدیدکنندگان" / "Viewers"   -> accounts_reached
    - "بازدیدهای نمایه" / "Profile visits" -> profile_visits

- Ignore the small number next to the eye/person icon at the very top
  of the screen, above the thumbnail strip (e.g. "1429", "2510", "12326",
  "3520"). It is not a metric for this frame and can legitimately differ
  from the Views/Viewers numbers shown further down. Never use it to
  fill any metric field — always read the metric rows lower on screen.

- If a value is shown as "--" or "-", extract it as null. Do not report
  it as 0.

- Ignore any percentage/rate-style metric not explicitly in the metric
  key list (e.g. "Comment rate"). Do not extract these at all.

- sticker_taps is always a single total, even when the screen breaks it
  down per sticker/mention (e.g. "@handle: 4" listed under it). Report
  only the total; never create per-sticker fields.

Reel Summary block (only map fields that correspond to existing keys):
  "Views"         -> views
  "Viewers"       -> accounts_reached
  "Follows"       -> follows (null if shown as "--")

Reels engagement icon row (self-contained; applies only to the row of small
icons with a number under each, shown on Reel insights screens — it does
not change any rule above):
  Identify each metric ONLY by the shape of its icon, never by its position
  in the row or by how many icons there are.
    heart                                   -> likes
    speech bubble                           -> replies   (this is the Comments
                                               count; JSON key stays "replies")
    two arrows forming a loop (repost icon) -> no key: do NOT put this
                                               number anywhere
    paper plane (send icon)                 -> shares
    bookmark (save icon)                    -> no key: do NOT put this
                                               number anywhere
  The number belonging to an icon is the one directly below that icon. Never
  shift a number to a neighbouring icon, and never fill a key from an icon
  whose shape is not listed here. If an icon's number is "--", leave the key
  null.
"""
