# Review round 01b — SUPERSEDED, do not review

An intermediate run, generated 2026-09-19 19:58, **22 minutes before**
`../round-01c/` (20:20). It is kept only as a record of what the last two fixes
changed. **Review `../round-01c/` instead**; leave the `feedback.json` files
here alone.

Same model (`2026-09-19b-sources` at 0.30) and same settings as 01c. The only
difference is pipeline code, both runs being from the working tree before commit
`eb53531`.

| video | 01b | **01c** | change |
|---|---|---|---|
| `1f2bdcac` | 4 | **3** | −213 |
| `4985b8cc` | 5 | **4** | −279 |
| `5361dd11` | 4 | **2** | −195, −222 |
| `8102e03d` | 11 | **5** | −354, −369, −402, −486, −621, −627 |
| `bf96a4ca` | 4 | **3** | −144 |
| `edc93edd` | 6 | **7** | set re-cut: −231/267/315/333, +201/237/243/252/327 |
| **total** | **34** | **24** | **no metric lost on any video** |

## The two fixes made between the runs

1. **The value backstop now guards counts only** (`processing/metric_coverage/`, the
   `guarded` set). It used to guard percentages too, and OCR reads small ones
   inconsistently ("1.1%" as "11%"), so each misread pulled in another frame —
   here, three copies of one Top sources list. Backstop-kept frames across the
   six videos fell from **17 to 5**, which is the whole of the 34 → 24 drop.
   Alongside it, `_canonical` strips the extra leading digit a bar glyph adds to
   a one-decimal percentage ("96.5%" read as "896.5"), so the same value on two
   frames compares equal.
2. **A section header no longer takes the next line's number**
   (`_is_section_header`). "Interactions ⓘ" at a screen edge was borrowing the
   donut's arc, read as digits, as its value. This removed false reads of
   `interactions` (8102e03d 384/402/414, edc93edd 264/267/276),
   `profile_activity` (edc93edd 201) and `views` (8102e03d 243/249/255).

`edc93edd` is the one video whose frames were re-cut rather than reduced: with
the false `interactions` reads gone, the set cover picked a different, larger
combination (6 → 7). Nothing is lost; OCR variance on counts is the next thing
to tighten.
