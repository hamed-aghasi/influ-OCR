"""The wire contract: metric fields, the pydantic response models, and the strict
JSON schema derived from them once at import (RESPONSE_FORMAT)."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

_METRIC_FIELDS = (
    "views", "followers", "non_followers", "accounts_reached",
    "interactions", "likes", "replies", "shares",
    "links_clicks", "sticker_taps", "navigation",
    "forward", "next_story", "back", "exited",
    "profile_activity", "profile_visits", "external_link_taps", "follows",
    "gender_men", "gender_women",
    "source_feed", "source_reels_tab", "source_profile", "source_explore",
    "source_stories", "source_search", "source_other",
)


_PERCENT_FIELDS = {"followers", "non_followers", "gender_men", "gender_women",
                   "source_feed", "source_reels_tab", "source_profile", "source_explore",
                   "source_stories", "source_search", "source_other"}


# No real Instagram metric exceeds this; observed model failure mode is
# digit-repetition hallucinations hundreds of digits long.
_MAX_PLAUSIBLE_VALUE = 10_000_000_000


class Metrics(BaseModel):
    views: Optional[int] = None
    followers: Optional[float] = None
    non_followers: Optional[float] = None
    accounts_reached: Optional[int] = None
    interactions: Optional[int] = None
    likes: Optional[int] = None
    replies: Optional[int] = None
    shares: Optional[int] = None
    links_clicks: Optional[int] = None
    sticker_taps: Optional[int] = None
    navigation: Optional[int] = None
    forward: Optional[int] = None
    next_story: Optional[int] = None
    back: Optional[int] = None
    exited: Optional[int] = None
    profile_activity: Optional[int] = None
    profile_visits: Optional[int] = None
    external_link_taps: Optional[int] = None
    follows: Optional[int] = None
    gender_men: Optional[float] = None      # Audience > Gender: Men %
    gender_women: Optional[float] = None    # Audience > Gender: Women %
    source_feed: Optional[float] = None       # Reel "Top sources of views", each a %
    source_reels_tab: Optional[float] = None
    source_profile: Optional[float] = None
    source_explore: Optional[float] = None
    source_stories: Optional[float] = None
    source_search: Optional[float] = None
    source_other: Optional[float] = None

    @field_validator("*", mode="after")
    @classmethod
    def _drop_implausible(cls, v, info):
        if v is None:
            return None
        if info.field_name in _PERCENT_FIELDS:
            return v if 0 <= v <= 100 else None
        return v if 0 <= v <= _MAX_PLAUSIBLE_VALUE else None


class Metadata(BaseModel):
    language: Optional[str] = None
    date_range: Optional[str] = None
    content_type: Optional[str] = None


# NB: no docstring on the wire models below — pydantic copies class docstrings
# into the schema as `description`, which then ships to the model on every
# request. `actual_frame` is likewise absent by design: it is filled in locally
# in ocr_single_batch, so asking the model to invent a filename it cannot know
# would cost tokens per frame and be overwritten regardless.
class FrameResult(BaseModel):
    frame_index: int = Field(ge=0)
    metrics: Optional[Metrics] = None
    metadata: Optional[Metadata] = None


class FrameResults(BaseModel):
    # Response envelope enforced via OpenRouter structured outputs.
    frames: List[FrameResult] = []


# Validation keywords that OpenAI-style `strict` structured outputs reject.
# Dropping them from the *wire* schema costs nothing: the pydantic models
# still enforce every one of them locally in _parse_content.
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
        "minLength", "maxLength", "pattern", "minItems", "maxItems", "format", "default",
    }
)


def _strict_schema(node):
    """Rewrite a pydantic JSON schema to satisfy `strict: true`.

    Strict mode requires every object to set `additionalProperties: false` and
    to list *all* its properties in `required`, and rejects the validation
    keywords above. pydantic emits none of that, so declaring strict over a raw
    model_json_schema() is a latent 400 — it happens to pass with the current
    model and would fail on the next one, retried 5x before failing the batch.
    """
    if isinstance(node, list):
        return [_strict_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    out = {k: _strict_schema(v) for k, v in node.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
    if out.get("type") == "object":
        out["additionalProperties"] = False
        out["required"] = list(out.get("properties") or {})
    return out


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "frame_results",
        "strict": True,
        "schema": _strict_schema(FrameResults.model_json_schema()),
    },
}
