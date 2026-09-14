import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class MemoryOut(BaseModel):
    """One remembered instruction, as the profile view shows it."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    text: str
    # What the user actually typed when this was captured. Shown so an
    # instruction that looks wrong can be traced to the turn that created it --
    # without it, a badly-captured preference is indistinguishable from one the
    # user really gave, and the only remedy is to delete and hope.
    source_message: str | None = None
    active: bool = True
    created_at: datetime


class ConversationMemoryOut(BaseModel):
    """What is remembered about ONE conversation.

    Two different kinds of memory, kept apart on purpose rather than merged
    into one list: `summary` is a lossy compression of what was discussed,
    regenerated as the chat grows; `preferences` are instructions kept
    verbatim and applied to every later turn. Showing them as one list would
    suggest deleting a line of the summary changes behaviour, which it does
    not.
    """

    session_id: uuid.UUID
    title: str
    summary: str | None = None
    preferences: list[MemoryOut] = []


class ProfileMemoryOut(BaseModel):
    # Scope NULL -- in force in every conversation.
    user_preferences: list[MemoryOut] = []
    conversations: list[ConversationMemoryOut] = []
