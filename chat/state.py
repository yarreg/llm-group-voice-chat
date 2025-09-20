"""Runtime conversation state models."""

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config import SceneConfig


class TurnStatus(str, Enum):
    GENERATED = "generated"
    TTS = "tts"
    FLP = "flp"
    READY = "ready"
    PLAYING = "playing"
    FINISHED = "finished"
    ERROR = "error"


_turn_counter = itertools.count()


def make_turn_id(conversation_id: str) -> str:
    return f"{conversation_id}-{int(time.time() * 1000)}-{next(_turn_counter)}"


@dataclass
class TurnState:
    turn_id: str
    idx: int
    speaker_id: str
    speaker_name: str
    text: str
    status: TurnStatus = TurnStatus.GENERATED
    media_url: str | None = None
    duration_ms: int = 0
    error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "idx": self.idx,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "text": self.text,
            "status": self.status.value,
            "media_url": self.media_url,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ConversationState:
    config: SceneConfig
    turns: list[TurnState] = field(default_factory=list)
    playing: bool = False
    current_turn_idx: int | None = None
    current_speaker_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "conversation_id": self.config.conversation.id,
            "title": self.config.conversation.title,
            "mode": self.config.conversation.mode,
            "playing": self.playing,
            "current_turn": self.current_turn_idx,
            "current_speaker": self.current_speaker_id,
            "turns": [turn.to_dict() for turn in self.turns],
        }

    def add_turn(self, turn: TurnState) -> None:
        self.turns.append(turn)
        self.current_turn_idx = turn.idx
        self.current_speaker_id = turn.speaker_id
