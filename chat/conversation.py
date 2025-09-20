import asyncio
import contextlib
import logging
import re
from asyncio import Queue
from collections import deque
from dataclasses import dataclass

from autogen_engine import AutoGenEngine
from config import ActorConfig, AppConfig, SceneConfig
from events import EventBroker
from media import MediaPipeline, MediaResult
from state import ConversationState, TurnState, TurnStatus, make_turn_id

logger = logging.getLogger(__name__)


@dataclass
class InjectedMessage:
    actor_id: str
    text: str


class ConversationManager:
    def __init__(
        self,
        app_config: AppConfig,
        scene_config: SceneConfig,
        media_pipeline: MediaPipeline,
        broker: EventBroker,
    ) -> None:
        self._app_config = app_config
        self._scene_config = scene_config
        self._media_pipeline = media_pipeline
        self._broker = broker
        self._engine = AutoGenEngine(app_config)

        self._state = ConversationState(scene_config)
        self._seed_messages = self._prepare_seed_messages()

        self._actors_by_id: dict[str, ActorConfig] = {actor.id: actor for actor in scene_config.actors}
        self._actors_by_name: dict[str, ActorConfig] = {actor.name: actor for actor in scene_config.actors}
        self._round_robin: deque[ActorConfig] = self._build_round_robin()

        for actor in scene_config.actors:
            system_prompt = self._compose_system_prompt(actor)
            self._engine.register_actor(actor, system_prompt)

        self._producer_task: asyncio.Task | None = None
        self._playback_task: asyncio.Task | None = None
        self._playback_queue: Queue[str] = Queue()
        self._media_futures: dict[str, asyncio.Future] = {}
        self._playback_waiters: dict[str, asyncio.Event] = {}
        self._injected: Queue[InjectedMessage] = Queue()
        self._running = asyncio.Event()
        self._stop_requested = False
        self._turn_limit = scene_config.conversation.max_turns
        self._generated_turn_count = 0
        self._termination_pattern = (
            re.compile(scene_config.conversation.termination_regex)
            if scene_config.conversation.termination_regex
            else None
        )

    @property
    def state(self) -> ConversationState:
        return self._state

    async def start(self) -> None:
        if self._producer_task and not self._producer_task.done():
            logger.info("Conversation already running")
            return
        logger.info("Starting conversation %s", self._scene_config.conversation.id)
        self._reset_state()
        self._stop_requested = False
        self._running.set()
        await self._broker.publish(
            {"type": "conversation.started", "conversation_id": self._scene_config.conversation.id}
        )
        self._state.playing = True

        loop = asyncio.get_running_loop()
        self._producer_task = loop.create_task(self._producer_loop())
        self._playback_task = loop.create_task(self._playback_loop())

    async def stop(self, reason: str = "stopped") -> None:
        if self._stop_requested:
            return
        logger.info("Stopping conversation: %s", reason)
        self._stop_requested = True
        self._running.clear()

        tasks = [self._producer_task, self._playback_task]
        for task in tasks:
            if task:
                task.cancel()
        for task in tasks:
            if task:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        await self._cleanup_pending_media()
        self._state.playing = False
        await self._broker.publish({"type": "conversation.stopped", "reason": reason})
        self._producer_task = None
        self._playback_task = None

    async def restart(self) -> None:
        await self.stop(reason="restart")
        await asyncio.sleep(0)
        await self.start()

    async def inject_user_message(self, actor_id: str, text: str) -> None:
        if not self._scene_config.conversation.allow_user_injection:
            raise ValueError("User injection disabled for this conversation")
        if actor_id not in self._actors_by_id:
            raise ValueError(f"Unknown actor: {actor_id}")
        logger.info("Injecting user message for actor %s", actor_id)
        await self._injected.put(InjectedMessage(actor_id=actor_id, text=text))

    async def _producer_loop(self) -> None:
        try:
            while self._running.is_set():
                if self._generated_turn_count >= self._turn_limit:
                    logger.info("Reached maximum number of turns: %s", self._turn_limit)
                    break

                injected: InjectedMessage | None = None
                try:
                    injected = self._injected.get_nowait()
                except asyncio.QueueEmpty:
                    injected = None

                next_actor = self._select_next_actor(injected)
                if not next_actor:
                    logger.info("No actor available, stopping conversation")
                    break

                turn_id = make_turn_id(self._scene_config.conversation.id)
                turn_idx = len(self._state.turns)
                text = injected.text if injected else await self._generate_text(next_actor)

                turn = TurnState(
                    turn_id=turn_id,
                    idx=turn_idx,
                    speaker_id=next_actor.id,
                    speaker_name=next_actor.name,
                    text=text,
                    status=TurnStatus.GENERATED,
                )
                self._state.add_turn(turn)
                self._generated_turn_count += 1

                await self._broker.publish(
                    {
                        "type": "turn.generated",
                        "turn_id": turn.turn_id,
                        "speaker_id": turn.speaker_id,
                        "text": turn.text,
                    }
                )

                playback_gate = asyncio.Event()
                self._playback_waiters[turn.turn_id] = playback_gate

                media_task = asyncio.create_task(self._generate_media(turn, next_actor))
                self._media_futures[turn.turn_id] = media_task
                await self._playback_queue.put(turn.turn_id)

                if self._termination_pattern and self._termination_pattern.search(text):
                    logger.info("Termination regex matched; stopping after current turn")
                    break

                await playback_gate.wait()
                if self._stop_requested:
                    break
        except asyncio.CancelledError:
            logger.debug("Producer loop cancelled")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected error in producer loop: %s", exc)
            await self._broker.publish({"type": "error", "turn_id": None, "message": str(exc)})
        finally:
            await self._playback_queue.put("__STOP__")

    async def _playback_loop(self) -> None:
        try:
            while True:
                turn_id = await self._playback_queue.get()
                if turn_id == "__STOP__":
                    break
                turn = self._find_turn(turn_id)
                future = self._media_futures.get(turn_id)
                if not future:
                    logger.warning("No media future for turn %s", turn_id)
                    continue
                try:
                    result: MediaResult = await future
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    turn.status = TurnStatus.ERROR
                    turn.error_message = str(exc)
                    await self._broker.publish(
                        {
                            "type": "error",
                            "turn_id": turn.turn_id,
                            "message": f"Media generation failed: {exc}",
                        }
                    )
                    self._release_playback_waiter(turn.turn_id)
                    continue

                turn.status = TurnStatus.READY
                turn.media_url = result.media_url
                turn.duration_ms = result.duration_ms

                await self._broker.publish(
                    {
                        "type": "assets.ready",
                        "turn_id": turn.turn_id,
                        "media_url": result.media_url,
                        "duration_ms": result.duration_ms,
                    }
                )

                await self._start_playback(turn)
        except asyncio.CancelledError:
            logger.debug("Playback loop cancelled")
        finally:
            self._state.playing = False

    async def _start_playback(self, turn: TurnState) -> None:
        await self._broker.publish({"type": "playback.started", "turn_id": turn.turn_id})
        self._release_playback_waiter(turn.turn_id)
        turn.status = TurnStatus.PLAYING
        if turn.duration_ms > 0:
            await asyncio.sleep(turn.duration_ms / 1000)
        turn.status = TurnStatus.FINISHED
        await self._broker.publish({"type": "playback.finished", "turn_id": turn.turn_id})

    def _find_turn(self, turn_id: str) -> TurnState:
        for turn in self._state.turns:
            if turn.turn_id == turn_id:
                return turn
        raise KeyError(f"Turn not found: {turn_id}")

    def _release_playback_waiter(self, turn_id: str) -> None:
        waiter = self._playback_waiters.pop(turn_id, None)
        if waiter:
            waiter.set()

    async def _generate_media(self, turn: TurnState, actor: ActorConfig) -> MediaResult:
        async def _progress(stage: str, percent: int) -> None:
            await self._broker.publish(
                {"type": "assets.progress", "turn_id": turn.turn_id, "stage": stage, "percent": percent}
            )

        result = await self._media_pipeline.generate(
            actor=actor,
            text=turn.text,
            conversation_id=self._scene_config.conversation.id,
            turn_id=turn.turn_id,
            progress=_progress,
        )
        return result

    def _select_next_actor(self, injected: InjectedMessage | None) -> ActorConfig | None:
        if injected:
            return self._actors_by_id[injected.actor_id]
        mode = self._scene_config.conversation.mode
        if mode == "round_robin":
            actor = self._round_robin[0]
            self._round_robin.rotate(-1)
            return actor
        # free mode fallback: same as round robin but without rotation logic
        return self._round_robin[0]

    async def _generate_text(self, actor: ActorConfig) -> str:
        messages: list[dict[str, str]] = []
        for entry in self._seed_messages:
            data = dict(entry)
            if data.get("source") == actor.name:
                data["role"] = "assistant"
            elif data.get("role") == "assistant" and data.get("source") != actor.name:
                data["role"] = "user"
            messages.append(data)
        for turn in self._state.turns:
            role = "assistant" if turn.speaker_id == actor.id else "user"
            messages.append(
                {
                    "role": role,
                    "content": turn.text,
                    "source": turn.speaker_name,
                }
            )
        reply = await self._engine.generate(actor=actor, messages=messages)
        return reply

    def _compose_system_prompt(self, actor: ActorConfig) -> str:
        pieces = [self._scene_config.prompts.global_system, actor.system_prompt]
        if actor.extra_prompt:
            pieces.append(actor.extra_prompt)
        return "\n".join(piece.strip() for piece in pieces if piece)

    def _build_round_robin(self) -> deque[ActorConfig]:
        actors = deque(self._scene_config.actors)
        start_name = self._scene_config.conversation.start_speaker
        while actors[0].name != start_name:
            actors.rotate(-1)
        return actors

    def _reset_state(self) -> None:
        self._state = ConversationState(self._scene_config)
        self._seed_messages = self._prepare_seed_messages()
        self._generated_turn_count = 0
        self._round_robin = self._build_round_robin()
        self._media_futures.clear()
        self._playback_waiters.clear()
        self._playback_queue = Queue()
        self._injected = Queue()

    def _prepare_seed_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for seed in self._scene_config.script.seed_messages:
            messages.append(
                {
                    "role": seed.role,
                    "content": seed.content,
                    "source": seed.speaker or "system",
                }
            )
        return messages

    async def _cleanup_pending_media(self) -> None:
        for future in list(self._media_futures.values()):
            if not future.done():
                future.cancel()
        self._media_futures.clear()
        for waiter in self._playback_waiters.values():
            waiter.set()
        self._playback_waiters.clear()
