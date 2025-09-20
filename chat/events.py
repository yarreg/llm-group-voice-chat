"""Simple in-memory pub/sub broker for websocket clients."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Subscription:
    queue: asyncio.Queue


class EventBroker:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    async def subscribe(self) -> Subscription:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        return Subscription(queue=queue)

    async def unsubscribe(self, subscription: Subscription) -> None:
        self._subscribers.discard(subscription.queue)

    async def publish(self, event: dict[str, object]) -> None:
        targets = list(self._subscribers)
        for queue in targets:
            await queue.put(event)

    async def stream(self, subscription: Subscription) -> AsyncIterator[dict[str, object]]:
        try:
            while True:
                event = await subscription.queue.get()
                yield event
        finally:
            await self.unsubscribe(subscription)
