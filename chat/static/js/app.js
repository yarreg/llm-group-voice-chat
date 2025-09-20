(() => {
    const context = window.JOYVASA_CONTEXT;
    const chatList = document.getElementById('chat-list');
    const actorCards = new Map();
    document.querySelectorAll('.actor-card').forEach(card => {
        actorCards.set(card.dataset.actorId, card);
    });

    const videos = new Map();
    document.querySelectorAll('.actor-card video').forEach(video => {
        const actorId = video.closest('.actor-card').dataset.actorId;
        videos.set(actorId, video);
    });

    const turns = new Map();

    const toastContainer = document.getElementById('toast-container');
    function showToast(message, variant = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-bg-light border-0 show ${variant}`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>`;
        toastContainer.appendChild(toast);
        setTimeout(() => toast.remove(), 5000);
    }

    function makeBubble(turnId, actorId, text) {
        const actor = context.actors[actorId];
        const li = document.createElement('li');
        li.dataset.turnId = turnId;
        li.style.borderLeftColor = actor.color;
        li.innerHTML = `<span class="speaker" style="color: ${actor.color};">${actor.name}</span><span class="text">${text}</span><div class="status small text-muted mt-1"></div>`;
        chatList.appendChild(li);
        chatList.scrollTop = 0;
        return li;
    }

    function updateStatus(turnId, text) {
        const bubble = chatList.querySelector(`li[data-turn-id="${turnId}"] .status`);
        if (bubble) {
            bubble.textContent = text;
        }
    }

    function setBubbleError(turnId, message) {
        const li = chatList.querySelector(`li[data-turn-id="${turnId}"]`);
        if (li) {
            li.classList.add('error');
            updateStatus(turnId, message);
        }
    }

    function attachVideo(turnId, mediaUrl, actorId) {
        const video = videos.get(actorId);
        if (!video) return;
        video.src = `${mediaUrl}?t=${Date.now()}`;
        video.load();
    }

    function handlePlaybackStart(turnId) {
        const turn = turns.get(turnId);
        if (!turn) return;
        const card = actorCards.get(turn.speaker_id);
        const video = videos.get(turn.speaker_id);
        if (card) {
            card.classList.add('playing');
        }
        if (video) {
            video.currentTime = 0;
            video.play().catch(() => {
                showToast('Unable to autoplay video – click to resume.', 'warning');
            });
        }
    }

    function handlePlaybackFinish(turnId) {
        const turn = turns.get(turnId);
        if (!turn) return;
        const card = actorCards.get(turn.speaker_id);
        const video = videos.get(turn.speaker_id);
        if (card) {
            card.classList.remove('playing');
        }
        if (video) {
            video.pause();
        }
    }

    function applySnapshot(snapshot) {
        snapshot.turns.forEach(turn => {
            turns.set(turn.turn_id, turn);
            if (!chatList.querySelector(`li[data-turn-id="${turn.turn_id}"]`)) {
                makeBubble(turn.turn_id, turn.speaker_id, turn.text);
            }
            if (turn.media_url) {
                attachVideo(turn.turn_id, turn.media_url, turn.speaker_id);
            }
            if (turn.status === 'playing') {
                handlePlaybackStart(turn.turn_id);
            }
        });
    }

    function handleEvent(event) {
        switch (event.type) {
            case 'conversation.state':
                applySnapshot(event);
                break;
            case 'turn.generated':
                turns.set(event.turn_id, { ...event, speaker_name: context.actors[event.speaker_id]?.name ?? event.speaker_id });
                makeBubble(event.turn_id, event.speaker_id, event.text);
                updateStatus(event.turn_id, 'Generating media…');
                break;
            case 'assets.progress':
                updateStatus(event.turn_id, `${event.stage.toUpperCase()} ${event.percent}%`);
                break;
            case 'assets.ready':
                const turn = turns.get(event.turn_id) || {};
                turn.media_url = event.media_url;
                turn.duration_ms = event.duration_ms;
                turns.set(event.turn_id, turn);
                attachVideo(event.turn_id, event.media_url, turn.speaker_id);
                updateStatus(event.turn_id, 'Ready to play');
                break;
            case 'playback.started':
                handlePlaybackStart(event.turn_id);
                updateStatus(event.turn_id, 'Playing…');
                break;
            case 'playback.finished':
                updateStatus(event.turn_id, 'Finished');
                handlePlaybackFinish(event.turn_id);
                break;
            case 'conversation.stopped':
                showToast(`Conversation stopped: ${event.reason}`, 'warning');
                break;
            case 'error':
                showToast(event.message, 'error');
                if (event.turn_id) {
                    setBubbleError(event.turn_id, event.message);
                }
                break;
            default:
                console.debug('Unhandled event', event);
        }
    }

    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const ws = new WebSocket(`${protocol}://${window.location.host}/ws`);
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleEvent(data);
            } catch (err) {
                console.error('Unable to parse websocket message', err, event.data);
            }
        };
        ws.onclose = () => {
            showToast('WebSocket connection closed', 'warning');
            setTimeout(connectWebSocket, 3000);
        };
        ws.onerror = () => {
            showToast('WebSocket error encountered', 'error');
        };
    }

    connectWebSocket();

    async function callApi(url, options = {}) {
        const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, ...options });
        if (!response.ok) {
            const message = await response.text();
            throw new Error(message || 'API error');
        }
        return response.json();
    }

    document.getElementById('btn-start').addEventListener('click', async () => {
        try {
            await callApi('/api/conversations/start');
        } catch (err) {
            showToast(err.message, 'error');
        }
    });

    document.getElementById('btn-stop').addEventListener('click', async () => {
        try {
            await callApi('/api/conversations/stop');
        } catch (err) {
            showToast(err.message, 'error');
        }
    });

    document.getElementById('btn-restart').addEventListener('click', async () => {
        try {
            await callApi('/api/conversations/restart');
        } catch (err) {
            showToast(err.message, 'error');
        }
    });

    document.getElementById('message-form').addEventListener('submit', async (event) => {
        event.preventDefault();
        const speaker = document.getElementById('speaker').value;
        const messageInput = document.getElementById('message');
        const text = messageInput.value.trim();
        if (!text) {
            messageInput.focus();
            return;
        }
        try {
            await callApi('/api/user_message', { body: JSON.stringify({ speaker_id: speaker, text }) });
            messageInput.value = '';
        } catch (err) {
            showToast(err.message, 'error');
        }
    });
})();
