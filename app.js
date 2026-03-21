const terminal = document.getElementById('terminal');
const startBtn = document.getElementById('start-btn');
const liveScreen = document.getElementById('live-screen');
const placeholderText = document.getElementById('placeholder-text');
const statusBadge = document.getElementById('connection-status');

let ws = null;

function appendLog(message, type = 'normal') {
    const div = document.createElement('div');
    div.className = `log-entry ${type === 'action' ? 'log-action' : ''} ${type === 'error' ? 'log-error' : ''} ${type === 'success' ? 'log-success' : ''}`;
    div.textContent = message;
    terminal.appendChild(div);
    terminal.scrollTop = terminal.scrollHeight; // Auto-scroll
}

startBtn.addEventListener('click', () => {
    const url = document.getElementById('target-url').value;
    const objective = document.getElementById('objective').value;

    if (!url || !objective) {
        alert("Please enter both a URL and an objective.");
        return;
    }

    // Clear previous session
    terminal.innerHTML = '';
    liveScreen.style.display = 'none';
    placeholderText.style.display = 'block';

    // Connect WebSocket
    ws = new WebSocket("ws://localhost:8000/ws");

    ws.onopen = () => {
        statusBadge.textContent = "Connected & Running";
        statusBadge.style.color = "var(--terminal-green)";
        startBtn.disabled = true;
        
        // Send objective to backend
        ws.send(JSON.stringify({ url, objective }));
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'image') {
            // Render the base64 screenshot
            liveScreen.src = `data:image/jpeg;base64,${data.data}`;
            liveScreen.style.display = 'block';
            placeholderText.style.display = 'none';
        } else {
            // It's a log, action, error, or success message
            appendLog(data.msg, data.type);
        }
        
        if (data.type === 'success' || (data.msg && data.msg.includes('Agent task finished'))) {
            ws.close();
        }
    };

    ws.onclose = () => {
        statusBadge.textContent = "Disconnected";
        statusBadge.style.color = "var(--text-dim)";
        startBtn.disabled = false;
    };
});