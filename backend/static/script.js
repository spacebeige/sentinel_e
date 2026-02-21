document.addEventListener('DOMContentLoaded', () => {
    // --- ELEMENTS ---
    const themeBtn = document.getElementById('theme-toggle');
    const htmlEl = document.documentElement;
    const bodyEl = document.body;
    const input = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const feed = document.getElementById('chat-feed');
    const modeSwitch = document.getElementById('mode-switch');
    const historyList = document.getElementById('history-list');
    const newChatBtn = document.querySelector('.new-chat');

    // --- STATE ---
    let isTyping = false;

    // --- 0. INITIAL ANIMATION ---
    const sidebarEls = document.querySelectorAll('.sidebar > *');
    sidebarEls.forEach((el, index) => {
        el.style.opacity = '0';
        el.style.transform = 'translateX(-10px)';
        el.style.transition = 'all 0.4s ease';
        setTimeout(() => {
            el.style.opacity = '1';
            el.style.transform = 'translateX(0)';
        }, 100 + (index * 100));
    });

    // --- 1. THEME TOGGLE ---
    const updateThemeIcon = (theme) => {
        const iconName = theme === 'light' ? 'moon' : 'sun';
        themeBtn.innerHTML = `<i data-lucide="${iconName}"></i>`;
        lucide.createIcons();
    };

    themeBtn.addEventListener('click', () => {
        const current = htmlEl.getAttribute('data-theme') || 'light';
        const newTheme = current === 'light' ? 'dark' : 'light';
        
        htmlEl.setAttribute('data-theme', newTheme);
        localStorage.setItem('sentinel-theme', newTheme);
        updateThemeIcon(newTheme);
    });

    const savedTheme = localStorage.getItem('sentinel-theme') || 'dark';
    htmlEl.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    // --- 2. MODE SWITCHING ---
    modeSwitch.addEventListener('change', () => {
        const mode = modeSwitch.value;
        bodyEl.setAttribute('data-mode-active', mode);
    });
    // Set initial
    bodyEl.setAttribute('data-mode-active', modeSwitch.value);

    // --- 3. INPUT HANDLING ---
    input.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        const hasContent = this.value.trim().length > 0;
        sendBtn.disabled = !hasContent;
        sendBtn.classList.toggle('ready', hasContent);
    });

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    // --- 4. HISTORY HANDLING ---
    async function loadHistory() {
        try {
            const res = await fetch('/api/history');
            const data = await res.json();
            historyList.innerHTML = '';
            
            data.forEach(item => {
                const li = document.createElement('li');
                li.className = 'history-item';
                li.textContent = item.filename; // Or formatted date
                li.title = item.timestamp;
                li.onclick = () => loadLog(item.filename);
                historyList.appendChild(li);
            });
        } catch (err) {
            console.error("Failed to load history", err);
        }
    }

    async function loadLog(filename) {
        try {
            const res = await fetch(`/api/history/${filename}`);
            const data = await res.json();
            
            // Clear feed or just append? Usually implies viewing that logs.
            // Let's clear feed for "viewing logs" mode, or just append a system message.
            // User requested "Render the result immediately in the chat window."
            
            appendMessage('ai', `Analysis of log: ${filename}`, false);
            const jsonStr = JSON.stringify(data, null, 2);
            appendMessage('ai', `<pre><code>${jsonStr}</code></pre>`, true);
            
        } catch (err) {
            console.error(err);
        }
    }

    if (newChatBtn) {
        newChatBtn.addEventListener('click', () => {
            feed.innerHTML = '';
            const welcome = document.createElement('div');
            welcome.className = 'message-row ai-row';
            welcome.innerHTML = `
                <div class="msg-avatar ai">S</div>
                <div class="msg-content">
                    <p><strong>SYSTEM CLEARED.</strong></p>
                    <p>Ready for new task.</p>
                </div>
            `;
            feed.appendChild(welcome);
        });
    }

    loadHistory();

    // --- 5. MESSAGE LOGIC ---
    async function sendMessage() {
        const text = input.value.trim();
        if(!text || isTyping) return;

        isTyping = true;
        appendMessage('user', text);
        
        input.value = '';
        input.style.height = 'auto';
        sendBtn.classList.remove('ready');
        sendBtn.disabled = true;

        const loadingId = addLoadingIndicator();

        try {
            const mode = modeSwitch.value;
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, mode: mode })
            });
            const data = await response.json();
            
            removeLoadingIndicator(loadingId);
            
            if (data.type === 'error') {
                appendMessage('ai', `**System Error:** ${data.content}`, true);
            } else if (data.type === 'json') {
                // Render JSON card
                const jsonStr = JSON.stringify(data.content, null, 2);
                appendMessage('ai', `### **DIAGNOSTIC OUTPUT**\n<pre><code>${jsonStr}</code></pre>`, true);
            } else {
                // Text/markdown
                const parsed = typeof marked !== 'undefined' ? marked.parse(data.content) : data.content;
                appendMessage('ai', parsed, true);
            }

        } catch (e) {
            removeLoadingIndicator(loadingId);
            appendMessage('ai', `**Network Error:** ${e.message}`, true);
        }
        
        isTyping = false;
        loadHistory(); // Refresh history list
    }

    function appendMessage(role, content, isHtml = false) {
        const row = document.createElement('div');
        row.className = `message-row ${role}-row`;
        
        const avatarClass = role === 'user' ? 'user' : 'ai';
        const avatarLetter = role === 'user' ? 'U' : 'S';
        
        // Structure
        row.innerHTML = `
            <div class="msg-avatar ${avatarClass}">${avatarLetter}</div>
            <div class="msg-content"></div>
        `;

        feed.appendChild(row);

        const contentDiv = row.querySelector('.msg-content');
        
        if (role === 'ai') {
            if (isHtml) {
                contentDiv.innerHTML = content;
            } else {
                // If simple text
                typeWriter(contentDiv, content);
            }
        } else {
            contentDiv.innerHTML = isHtml ? content : content.replace(/\n/g, '<br>');
        }

        lucide.createIcons();
        scrollToBottom();
    }

    function typeWriter(element, text) {
        element.innerHTML = text; // Simplify for now to focus on logic
        element.animate([
            { opacity: 0, transform: 'translateY(5px)' },
            { opacity: 1, transform: 'translateY(0)' }
        ], {
            duration: 400,
            easing: 'ease-out'
        });
    }

    function addLoadingIndicator() {
        const id = 'loading-' + Date.now();
        const loader = document.createElement('div');
        loader.id = id;
        loader.className = 'message-row ai-row loading-indicator';
        loader.innerHTML = `
            <div class="msg-avatar ai">S</div>
            <div class="msg-content typing-dots" style="font-family:var(--font-mono); color:var(--text-muted);">
                PROCESSING<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
            </div>
        `;
        feed.appendChild(loader);
        
        const dots = loader.querySelectorAll('.dot');
        let dotStep = 0;
        const dotInt = setInterval(() => {
            dots.forEach(d => d.style.opacity = 0.2);
            dots[dotStep].style.opacity = 1;
            dotStep = (dotStep + 1) % 3;
        }, 300);
        loader.dataset.interval = dotInt;

        scrollToBottom();
        return id;
    }

    function removeLoadingIndicator(id) {
        const el = document.getElementById(id);
        if (el) {
            clearInterval(el.dataset.interval);
            el.remove();
        }
    }

    function scrollToBottom() {
        feed.scrollTo({
            top: feed.scrollHeight,
            behavior: 'smooth'
        });
    }
});
