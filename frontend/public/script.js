document.addEventListener('DOMContentLoaded', () => {
    // --- ELEMENTS ---
    const themeBtn = document.getElementById('theme-toggle');
    const htmlEl = document.documentElement;
    const bodyEl = document.body;
    const input = document.getElementById('user-input');
    const sendBtn = document.querySelector('.send-btn'); // Changed to class selector based on HTML usually having class
    const feed = document.getElementById('chat-feed');
    const modeSwitch = document.getElementById('mode-switch');
    const historyList = document.getElementById('history-list');
    const newChatBtn = document.querySelector('.new-chat');
    
    // File Upload Elements
    const fileInput = document.getElementById('file-input');
    const attachBtn = document.getElementById('attach-btn'); // Corrected selector
    
    // Dynamically create preview container if it does not exist
    let filePreviewContainer = document.getElementById('file-preview-container');
    if (!filePreviewContainer) {
        filePreviewContainer = document.createElement('div');
        filePreviewContainer.id = 'file-preview-container';
        filePreviewContainer.className = 'file-preview hidden';
        // Insert it inside the input container, before the input wrapper if possible
        const inputContainer = document.querySelector('.input-container');
        if(inputContainer) {
            inputContainer.insertBefore(filePreviewContainer, inputContainer.firstChild);    
        }
    }

    // --- STATE ---
    let isTyping = false;
    let currentFile = null;

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
        if (themeBtn) {
            themeBtn.innerHTML = `<i data-lucide="${iconName}"></i>`;
            if(window.lucide) window.lucide.createIcons();
        }
    };

    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const current = htmlEl.getAttribute('data-theme') || 'dark';
            const newTheme = current === 'light' ? 'dark' : 'light';
            
            htmlEl.setAttribute('data-theme', newTheme);
            localStorage.setItem('sentinel-theme', newTheme);
            updateThemeIcon(newTheme);
        });
    }

    const savedTheme = localStorage.getItem('sentinel-theme') || 'dark';
    htmlEl.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    // --- 2. MODE SWITCHING ---
    const modeSwitch = document.getElementById('mode-switch');
    
    if (modeSwitch) {
        modeSwitch.addEventListener('change', () => {
             const mode = modeSwitch.value;
             bodyEl.setAttribute('data-mode-active', mode);
             
             // Refresh history for the selected mode
             loadHistory();
             feed.innerHTML = '';
             showWelcomeMessage();
        });
        // Set initial
        bodyEl.setAttribute('data-mode-active', modeSwitch.value);
    }

    // --- 3. INPUT HANDLING ---
    const subModeBtns = document.querySelectorAll('.sub-mode-btn');
    const roundsParam = document.getElementById('rounds-control');
    const roundsInput = document.getElementById('rounds-input');
    const killSwitchBtn = document.getElementById('kill-switch-btn');
    
    if (subModeBtns.length > 0) {
        subModeBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                // Update UI state
                subModeBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                const selected = btn.getAttribute('data-sub');
                experimentalMode = selected;
                
                updateControlsVisibility();
            });
        });
    }
    
    if (roundsInput) {
        roundsInput.addEventListener('change', (e) => {
            rounds = parseInt(e.target.value) || 6;
        });
    }
    
    if (killSwitchBtn) {
        killSwitchBtn.addEventListener('click', () => {
            killSwitch = !killSwitch;
            const label = killSwitchBtn.querySelector('span');
            if (label) label.textContent = `KILL_SWITCH: ${killSwitch ? 'ON' : 'OFF'}`;
            
            if (killSwitch) {
                killSwitchBtn.classList.add('active');
                killSwitchBtn.style.color = '#fff';
            } else {
                killSwitchBtn.classList.remove('active');
                killSwitchBtn.style.color = '#f87171';
            }
        });
    }
    
    function updateControlsVisibility() {
        // Rounds: Only for 'experimental' sub-mode
        if (roundsParam) {
            if (experimentalMode === 'experimental') {
                roundsParam.style.display = 'flex';
                roundsParam.classList.remove('hidden');
            } else {
                roundsParam.style.display = 'none';
                roundsParam.classList.add('hidden');
            }
        }
        
        // Kill Switch: Only for 'forensic'
        // (App.js logic says forensic has kill switch, experimental has rounds)
        if (killSwitchBtn) {
            if (experimentalMode === 'forensic') {
                killSwitchBtn.style.display = 'flex';
                killSwitchBtn.classList.remove('hidden');
            } else {
                killSwitchBtn.style.display = 'none';
                killSwitchBtn.classList.add('hidden');
            }
        }
    }
    
    // Initialize sub-mode active state
    const initialActive = document.querySelector('.sub-mode-btn.active');
    if (initialActive) {
        experimentalMode = initialActive.getAttribute('data-sub');
        updateControlsVisibility();
    }

    // --- 3. INPUT HANDLING ---
    if (input) {
        input.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
            checkInputState();
        });

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }
    
    function checkInputState() {
        if (!input || !sendBtn) return;
        const hasContent = input.value.trim().length > 0;
        const hasFile = currentFile !== null;
        
        if (hasContent || hasFile) {
            sendBtn.disabled = false;
            sendBtn.classList.add('ready');
        } else {
            sendBtn.disabled = true;
            sendBtn.classList.remove('ready');
        }
    }

    // --- 3b. FILE UPLOAD HANDLING ---
    if (attachBtn && fileInput) {
        attachBtn.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                currentFile = e.target.files[0];
                showFilePreview(currentFile);
            } else {
                clearFile();
            }
            checkInputState();
        });
    }

    function showFilePreview(file) {
        filePreviewContainer.innerHTML = `
            <div class="file-tag" style="display: inline-flex; align-items: center; background: var(--bg-secondary); padding: 4px 8px; border-radius: 4px; margin-bottom: 8px; font-size: 0.85rem;">
                <i data-lucide="file" style="width: 14px; height: 14px; margin-right: 6px;"></i>
                <span class="filename">${file.name}</span>
                <button class="remove-file" id="remove-file-btn" style="background: none; border: none; margin-left: 8px; cursor: pointer; color: var(--text-muted);">
                    <i data-lucide="x" style="width: 14px; height: 14px;"></i>
                </button>
            </div>
        `;
        filePreviewContainer.classList.remove('hidden');
        if(window.lucide) window.lucide.createIcons();

        document.getElementById('remove-file-btn').addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent triggering other clicks
            clearFile();
            // Clear the actual input value so the same file can be selected again
            fileInput.value = '';
        });
    }

    function clearFile() {
        currentFile = null;
        fileInput.value = ''; // Reset input
        filePreviewContainer.innerHTML = '';
        filePreviewContainer.classList.add('hidden');
        checkInputState();
    }


    // --- 4. HISTORY HANDLING ---
    async function loadHistory() {
        if (!historyList) return;
        const currentMode = modeSwitch ? modeSwitch.value : 'standard';
        
        try {
            const res = await fetch('http://localhost:8000/api/history');
            const data = await res.json();
            historyList.innerHTML = '';
            
            // Sort by timestamp desc if needed, but usually backend does it
            const runs = data.runs || data; // Handle if wrapped
            
            if (Array.isArray(runs)) {
                // Filter by mode
                const filteredRuns = runs.filter(item => item.mode === currentMode);
                
                filteredRuns.forEach(item => {
                    const li = document.createElement('li');
                    li.className = 'history-item';
                    
                    // Format display
                    const dateStr = item.timestamp || item.created_at ? new Date(item.timestamp || item.created_at).toLocaleString() : "Unknown Date";
                    const previewText = item.preview || item.chat_name || "No content";
                    
                    li.innerHTML = `
                        <i data-lucide="message-square"></i>
                        <div class="history-content">
                            <span class="history-preview">${previewText}</span>
                            <span class="history-date">${dateStr}</span>
                        </div>
                    `;
                    li.onclick = () => loadLog(item.id || item.filename);
                    historyList.appendChild(li);
                });
            }
            if(window.lucide) window.lucide.createIcons();
        } catch (err) {
            console.error("Failed to load history", err);
        }
    }

    async function loadLog(filename) {
        try {
            const res = await fetch(`http://localhost:8000/api/history/${filename}`);
            const data = await res.json();
            
            feed.innerHTML = ''; // Selectively clear feed for log viewing
            
            appendSystemMessage(`Log Viewer: Loaded \`${filename}\``);
            
            // NEW FORMAT (Database)
            if (data.messages && Array.isArray(data.messages)) {
                data.messages.forEach(msg => {
                    if (msg.role === 'user') {
                        appendMessage('user', msg.content);
                    } else if (msg.role === 'assistant' || msg.role === 'ai') {
                        appendMessage('ai', msg.content, true);
                    } else if (msg.role === 'system') {
                        appendSystemMessage(msg.content);
                    }
                });
                return; // Done
            }

            // OLD FORMAT (Files)
            // If the log has input/output structure
            if (data.input) {
                appendMessage('user', typeof data.input === 'string' ? data.input : JSON.stringify(data.input));
            } else {
                appendSystemMessage("No explicit input found in log.");
            }
            
            if (data.analysis) {
                 appendMessage('ai', data.analysis, true); 
            } else if (data.response) {
                 appendMessage('ai', data.response, true);
            } else {
                 appendMessage('ai', `<pre><code>${JSON.stringify(data, null, 2)}</code></pre>`, true);
            }
            
        } catch (err) {
            console.error(err);
        }
    }

    if (newChatBtn) {
        newChatBtn.addEventListener('click', () => {
            feed.innerHTML = '';
            showWelcomeMessage();
        });
    }

    function showWelcomeMessage() {
        const welcome = document.createElement('div');
        welcome.className = 'message-row ai-row';
        welcome.innerHTML = `
            <div class="msg-avatar ai">S</div>
            <div class="msg-content">
                <p><strong>SYSTEM READY.</strong></p>
                <p>Select a mode and begin.</p>
            </div>
        `;
        feed.appendChild(welcome);
    }

    // --- 5. MESSAGE LOGIC ---
    async function sendMessage() {
        const text = input.value.trim();
        
        // Allow sending if there is text OR a file
        if ((!text && !currentFile) || isTyping) return;

        isTyping = true;
        
        // Display user message in UI
        let displayHtml = '';
        if (currentFile) {
            displayHtml += `<div class="file-attachment" style="margin-bottom: 5px; font-weight: bold;"><i data-lucide="file"></i> Attached: ${currentFile.name}</div>`;
        }
        if (text) {
            displayHtml += `<div>${text.replace(/\n/g, '<br>')}</div>`;
        }
        appendMessage('user', displayHtml, true); 
        
        // Prepare to send
        const messageToSend = text;
        const fileToSend = currentFile; 
        
        // Reset UI immediately
        input.value = '';
        input.style.height = 'auto';
        clearFile(); // Clears currentFile variable and UI
        
        if (sendBtn) {
            sendBtn.classList.remove('ready');
            sendBtn.disabled = true;
        }

        const loadingId = addLoadingIndicator();


        try {
            const mode = modeSwitch ? modeSwitch.value : 'standard';
            
            // USE FORMDATA NOW
            const formData = new FormData();
            
            // Backend expects 'text' for content, not 'message'. 
            // Standard: text: Form(None)
            // Experimental: text: Form(None)
            formData.append('text', messageToSend || `[Attached File: ${fileToSend ? fileToSend.name : "Unknown"}]`);
            
            if (fileToSend) {
                formData.append('file', fileToSend);
            }

            // Determine correct endpoint based on mode
            // Standard: /run/standard
            // Experimental: /run/experimental
            let endpoint = '/run/standard';
            
            if (mode === 'standard') {
                endpoint = '/run/standard';
            } else if (mode === 'sigma' || mode === 'experimental') { // Handle 'sigma' as alias for experimental
                endpoint = '/run/experimental';
                // Add sub-mode parameter if needed. Defaulting to 'full' for now based on previous context.
                // If you want UI for sub-modes, we'd need to add that.
                formData.append('mode', 'full'); 
            }

            // Ensure we hit the backend port (8000)
            // Since this script runs on frontend (3000), we need full URL
            const apiUrl = `http://localhost:8000${endpoint}`;

            const response = await fetch(apiUrl, {
                method: 'POST',
                body: formData 
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`Server error: ${response.status} - ${errText}`);
            }

            const data = await response.json();
            
            removeLoadingIndicator(loadingId);
            
            if (data.error) {
                 appendSystemMessage(`Error: ${data.error}`);
            } else {
                // STABILIZATION UPDATE: Handle Strict Output Contract
                // Structure: { chat_id, chat_name, mode, data: {}, metadata: {} }
                
                const payload = data.data || {}; // Extract payload data
                const metadata = data.metadata || {};

                let content = "";
                let analysisData = null;
                let runId = data.chat_id || data.id;

                if (mode === 'standard') {
                    content = payload.priority_answer || "No response generated.";
                } else {
                    // Experimental mode
                    content = payload.priority_answer || payload.summary || JSON.stringify(payload, null, 2);
                    analysisData = payload; // or whole data?
                    
                    // Specific handling for experimental output format if strict
                    if (payload.metrics) {
                         content = `**Experimental Analysis Complete**\n\n` + 
                                   `**HFI Score:** ${payload.metrics.HFI.toFixed(2)}\n` + 
                                   `**Rounds:** ${metadata.rounds_executed}\n\n` +
                                   `*Check 'Forensic Log' for details.*`;
                    }
                }
                
                // Update chat history list proactively if needed
                // But generally handled by refresh or backend query
                
                // Append content
                let finalContent = content;
                if (typeof marked !== 'undefined') {
                    // If content is object, stringify first?
                    if (typeof content === 'object') {
                        finalContent = `<pre>${JSON.stringify(content, null, 2)}</pre>`;
                    } else {
                         finalContent = marked.parse(content);
                    }
                } else {
                    finalContent = `<div style="white-space: pre-wrap">${content}</div>`;
                }
                
                // Add Experimental Details Button if available
                let extraHtml = "";
                if (analysisData) {
                    const jsonStr = encodeURIComponent(JSON.stringify(analysisData, null, 2));
                    extraHtml += `
                        <div style="margin-top: 10px;">
                            <details>
                                <summary style="cursor: pointer; color: var(--text-accent); font-size: 0.8em;">View Forensic Data</summary>
                                <pre style="font-size: 0.7em; background: rgba(0,0,0,0.3); padding: 10px; overflow: auto; max-height: 300px;">${JSON.stringify(analysisData, null, 2)}</pre>
                            </details>
                        </div>
                    `;
                }

                // Add Feedback Buttons (only if runId exists)
                if (runId) {
                    extraHtml += `
                        <div class="feedback-box" style="margin-top: 8px; display: flex; align-items: center; gap: 8px;">
                            <button class="feedback-btn" onclick="sendFeedback('${runId}', 'up', this)" title="Helpful" style="background:none; border:none; cursor:pointer; color:inherit; opacity:0.7;">
                                <i data-lucide="thumbs-up" style="width: 14px; height: 14px;"></i>
                            </button>
                            <button class="feedback-btn" onclick="sendFeedback('${runId}', 'down', this)" title="Not Helpful" style="background:none; border:none; cursor:pointer; color:inherit; opacity:0.7;">
                                <i data-lucide="thumbs-down" style="width: 14px; height: 14px;"></i>
                            </button>
                        </div>
                    `;
                }
                
                appendMessage('ai', finalContent + extraHtml, true);
                
                // Refresh log list after a moment
                setTimeout(loadHistory, 1000);
            }

        } catch (e) {
            removeLoadingIndicator(loadingId);
            appendSystemMessage(`Network Error: ${e.message}`);
            console.error(e);
        }
        
        isTyping = false;
        // loadHistory(); 
    }

    function appendMessage(role, content, isHtml = false) {
        const row = document.createElement('div');
        row.className = `message-row ${role}-row`;
        
        const avatarClass = role === 'user' ? 'user' : 'ai';
        const avatarLetter = role === 'user' ? 'U' : 'S';
        
        row.innerHTML = `
            <div class="msg-avatar ${avatarClass}">${avatarLetter}</div>
            <div class="msg-content"></div>
        `;

        feed.appendChild(row);

        const contentDiv = row.querySelector('.msg-content');
        
        if (isHtml) {
            contentDiv.innerHTML = content;
        } else {
            contentDiv.innerText = content;
        }

        if(window.lucide) window.lucide.createIcons();
        scrollToBottom();
    }
    
    function appendSystemMessage(text) {
        appendMessage('ai', `**SYSTEM**: ${text}`, false);
    }

    function addLoadingIndicator() {
        const id = 'loading-' + Date.now();
        const loader = document.createElement('div');
        loader.id = id;
        loader.className = 'message-row ai-row loading-indicator';
        loader.innerHTML = `
            <div class="msg-avatar ai">S</div>
            <div class="msg-content typing-dots" style="color: var(--text-muted);">
                PROCESSING<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
            </div>
        `;
        feed.appendChild(loader);
        
        const dots = loader.querySelectorAll('.dot');
        let dotStep = 0;
        const interval = setInterval(() => {
            dots.forEach(d => d.style.opacity = 0.3);
            dots[dotStep].style.opacity = 1;
            dotStep = (dotStep + 1) % 3;
        }, 300);
        
        loader.dataset.interval = interval;
        scrollToBottom();
        return id;
    }

    function removeLoadingIndicator(id) {
        const el = document.getElementById(id);
        if (el) {
            if(el.dataset.interval) clearInterval(el.dataset.interval);
            el.remove();
        }
    }

    function scrollToBottom() {
        feed.scrollTo({
            top: feed.scrollHeight,
            behavior: 'smooth'
        });
    }

    // --- 6. FEEDBACK LOGIC ---
    window.sendFeedback = async function(runId, type, btnElement) {
        if(!runId) return;
        
        // Visual Update
        const container = btnElement.parentElement;
        const allBtns = container.querySelectorAll('.feedback-btn');
        allBtns.forEach(b => b.classList.remove('active'));
        btnElement.classList.add('active');
        
        const status = container.querySelector('.feedback-status');
        if(status) {
            status.innerHTML = "Processing..."; 
            status.style.opacity = '1';
        }
        
        try {
            const formData = new FormData();
            formData.append('run_id', runId);
            formData.append('feedback', type);

            const res = await fetch('http://localhost:8000/feedback', {
                method: 'POST',
                body: formData // Browser sets boundary
            });
            
            if(res.ok) {
                if(status) status.innerHTML = "Feedback Recorded.";
            } else {
                if(status) status.innerHTML = "Failed.";
            }
        } catch(e) {
            console.error(e);
            if(status) status.innerHTML = "Error.";
        }
        
        setTimeout(() => {
            if(status) status.style.opacity = '0';
        }, 2000);
    };

    // Initialize
    loadHistory();
    if (feed.children.length === 0) showWelcomeMessage();
    if(window.lucide) window.lucide.createIcons();
});