import re

with open('src/app/components/ChatPage.tsx', 'r') as f:
    content = f.read()

# 1. Fix imports from ../api
content = content.replace(
    'import { getChatHistory, getChatMessages, getChatDetails } from "../api";',
    'import { getHistory as getChatHistory, getChatMessages, getSessionDescriptive as getChatDetails } from "@services/api";'
)
content = content.replace('} from "../api";', '} from "../types";')

# 2. Fix adaptRunResponse
content = content.replace('import { adaptRunResponse } from "../services/adapter";', '')
content = content.replace('response = adaptRunResponse(rawResponse);', 'response = rawResponse;')

# 3. Fix useSessionPersistence
content = content.replace('import { useSessionPersistence } from "../hooks/useSessionPersistence";', '')
content = content.replace('const { restore, persist, reset: resetSession } = useSessionPersistence();', 'const restore = () => ({}); const persist = () => {}; const resetSession = () => {};')
content = content.replace('useSessionPersistence(currentChatId, messages);', '')

# 4. Fix useSupabaseAuth
content = content.replace('import { useSupabaseAuth } from "../hooks/useSupabaseAuth";', "import { useSupabaseAuth } from '@hooks/useSupabaseAuth';")

with open('src/app/components/ChatPage.tsx', 'w') as f:
    f.write(content)
