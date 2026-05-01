"""
============================================================
INTEGRATION GUIDE — Neon Refactor v5.0
============================================================

This guide explains how to integrate all new components
into your existing FastAPI application.

Timeline:
  Phase A: Database setup (30 min)
  Phase B: API integration (45 min)
  Phase C: Frontend updates (60 min)
  Phase D: Testing (60 min)
  Total: ~3 hours for full integration

============================================================
PHASE A: DATABASE SETUP (30 MIN)
============================================================

1. Create Neon PostgreSQL database:
   
   • Go to https://neon.tech
   • Create new project
   • Copy connection string: postgresql://user:pass@host/db
   • Add to .env: DATABASE_URL=<connection_string>

2. Install dependencies (already in requirements.txt):
   
   • sqlalchemy>=2.0.0
   • asyncpg>=0.29.0
   • alembic>=1.13.0
   • pydantic>=2.0.0

3. Run Alembic migrations:
   
   cd backend/
   alembic upgrade head
   
   This applies: 001_normalize_neon_schema.py
   
   ✓ Creates tables: users, sessions, chats, messages, memory, etc.
   ✓ Adds proper foreign keys
   ✓ Creates indexes

4. Verify database connection:
   
   python -c \"from database.connection_v2 import check_db_connection; import asyncio; asyncio.run(check_db_connection())\"
   
   Should print: ✓ Database connection successful

============================================================
PHASE B: API INTEGRATION (45 MIN)
============================================================

1. Update main.py imports:
   
   # OLD:
   from database.connection import get_db, init_db
   from database.crud import create_chat, add_message, ...
   from gateway.auth import get_current_user
   
   # NEW:
   from database.connection_v2 import get_db, init_db, check_db_connection, check_db_health, init_redis
   from api.endpoints_v2 import router as api_v2_router
   from gateway.auth_v2 import get_current_user, ensure_user_exists, check_auth_setup

2. Initialize Neon on startup:
   
   @app.on_event(\"startup\")
   async def startup_events():
       print(\"[Startup] Checking database...\")
       connected = await check_db_connection()
       if not connected:
           print(\"✗ Cannot connect to database. Exiting.\")
           exit(1)
       
       print(\"[Startup] Initializing Redis...\")
       await init_redis()
       
       print(\"[Startup] Auth configuration:\", await check_auth_setup())
       
       print(\"✓ Server ready\")

3. Register new API router:
   
   # In main.py, after creating app:
   app.include_router(api_v2_router, prefix=\"/api\", tags=[\"API v2\"])

4. Add health endpoint:
   
   @app.get(\"/health\")
   async def health_check():
       db_health = await check_db_health()
       return {
           \"status\": \"healthy\",
           \"database\": db_health,
       }

5. Update lifespan / shutdown:
   
   @app.on_event(\"shutdown\")
   async def shutdown_events():
       from database.connection_v2 import close_db, close_redis
       await close_db()
       await close_redis()
       print(\"✓ Shutdown complete\")

============================================================
PHASE C: FRONTEND UPDATES (60 MIN)
============================================================

Key files to update:
  • frontend/src/stores/ — Redux/Zustand state management
  • frontend/src/services/api.js — API client
  • frontend/src/pages/ChatPage.js — Chat component
  • frontend/src/App.js — Root app

1. Update API client (frontend/src/services/api.js):
   
   const API_BASE = process.env.REACT_APP_API_URL || \"http://localhost:8000/api\";
   
   // Create session on app load
   export async function createSession(client = \"web\") {
       const res = await fetch(`${API_BASE}/session\", {
           method: \"POST\",
           headers: { \"Authorization\": `Bearer ${token}` },
           body: JSON.stringify({ client }),
       });
       return res.json();
   }
   
   // Load chat history
   export async function loadHistory(limit = 50) {
       const res = await fetch(`${API_BASE}/history?limit=${limit}`, {
           headers: { \"Authorization\": `Bearer ${token}` },
       });
       const data = await res.json();
       if (!data.success) return { chats: [] };  // Always return chats array
       return data.data;
   }
   
   // Create chat
   export async function createChat(title = \"New Chat\") {
       const res = await fetch(`${API_BASE}/chat\", {
           method: \"POST\",
           headers: { \"Authorization\": `Bearer ${token}` },
           body: JSON.stringify({ title }),
       });
       return res.json();
   }
   
   // Get chat
   export async function getChat(chatId) {
       const res = await fetch(`${API_BASE}/chat/${chatId}\", {
           headers: { \"Authorization\": `Bearer ${token}` },
       });
       return res.json();
   }
   
   // Send message
   export async function sendMessage(chatId, role, content) {
       const res = await fetch(`${API_BASE}/chat/${chatId}/message\", {
           method: \"POST\",
           headers: { \"Authorization\": `Bearer ${token}` },
           body: JSON.stringify({ role, content }),
       });
       return res.json();
   }
   
   // Get memory
   export async function getMemory() {
       const res = await fetch(`${API_BASE}/memory\", {
           headers: { \"Authorization\": `Bearer ${token}` },
       });
       return res.json();
   }

2. Update state management (frontend/src/stores/chatStore.js):
   
   // Zustand store example
   import create from 'zustand';
   
   export const useChatStore = create((set, get) => ({
       // State
       chats: [],
       currentChat: null,
       sessionId: null,
       loading: false,
       error: null,
       
       // Actions
       loadHistory: async (token) => {
           set({ loading: true });
           try {
               const data = await api.loadHistory();
               if (data && data.chats) {
                   set({ chats: data.chats, error: null });
               } else {
                   set({ chats: [], error: null });  // Empty, not error
               }
           } catch (err) {
               set({ error: err.message });
               set({ chats: [] });  // Keep previous state
           }
           set({ loading: false });
       },
       
       selectChat: (chatId) => {
           set({ currentChat: chatId });
       },
       
       addMessage: (chatId, message) => {
           set(state => ({
               chats: state.chats.map(chat =>
                   chat.id === chatId
                       ? { ...chat, messages: [...(chat.messages || []), message] }
                       : chat
               )
           }));
       },
   }));

3. Update App.js (on load):
   
   useEffect(() => {
       const initializeApp = async () => {
           // Wait for auth
           const { isLoaded, isSignedIn, sessionId } = useAuth();
           if (!isLoaded) return;
           
           if (!isSignedIn) {
               navigate(\"/login\");
               return;
           }
           
           // Get token
           const token = await getToken();
           
           // Create session
           const sessionRes = await api.createSession(\"web\");
           if (sessionRes.success) {
               setSessionId(sessionRes.data.session_id);
           }
           
           // Load chat history
           const historyRes = await api.loadHistory();
           if (historyRes.success && historyRes.data.chats) {
               setChatHistory(historyRes.data.chats);
           } else {
               setChatHistory([]);  // Empty array, not null
           }
       };
       
       initializeApp();
   }, [isLoaded]);

4. Update chat page (on refresh):
   
   // When user opens /chat/:id
   useEffect(() => {
       const loadChat = async () => {
           const chatRes = await api.getChat(chatId);
           if (chatRes.success) {
               setChat(chatRes.data);
               setMessages(chatRes.data.messages || []);
           } else {
               // Chat not found or error
               setError(chatRes.error?.message || \"Failed to load chat\");
               setMessages([]);
           }
       };
       
       loadChat();
   }, [chatId]);

5. Update message sending:
   
   const sendMessage = async (content) => {
       try {
           const res = await api.sendMessage(chatId, \"user\", content);
           if (res.success) {
               // Add to state
               addMessage(chatId, res.data);
           } else {
               // Failed, show error but keep state
               console.error(res.error?.message);
           }
       } catch (err) {
           console.error(err);
       }
   };

============================================================
PHASE D: TESTING (60 MIN)
============================================================

1. Manual testing checklist:

   ✓ Login → Database creates user
   ✓ Create chat → Chat persists in DB
   ✓ Send message → Message persists in DB
   ✓ Refresh page → Chat history loads
   ✓ Logout → Session ends
   ✓ Login again → Same chats visible
   ✓ Copy text → Works without error
   ✓ No white screen on error

2. Database validation:

   psql -d <database_url>
   
   SELECT COUNT(*) FROM users;  -- Should see your user
   SELECT COUNT(*) FROM chats WHERE user_id = '<your_id>';
   SELECT COUNT(*) FROM messages WHERE user_id = '<your_id>';

3. API testing:

   # Create session
   curl -X POST http://localhost:8000/api/session \\
        -H \"Authorization: Bearer <token>\" \\
        -H \"Content-Type: application/json\"
   
   # Load history
   curl -X GET http://localhost:8000/api/history \\
        -H \"Authorization: Bearer <token>\"
   
   # Create chat
   curl -X POST http://localhost:8000/api/chat \\
        -H \"Authorization: Bearer <token>\" \\
        -H \"Content-Type: application/json\" \\
        -d '{\"title\": \"Test Chat\"}'

============================================================
MIGRATION PATH (ZERO DOWNTIME)
============================================================

The new schema coexists with old schema:

1. Deploy new code with BOTH old and new endpoints
2. Frontend gradually migrates to new /api/v2 endpoints
3. Old endpoints remain available until fully migrated
4. Once stable, deprecate old endpoints

This ensures:
  • Zero downtime
  • Gradual migration
  • Easy rollback if needed

============================================================
TROUBLESHOOTING
============================================================

Problem: \"Database connection failed\"
Solution:
  • Verify DATABASE_URL is set in .env
  • Check Neon connection string
  • Ensure firewall allows connection
  • Run: python -c \"from database.connection_v2 import check_db_connection; import asyncio; asyncio.run(check_db_connection())\"

Problem: \"Tables don't exist\"
Solution:
  • Run Alembic migration: alembic upgrade head
  • Check migration file: backend/alembic/versions/001_normalize_neon_schema.py
  • Verify database:psql -d <url> -c \"\\dt\"

Problem: \"Auth token invalid\"
Solution:
  • Verify CLERK_SECRET_KEY is set in .env
  • Check token format: Bearer <token>
  • Test token decoding in auth_v2.py

Problem: \"Chat history not loading\"
Solution:
  • Check /api/history returns success=true
  • Verify chats array exists (never null)
  • Check database for chats: SELECT * FROM chats WHERE user_id = '<id>';

Problem: \"Frontend shows blank page\"
Solution:
  • Check browser console for errors
  • Verify API responses have {success, data, error}
  • Ensure empty arrays [] instead of null
  • Test API endpoints directly with curl

============================================================
NEXT STEPS
============================================================

After integration, consider:

1. Context Window Builder (PHASE 5)
   • Combine recent messages + memory
   • Enforce token limits
   • Deterministic ordering

2. Memory System (PHASE 6)
   • Extract facts from messages
   • Upsert to memory table
   • Weight by recency/frequency

3. Visual/Metadata Handling (PHASE 7)
   • Store image URLs (not base64)
   • Extract metadata
   • Support search/filtering

4. Production Deployment
   • Configure Render environment variables
   • Set up Vercel frontend deployment
   • Enable monitoring/logging
   • Set up backups for Neon

============================================================
RESOURCES
============================================================

Documentation:
  • Neon: https://neon.tech/docs
  • SQLAlchemy Async: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
  • FastAPI: https://fastapi.tiangolo.com/
  • Alembic: https://alembic.sqlalchemy.org/
  • Clerk Auth: https://clerk.com/docs/quickstarts/nextjs

Files created:
  • backend/database/models_v2.py — Normalized schema
  • backend/database/crud_v2.py — CRUD operations
  • backend/database/connection_v2.py — Connection pooling
  • backend/gateway/auth_v2.py — Auth integration
  • backend/api/endpoints_v2.py — API endpoints
  • backend/utils/safe_responses.py — Response builders
  • backend/alembic/versions/001_normalize_neon_schema.py — Migration

Test files:
  • backend/tests/test_crud_v2.py — CRUD unit tests (create if needed)
  • backend/tests/test_api_v2.py — API integration tests (create if needed)
  • frontend/src/__tests__/api.test.js — Frontend tests (create if needed)

============================================================
"""
