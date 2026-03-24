# Session Handling Per-User - DEBUG REPORT

## Status: ❌ CRITICAL BUGS FOUND

Session handling is **NOT properly isolated per user**. Multiple security and data isolation vulnerabilities discovered in the connection pipeline.

---

## ISSUE #1: Session Cache Has NO User Isolation

### Location
[backend/main.py](backend/main.py) - `_get_session()` function (lines 495-525)

### Problem
```python
async def _get_session(chat_id: str, user_id: str = ""):
    """Get or create session pair."""
    global omega_sessions, memory_sessions

    if chat_id in omega_sessions:  # ❌ NO USER VERIFICATION!
        return omega_sessions[chat_id], memory_sessions.get(chat_id, MemoryEngine(user_id=user_id))
```

**The Bug:**
- Function checks `if chat_id in omega_sessions` WITHOUT verifying user ownership
- If User A creates a chat, their OmegaCognitiveKernel is cached
- If User B requests the SAME chat_id, they get User A's cached kernel!
- The `user_id` parameter is ONLY used for new sessions, not validation

**Attack Scenario:**
```
1. User A makes request → run_sentinel → chat_id="chat-123" created
2. Omega kernel cached in omega_sessions["chat-123"]
3. User B makes request with chat_id="chat-123"
4. _get_session returns User A's cached kernel to User B
5. User B can read/modify User A's entire session state
```

### Impact
- 🔴 **CRITICAL**: Cross-user session leakage
- Memory contents exposed between users
- Knowledge learner state shared between users
- Reasoning traces visible across users

---

## ISSUE #2: Database CRUD Has NO User Ownership Verification

### Location
[backend/database/crud.py](backend/database/crud.py)

### Problems

#### Problem 2a: `get_chat()` Doesn't Filter by User
```python
async def get_chat(db: AsyncSession, chat_id: UUID) -> Optional[Chat]:
    result = await db.execute(select(Chat).where(Chat.id == chat_id))
    return result.scalars().first()  # ❌ No user_id filter!
```

**Issue:** Any user can retrieve any chat from the database

#### Problem 2b: `list_chats()` Doesn't Filter by User
```python
async def list_chats(db: AsyncSession, limit: int = 50, offset: int = 0) -> List[Chat]:
    result = await db.execute(
        select(Chat).order_by(Chat.updated_at.desc()).limit(limit).offset(offset)
    )
    return result.scalars().all()  # ❌ No user_id filter!
```

**Issue:** Users can list ALL chats ever created by any user

#### Problem 2c: `get_chat_messages()` Doesn't Verify Chat Ownership
```python
async def get_chat_messages(db: AsyncSession, chat_id: UUID) -> List[Message]:
    result = await db.execute(
        select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at.asc())
    )
    return result.scalars().all()  # ❌ No user ownership check!
```

**Issue:** Any user can read any chat's messages

### Impact
- 🔴 **CRITICAL**: Data theft across users
- Users can list other users' chat history
- Users can read other users' messages
- Privacy completely compromised

---

## ISSUE #3: No User Isolation in Endpoints

### Location
[backend/main.py](backend/main.py) - `run_sentinel()` endpoint (lines 625+)

### Problem
```python
@app.post("/api/run")
async def run_sentinel(
    request: SentinelRequest,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
    frontend_context: Optional[str] = None,
):
    user_id = user["user_id"]
    
    # ❌ NO VERIFICATION that user owns the chat!
    if request.chat_id:
        chat = await get_chat(db, request.chat_id)  # Gets ANY chat, not user's chat
    
    kernel, memory = await _get_session(str(chat.id), user_id)  # ❌ No validation
```

**Issue:**
- `get_chat()` doesn't verify `chat.user_id == user_id`
- If chat_id doesn't exist, new chat created
- But if it DOES exist, ANY user can use it

### Attack Scenario
```
1. User A creates chat "chat-123" with 100 turns of reasoning
2. User B calls /api/run with chat_id="chat-123"
3. Backend fetches chat without user verification
4. User B now has User A's cached session + can continue their conversation
5. User B can see all of User A's previous message history + reasoning
```

---

## ISSUE #4: Session Restore Has NO User Validation

### Location
[backend/main.py](backend/main.py) - `_restore_session()` function (lines 498-516)

### Problem
```python
async def _restore_session(chat_id: str, user_id: str = ""):
    """Restore session from SQLite first, then Redis fallback for backward compatibility."""
    try:
        cached = await asyncio.to_thread(_sqlite_read_session, chat_id)
        if not cached:
            cached = await redis_client.get(f"session:{chat_id}")

        if not cached:
            return None, None

        data = json.loads(cached)
        kernel = OmegaCognitiveKernel.restore_from_session(...)
        memory = MemoryEngine.deserialize(...)
        return kernel, memory
    except Exception as e:
        logger.warning(f"Session restore failed: {e}")
    return None, None
```

**Issue:**
- `user_id` parameter is accepted but NEVER USED
- Any cached session can be restored by providing the chat_id
- No ownership verification before deserialization

---

## ISSUE #5: Firebase Integration Not Blocking Unauthorized Access

### Location
[backend/gateway/firebase_service.py](backend/gateway/firebase_service.py)

### Problem
Even though Firebase is initialized, it's NEVER called to verify session ownership:

```python
# In main.py startup
firebase_service = await asyncio.to_thread(get_firebase_service)
if firebase_is_enabled():
    logger.info("✓ Firebase Admin SDK initialized successfully")

# ❌ But then in run_sentinel(), Firebase is NEVER used to verify chat ownership!
```

**Issue:**
- Firebase has user profiles and sessions stored in Firestore
- But the session retrieval pipeline doesn't check Firestore for authorization
- Firebase verification is optional, not enforced

---

## ISSUE #6: Memory Engine Doesn't Enforce User Isolation

### Location
[backend/memory/memory_engine.py](backend/memory/memory_engine.py)

### Problem
MemoryEngine stores `user_id` but doesn't enforce isolation when deserializing:

```python
# In _get_session:
memory = memory_sessions.get(chat_id, MemoryEngine(user_id=user_id))

# ❌ When cached, memory from different user is returned
# ❌ MemoryEngine.deserialize() doesn't validate user_id
```

---

## ISSUE #7: No Audit Trail for Cross-User Access

### Problem
- No logging when sessions are accessed
- No detection mechanism if User B accesses User A's session
- No alerting if cross-user access occurs

---

## CONNECTION PIPELINE FLOW (CURRENT - BROKEN)

```
┌─────────────────────────────────────────────────────────┐
│ Frontend Request (JWT token or anonymous session)       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ get_current_user() - ✅ Extracts user_id                 │
│  Returns: {user_id, authenticated, token_type}          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ run_sentinel(request, user)                   │
│  - Extracts user_id from JWT ✅               │
│  - Gets chat_id from request                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ get_chat(db, chat_id) ❌❌❌ NO USER FILTER!               │
│  - Queries: SELECT * FROM chats WHERE id = chat_id      │
│  - Returns chat owned by ANY user                       │
│  - NO verification that chat.user_id == current_user_id │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ _get_session(chat_id, user_id) ❌❌❌ NO CACHE VALIDATION! │
│  - Checks: if chat_id in omega_sessions                 │
│  - Returns cached kernel WITHOUT user verification      │
│  - user_id parameter only used for NEW sessions         │
│  - NO check: if cached_kernel.user_id == current_user_id│
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ _restore_session(chat_id, user_id) ❌ NO OWNERSHIP CHECK! │
│  - Reads from SQLite: session_cache[chat_id]           │
│  - Deserializes kernel + memory                        │
│  - NO verification of user_id ownership               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ OmegaCognitiveKernel + MemoryEngine returned            │
│  - Potentially from DIFFERENT user!                    │
│  - Session state fully accessible to wrong user        │
│  - All knowledge + reasoning exposed                   │
└────────────────────────────────────────────────────────┘
```

---

## REQUIRED FIXES

### FIX #1: Add User Ownership Verification to CRUD

**File:** `backend/database/crud.py`

```python
async def get_chat(db: AsyncSession, chat_id: UUID, user_id: str = None) -> Optional[Chat]:
    query = select(Chat).where(Chat.id == chat_id)
    if user_id:
        query = query.where(Chat.user_id == user_id)
    result = await db.execute(query)
    return result.scalars().first()

async def list_chats(db: AsyncSession, user_id: str, limit: int = 50, offset: int = 0) -> List[Chat]:
    result = await db.execute(
        select(Chat)
        .where(Chat.user_id == user_id)  # ✅ ADD USER FILTER
        .order_by(Chat.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()

async def get_chat_messages(db: AsyncSession, chat_id: UUID, user_id: str = None) -> List[Message]:
    query = select(Message).where(Message.chat_id == chat_id)
    # Verify ownership via join
    if user_id:
        query = query.join(Chat).where(Chat.user_id == user_id)
    result = await db.execute(query.order_by(Message.created_at.asc()))
    return result.scalars().all()
```

### FIX #2: Add Ownership Validation to Session Cache

**File:** `backend/main.py`

```python
async def _get_session(chat_id: str, user_id: str = ""):
    """Get or create session pair with user validation."""
    global omega_sessions, memory_sessions

    # ✅ Check cache ownership
    if chat_id in omega_sessions:
        cached_kernel = omega_sessions[chat_id]
        # Verify user_id matches
        if hasattr(cached_kernel, '_owner_user_id'):
            if cached_kernel._owner_user_id != user_id:
                logger.warning(f"Attempted unauthorized session access: {chat_id} by user {user_id}")
                raise HTTPException(status_code=403, detail="Unauthorized chat access")
        
        return cached_kernel, memory_sessions.get(chat_id, MemoryEngine(user_id=user_id))

    # Try persisted cache with user validation
    kernel, memory = await _restore_session(chat_id, user_id)
    if kernel:
        omega_sessions[chat_id] = kernel
        memory_sessions[chat_id] = memory
        return kernel, memory

    # Create new
    _evict_sessions()
    kernel = OmegaCognitiveKernel(...)
    kernel._owner_user_id = user_id  # ✅ Store owner
    memory = MemoryEngine(user_id=user_id)
    omega_sessions[chat_id] = kernel
    memory_sessions[chat_id] = memory
    return kernel, memory
```

### FIX #3: Update Endpoints to Do Authorization Checks

**File:** `backend/main.py`

```python
@app.post("/api/run")
async def run_sentinel(
    request: SentinelRequest,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(get_current_user),
    frontend_context: Optional[str] = None,
):
    user_id = user["user_id"]

    chat = None
    if request.chat_id:
        # ✅ Pass user_id to verify ownership
        chat = await get_chat(db, request.chat_id, user_id=user_id)
        if not chat:
            # Chat doesn't exist or doesn't belong to this user
            raise HTTPException(status_code=403, detail="Chat not found or unauthorized")
    
    if not chat:
        chat_name = generate_chat_name(effective_text, request.mode)
        chat = await create_chat(db, chat_name, request.mode, user_id=user_id)

    # ✅ _get_session will verify user_id
    kernel, memory = await _get_session(str(chat.id), user_id)
```

### FIX #4: Use Firebase to Verify Session Ownership

**File:** `backend/main.py`

```python
async def _get_session(chat_id: str, user_id: str = ""):
    """Get or create session pair with user validation."""
    global omega_sessions, memory_sessions

    # ✅ Optional Firebase verification
    firebase = get_firebase_service() if firebase_is_enabled() else None
    if firebase and firebase.enabled:
        try:
            session_doc = firebase.get_session(chat_id)
            if session_doc and session_doc.get("user_id") != user_id:
                raise HTTPException(status_code=403, detail="Unauthorized session access")
        except Exception as e:
            logger.warning(f"Firebase session check failed: {e}")
    
    # Cache logic with verification...
```

### FIX #5: Add Session Ownership Metadata

**File:** `backend/core/omega_kernel.py` or similar

```python
class OmegaCognitiveKernel:
    def __init__(self, ..., owner_user_id: str = None):
        self._owner_user_id = owner_user_id
        # ... rest of init
    
    def serialize_session(self):
        return {
            "owner_user_id": self._owner_user_id,  # ✅ Include ownership
            "kernel_state": {...},
            # ... other data
        }
    
    @classmethod
    def restore_from_session(cls, data, ..., user_id: str = None):
        # ✅ Verify ownership on restore
        if user_id and data.get("owner_user_id") != user_id:
            raise ValueError(f"Session ownership mismatch")
        
        kernel = cls(...)
        kernel._owner_user_id = data.get("owner_user_id")
        return kernel
```

---

## TESTING RECOMMENDATIONS

### Test 1: Cross-User Session Access
```python
# 1. User A creates chat "chat-123"
# 2. User B tries to access same chat_id
# 3. Should fail with 403 Unauthorized
```

### Test 2: Database Query Isolation
```python
# 1. User A creates Chat X
# 2. User B calls list_chats()
# 3. Should NOT see Chat X
```

### Test 3: Message Privacy
```python
# 1. User A creates chat with 10 messages
# 2. User B calls /api/chat/123/messages
# 3. Should fail with 403
```

### Test 4: Session Restore Verification
```python
# 1. Kill User A's session
# 2. User B tries to restore with chat_id (from User A)
# 3. Should create NEW session, not restore User A's
```

---

## DEPLOYMENT IMPACT

**Current Status:** 🔴 PRODUCTION DANGEROUS

**Risk Level:** CRITICAL

**Recommendation:** 
- ⚠️ DO NOT expose to multiple users until FIXED
- Deploy these fixes BEFORE multi-user deployment
- Add integration tests to prevent regression

---

## Timeline Summary

| Component | Status | Risk |
|-----------|--------|------|
| Auth extraction | ✅ Working | Low |
| Database CRUD | ❌ No user filter | CRITICAL |
| Session cache | ❌ No ownership | CRITICAL |
| Session restore | ❌ No validation | CRITICAL |
| Endpoint auth | ⚠️ Partial | HIGH |
| Firebase integration | ✅ Ready | Low* |
| Memory isolation | ⚠️ Partial | HIGH |

*Firebase ready but not being used for authorization

---

## Files to Update

1. `backend/database/crud.py` - Add user_id filters
2. `backend/main.py` - Add ownership validation
3. `backend/core/omega_kernel.py` - Add owner tracking
4. `backend/memory/memory_engine.py` - Add user validation
5. Unit test files (create)
6. Integration test files (create)

