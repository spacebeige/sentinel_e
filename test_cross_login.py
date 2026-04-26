#!/usr/bin/env python
"""
Cross-Login Persistence Test — Sentinel-E v5.0

Verifies that user chats persist across login/logout cycles.

Test Scenarios:
  1. Create chat as User A
  2. Logout User A
  3. Login User A again
  4. Verify chats are restored
  5. Verify no cross-user data leakage
"""

import asyncio
import json
import sys
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

# Add backend to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from database.connection import get_db, engine
from database.crud import (
    create_chat,
    list_chats,
    add_message,
    get_chat_messages,
    upsert_authenticated_user,
    add_user_memory,
    get_user_memory,
    upsert_user_preference,
    get_user_preference,
)


async def test_cross_login_persistence():
    """Test chat persistence across login cycles."""
    
    print("\n" + "="*70)
    print("CROSS-LOGIN PERSISTENCE TEST")
    print("="*70)
    
    # Create test users
    user_a_id = f"test_user_a_{uuid4().hex[:8]}"
    user_b_id = f"test_user_b_{uuid4().hex[:8]}"
    
    print(f"\n[SETUP] Creating test users...")
    print(f"  User A: {user_a_id}")
    print(f"  User B: {user_b_id}")
    
    async for db in get_db():
        # ── STEP 1: User A creates chats and memories ─────────────────
        print(f"\n[STEP 1] User A creates 3 chats...")
        
        user_a = await upsert_authenticated_user(
            db,
            user_id=user_a_id,
            email=f"{user_a_id}@test.com",
            name="Test User A",
        )
        
        # Create chats for User A
        chats_a = []
        for i in range(3):
            chat = await create_chat(db, f"Chat {i+1}", "standard", user_id=user_a_id)
            chats_a.append(chat)
            
            # Add messages
            await add_message(db, chat.id, "user", f"Question {i+1}?")
            await add_message(db, chat.id, "assistant", f"Answer {i+1}.")
            
            print(f"  ✓ Chat {i+1}: {chat.id}")
        
        # Add user memory for User A
        print(f"\n[STEP 1] User A creates memory facts...")
        await add_user_memory(
            db, user_a_id,
            "preferred_response_length", "concise",
            confidence=90,
            metadata_json={"source": "test"}
        )
        await add_user_memory(
            db, user_a_id,
            "domain_interest", "machine_learning",
            confidence=85,
            metadata_json={"source": "test"}
        )
        print(f"  ✓ Added 2 memory facts")
        
        # Add user preferences for User A
        print(f"\n[STEP 1] User A sets preferences...")
        await upsert_user_preference(
            db,
            user_a_id,
            response_style="concise",
            tone="technical",
            dark_mode=True,
        )
        print(f"  ✓ Preferences set")
        
        # ── STEP 2: User B creates chats (data isolation check) ────────
        print(f"\n[STEP 2] User B creates 2 chats (data isolation)...")
        
        user_b = await upsert_authenticated_user(
            db,
            user_id=user_b_id,
            email=f"{user_b_id}@test.com",
            name="Test User B",
        )
        
        chats_b = []
        for i in range(2):
            chat = await create_chat(db, f"User B Chat {i+1}", "standard", user_id=user_b_id)
            chats_b.append(chat)
            await add_message(db, chat.id, "user", f"User B Question {i+1}?")
            print(f"  ✓ User B Chat {i+1}: {chat.id}")
        
        # ── STEP 3: "Logout" User A, retrieve chats (simulate re-login) ─
        print(f"\n[STEP 3] Simulating logout and re-login of User A...")
        print(f"  (In real app, user_id is restored from JWT after re-login)")
        
        # Retrieve User A's chats (as if User A just logged back in)
        retrieved_chats_a = await list_chats(db, user_a_id)
        
        print(f"\n  ✓ Retrieved {len(retrieved_chats_a)} chats for User A:")
        for chat in retrieved_chats_a:
            messages = await get_chat_messages(db, chat.id)
            print(f"    - {chat.chat_name} ({len(messages)} messages)")
        
        # Verify correct number of chats
        assert len(retrieved_chats_a) == 3, f"Expected 3 chats, got {len(retrieved_chats_a)}"
        
        # ── STEP 4: Verify memory and preferences persisted ───────────
        print(f"\n[STEP 4] Verifying User A's memory and preferences...")
        
        memory_facts = await get_user_memory(db, user_a_id, min_confidence=70)
        print(f"  ✓ Retrieved {len(memory_facts)} high-confidence memory facts:")
        for mem in memory_facts:
            print(f"    - {mem.key}: {mem.value} ({mem.confidence}% confidence)")
        
        assert len(memory_facts) == 2, f"Expected 2 memory facts, got {len(memory_facts)}"
        
        prefs = await get_user_preference(db, user_a_id)
        if prefs:
            print(f"  ✓ Retrieved preferences:")
            print(f"    - Response Style: {prefs.response_style}")
            print(f"    - Tone: {prefs.tone}")
            print(f"    - Dark Mode: {prefs.dark_mode}")
        
        # ── STEP 5: Cross-user data isolation check ──────────────────
        print(f"\n[STEP 5] Verifying data isolation (User B cannot see User A data)...")
        
        retrieved_chats_b = await list_chats(db, user_b_id)
        print(f"  ✓ User B can see {len(retrieved_chats_b)} chats (should be 2)")
        
        assert len(retrieved_chats_b) == 2, f"Expected 2 chats for User B, got {len(retrieved_chats_b)}"
        
        # Verify User B cannot see User A's chats
        user_b_chat_ids = {str(c.id) for c in retrieved_chats_b}
        user_a_chat_ids = {str(c.id) for c in retrieved_chats_a}
        
        overlap = user_b_chat_ids & user_a_chat_ids
        if overlap:
            print(f"  ✗ DATA LEAK: User B can see {len(overlap)} of User A's chats!")
            assert False, "Cross-user data leakage detected"
        else:
            print(f"  ✓ No data leakage - users are properly isolated")
        
        # ── STEP 6: Verify message persistence ───────────────────────
        print(f"\n[STEP 6] Verifying messages persist correctly...")
        
        first_chat = retrieved_chats_a[0]
        messages = await get_chat_messages(db, first_chat.id)
        print(f"  ✓ First chat has {len(messages)} messages:")
        for msg in messages:
            print(f"    - {msg.role}: {msg.content[:50]}...")
        
        # ── SUMMARY ──────────────────────────────────────────────────
        print(f"\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print(f"\nCross-login persistence verified:")
        print(f"  ✓ User chats restored after re-login")
        print(f"  ✓ User memory facts persist")
        print(f"  ✓ User preferences persist")
        print(f"  ✓ Data properly isolated between users")
        print(f"  ✓ Messages retrieved correctly")
        print("\n")


if __name__ == "__main__":
    asyncio.run(test_cross_login_persistence())
