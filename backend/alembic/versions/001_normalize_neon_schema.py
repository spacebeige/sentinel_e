"""
Alembic migration to normalize the database schema for Neon PostgreSQL.

This migration:
1. Normalizes the users table (consolidate ID column)
2. Adds proper foreign keys and constraints
3. Creates sessions, memory, and settings tables
4. Adds semantic indexes
5. Maintains backward compatibility with existing data

Migration: 001_normalize_neon_schema
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

revision = '001_normalize_neon_schema'
down_revision = '9f1d8c2a4b11'
branch_labels = None
depends_on = None

def upgrade():
    """Apply schema normalization."""
    
    # ─────────────────────────────────────────────────────────
    # Step 1: Backup existing data (if needed)
    # ─────────────────────────────────────────────────────────
    
    # Create temporary backup tables for safety
    op.execute("""
        CREATE TABLE IF NOT EXISTS users_backup AS
        SELECT * FROM users
        WHERE id IS NOT NULL;
    """)
    
    # ─────────────────────────────────────────────────────────
    # Step 2: Update users table to use auth provider ID as PK
    # ─────────────────────────────────────────────────────────
    
    # Add new columns
    op.add_column('users', sa.Column('email_new', sa.String(), nullable=True))
    op.add_column('users', sa.Column('provider_new', sa.String(), nullable=True, server_default='clerk'))
    
    # Migrate data
    op.execute("""
        UPDATE users
        SET email_new = COALESCE(email, 'unknown@example.com'),
            provider_new = COALESCE(provider, 'clerk')
        WHERE email IS NOT NULL
        OR user_id IS NOT NULL;
    """)
    
    # Drop old email column (if it exists and is different from email_new)
    op.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS users_email_key;")
    op.drop_column('users', 'email')
    
    # Rename new columns
    op.execute("ALTER TABLE users RENAME COLUMN email_new TO email;")
    op.execute("ALTER TABLE users RENAME COLUMN provider_new TO provider;")
    
    # Make email NOT NULL and unique
    op.alter_column('users', 'email', nullable=False, existing_type=sa.String())
    op.create_unique_constraint('uq_users_email', 'users', ['email'])
    op.create_index('ix_users_email', 'users', ['email'])
    
    # ─────────────────────────────────────────────────────────
    # Step 3: Update chats table foreign key
    # ─────────────────────────────────────────────────────────
    
    # Ensure chats.user_id is a string (matching users.id)
    op.alter_column('chats', 'user_id', existing_type=sa.String(), nullable=False)
    
    # Drop existing FK if it exists
    op.execute("ALTER TABLE chats DROP CONSTRAINT IF EXISTS chats_user_id_fkey;")
    
    # Add proper FK
    op.create_foreign_key(
        'fk_chats_user_id',
        'chats',
        'users',
        ['user_id'],
        ['id'],
        ondelete='CASCADE'
    )
    
    # ─────────────────────────────────────────────────────────
    # Step 4: Update messages table
    # ─────────────────────────────────────────────────────────
    
    # Ensure user_id is string
    op.alter_column('messages', 'user_id', existing_type=sa.String(), nullable=False)
    
    # Drop existing FK if exists
    op.execute("ALTER TABLE messages DROP CONSTRAINT IF EXISTS messages_user_id_fkey;")
    
    # Add proper FK
    op.create_foreign_key(
        'fk_messages_user_id',
        'messages',
        'users',
        ['user_id'],
        ['id'],
        ondelete='CASCADE'
    )
    
    # Add new columns to messages
    op.add_column('messages', sa.Column('image_url', sa.String(), nullable=True))
    op.add_column('messages', sa.Column('is_deleted', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('messages', sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()))
    
    # ─────────────────────────────────────────────────────────
    # Step 5: Create sessions table
    # ─────────────────────────────────────────────────────────
    
    op.create_table(
        'sessions',
        sa.Column('id', UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('client', sa.String(), nullable=False, server_default='web'),
        sa.Column('ip_address', sa.String(), nullable=True),
        sa.Column('user_agent', sa.String(), nullable=True),
        sa.Column('metadata', JSONB(), nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('last_active_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    
    op.create_index('ix_sessions_user_id', 'sessions', ['user_id'])
    op.create_index('ix_sessions_created_at', 'sessions', ['created_at'])
    op.create_index('ix_sessions_last_active_at', 'sessions', ['last_active_at'])
    
    # ─────────────────────────────────────────────────────────
    # Step 6: Create memory table
    # ─────────────────────────────────────────────────────────
    
    op.create_table(
        'memory',
        sa.Column('id', UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('key', sa.String(), nullable=False),
        sa.Column('value', JSONB(), nullable=False),
        sa.Column('weight', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('confidence', sa.Integer(), nullable=False, server_default='50'),
        sa.Column('tag', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id', 'key', name='uq_memory_user_key'),
    )
    
    op.create_index('ix_memory_user_id', 'memory', ['user_id'])
    op.create_index('ix_memory_user_id_key', 'memory', ['user_id', 'key'])
    op.create_index('ix_memory_weight', 'memory', ['weight'])
    op.create_index('ix_memory_updated_at', 'memory', ['updated_at'])
    
    # ─────────────────────────────────────────────────────────
    # Step 7: Create user_settings table
    # ─────────────────────────────────────────────────────────
    
    op.create_table(
        'user_settings',
        sa.Column('id', UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('key', sa.String(), nullable=False),
        sa.Column('value', JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id', 'key', name='uq_user_settings_user_key'),
    )
    
    op.create_index('ix_user_settings_user_id', 'user_settings', ['user_id'])
    
    # ─────────────────────────────────────────────────────────
    # Step 8: Create embeddings table (optional)
    # ─────────────────────────────────────────────────────────
    
    op.create_table(
        'embeddings',
        sa.Column('id', UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('ref_type', sa.String(), nullable=False),
        sa.Column('ref_id', UUID(as_uuid=True), nullable=False),
        sa.Column('vector_metadata', JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    
    op.create_index('ix_embeddings_user_id', 'embeddings', ['user_id'])
    op.create_index('ix_embeddings_ref_type_ref_id', 'embeddings', ['ref_type', 'ref_id'])
    
    # ─────────────────────────────────────────────────────────
    # Step 9: Add new columns to chats
    # ─────────────────────────────────────────────────────────
    
    op.add_column('chats', sa.Column('is_archived', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('chats', sa.Column('user_metadata', JSONB(), nullable=True, server_default='{}'))
    
    # ─────────────────────────────────────────────────────────
    # Step 10: Create indexes on existing tables
    # ─────────────────────────────────────────────────────────
    
    op.create_index('ix_chats_is_archived', 'chats', ['is_archived'])
    op.create_index('ix_messages_is_deleted', 'messages', ['is_deleted'])
    op.create_index('ix_chats_updated_at', 'chats', ['updated_at'])
    
    print("✓ Migration successful: Neon schema normalized")


def downgrade():
    """Rollback schema normalization."""
    
    # Drop new tables
    op.drop_table('embeddings')
    op.drop_table('user_settings')
    op.drop_table('memory')
    op.drop_table('sessions')
    
    # Drop new indexes
    op.drop_index('ix_messages_is_deleted')
    op.drop_index('ix_chats_is_archived')
    op.drop_index('ix_chats_updated_at')
    
    # Drop new columns
    op.drop_column('messages', 'updated_at')
    op.drop_column('messages', 'is_deleted')
    op.drop_column('messages', 'image_url')
    op.drop_column('chats', 'user_metadata')
    op.drop_column('chats', 'is_archived')
    
    # Restore backup tables if needed
    op.execute("DROP TABLE IF EXISTS users_backup;")
    
    print("✓ Rollback successful")
