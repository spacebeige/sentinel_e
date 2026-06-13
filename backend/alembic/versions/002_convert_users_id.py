"""convert users.id from uuid to varchar

Revision ID: 002_convert_users_id
Revises: 001_normalize_neon_schema
Create Date: 2026-06-13 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision = '002_convert_users_id'
down_revision = '001_normalize_neon_schema'
branch_labels = None
depends_on = None

def upgrade():
    # 1. Drop foreign keys that reference users.id
    op.drop_constraint('fk_chats_user_id', 'chats', type_='foreignkey')
    op.drop_constraint('fk_messages_user_id', 'messages', type_='foreignkey')
    op.drop_constraint('sessions_user_id_fkey', 'sessions', type_='foreignkey')
    op.drop_constraint('memory_user_id_fkey', 'memory', type_='foreignkey')
    op.drop_constraint('user_settings_user_id_fkey', 'user_settings', type_='foreignkey')
    op.drop_constraint('embeddings_user_id_fkey', 'embeddings', type_='foreignkey')
    # Note: context_windows was created without an explicit foreign key in cb896e9e1284

    # 2. Alter users.id from UUID to VARCHAR
    # We must explicitly cast existing UUIDs to VARCHAR
    op.execute('ALTER TABLE users ALTER COLUMN id TYPE VARCHAR USING id::VARCHAR')

    # 3. Ensure dependent columns are VARCHAR (They should already be, but ensure safety)
    op.execute('ALTER TABLE chats ALTER COLUMN user_id TYPE VARCHAR USING user_id::VARCHAR')
    op.execute('ALTER TABLE messages ALTER COLUMN user_id TYPE VARCHAR USING user_id::VARCHAR')
    op.execute('ALTER TABLE sessions ALTER COLUMN user_id TYPE VARCHAR USING user_id::VARCHAR')
    op.execute('ALTER TABLE memory ALTER COLUMN user_id TYPE VARCHAR USING user_id::VARCHAR')
    op.execute('ALTER TABLE user_settings ALTER COLUMN user_id TYPE VARCHAR USING user_id::VARCHAR')
    op.execute('ALTER TABLE embeddings ALTER COLUMN user_id TYPE VARCHAR USING user_id::VARCHAR')

    # 4. Recreate foreign keys
    op.create_foreign_key('fk_chats_user_id', 'chats', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_messages_user_id', 'messages', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('sessions_user_id_fkey', 'sessions', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('memory_user_id_fkey', 'memory', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('user_settings_user_id_fkey', 'user_settings', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('embeddings_user_id_fkey', 'embeddings', 'users', ['user_id'], ['id'], ondelete='CASCADE')

def downgrade():
    # Reverse the process: convert back to UUID
    # This requires that all IDs in the database are valid UUIDs, which might fail if Clerk IDs exist.
    
    # 1. Drop foreign keys
    op.drop_constraint('fk_chats_user_id', 'chats', type_='foreignkey')
    op.drop_constraint('fk_messages_user_id', 'messages', type_='foreignkey')
    op.drop_constraint('sessions_user_id_fkey', 'sessions', type_='foreignkey')
    op.drop_constraint('memory_user_id_fkey', 'memory', type_='foreignkey')
    op.drop_constraint('user_settings_user_id_fkey', 'user_settings', type_='foreignkey')
    op.drop_constraint('embeddings_user_id_fkey', 'embeddings', type_='foreignkey')

    # 2. Alter users.id back to UUID (Will fail if non-UUIDs exist)
    op.execute('ALTER TABLE users ALTER COLUMN id TYPE UUID USING id::UUID')

    # 3. Alter dependent columns back to UUID
    op.execute('ALTER TABLE chats ALTER COLUMN user_id TYPE UUID USING user_id::UUID')
    op.execute('ALTER TABLE messages ALTER COLUMN user_id TYPE UUID USING user_id::UUID')
    op.execute('ALTER TABLE sessions ALTER COLUMN user_id TYPE UUID USING user_id::UUID')
    op.execute('ALTER TABLE memory ALTER COLUMN user_id TYPE UUID USING user_id::UUID')
    op.execute('ALTER TABLE user_settings ALTER COLUMN user_id TYPE UUID USING user_id::UUID')
    op.execute('ALTER TABLE embeddings ALTER COLUMN user_id TYPE UUID USING user_id::UUID')

    # 4. Recreate foreign keys
    op.create_foreign_key('fk_chats_user_id', 'chats', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_messages_user_id', 'messages', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('sessions_user_id_fkey', 'sessions', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('memory_user_id_fkey', 'memory', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('user_settings_user_id_fkey', 'user_settings', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('embeddings_user_id_fkey', 'embeddings', 'users', ['user_id'], ['id'], ondelete='CASCADE')
