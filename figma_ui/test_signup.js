import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.VITE_SUPABASE_ANON_KEY;

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.error("Missing supabase credentials");
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function testSignup() {
  const email = `test_rate_limit_${Date.now()}@sentinel.dev`;
  console.log(`Testing signup for ${email}`);
  const { data, error } = await supabase.auth.signUp({
    email,
    password: 'SecurePass123!'
  });
  
  if (error) {
    console.error("Signup failed:", error);
  } else {
    console.log("Signup success:", data);
  }
}

testSignup();
