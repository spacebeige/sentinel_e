import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.VITE_SUPABASE_ANON_KEY;

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function testRecovery() {
  console.log('Testing password recovery');
  const { data, error } = await supabase.auth.resetPasswordForEmail('oomkaragarkhed0710@gmail.com');
  
  if (error) {
    console.error("Recovery failed:", error);
  } else {
    console.log("Recovery success:", data);
  }
}

testRecovery();
