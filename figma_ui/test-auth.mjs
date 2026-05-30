import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://kyqoygozcxxsmlkkraub.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt5cW95Z296Y3h4c21sa2tyYXViIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg4ODMyNTUsImV4cCI6MjA5NDQ1OTI1NX0.jvczSFd2H8ij2dhlg7HZxlOePFYE2wPLlfeyndkFUYw';

const supabase = createClient(supabaseUrl, supabaseKey);

async function test() {
  console.log("--- ATTEMPTING SIGNUP ---");
  const signupRes = await supabase.auth.signUp({
    email: 'test-commander-12345@sentinel.dev',
    password: 'password123'
  });
  console.log("SIGNUP USER:", signupRes.data?.user?.id);
  console.log("SIGNUP SESSION:", signupRes.data?.session ? "EXISTS" : "NULL");
  console.log("SIGNUP ERROR:", signupRes.error);

  console.log("\n--- ATTEMPTING LOGIN ---");
  const loginRes = await supabase.auth.signInWithPassword({
    email: 'test-commander-12345@sentinel.dev',
    password: 'password123'
  });
  
  console.log("LOGIN USER:", loginRes.data?.user?.id);
  console.log("LOGIN SESSION:", loginRes.data?.session ? "EXISTS" : "NULL");
  console.log("LOGIN ERROR:", loginRes.error);
}

test();
