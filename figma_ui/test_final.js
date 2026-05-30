import puppeteer from 'puppeteer';

const URL = 'http://localhost:5174';
const TEST_EMAIL = `commander_${Math.floor(Math.random() * 100000)}@sentinel.dev`;
const TEST_PASS = 'SecurePass123!';
const TEST_NAME = 'Alpha Commander';

async function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function run() {
  console.log('--- FINAL VALIDATION SCRIPT START ---');
  
  const browser = await puppeteer.launch({ 
    headless: 'new', 
    args: ['--no-sandbox', '--disable-setuid-sandbox'] 
  });
  
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  
  // Track console logs
  const logs = [];
  page.on('console', msg => {
    logs.push(msg.text());
    console.log(`[BROWSER] ${msg.text()}`);
  });

  try {
    // ------------------------------------------------------------------
    // TEST A — Real Signup
    // ------------------------------------------------------------------
    console.log('\n>>> TEST A: Real Signup');
    await page.goto(`${URL}/signup`, { waitUntil: 'networkidle2' });
    await page.type('input[type="text"]', TEST_NAME);
    await page.type('input[type="email"]', TEST_EMAIL);
    await page.type('input[type="password"]', TEST_PASS);
    await page.click('button[type="submit"]');
    
    await delay(3000);
    let currentUrl = page.url();
    
    // Check if we hit a 429
    const has429 = logs.some(l => l.includes('429'));
    if (has429) {
      console.log('BLOCKED (429): Test A Signup blocked by Supabase Rate Limit.');
      console.log('Cannot proceed with Test B (Login) for a new user because account was not created.');
    } else {
      // Check if we are on Complete Profile or Chat
      if (currentUrl.includes('/complete-profile')) {
        console.log('Signup Success: Reached /complete-profile');
        const skipBtn = await page.$('button[type="button"]');
        if (skipBtn) await skipBtn.click();
        await delay(2000);
      }
      
      currentUrl = page.url();
      if (currentUrl.includes('/chat')) {
        console.log('PASS: Test A Real Signup (Chat Loads)');
        
        // Capture session info from localStorage
        const authData = await page.evaluate(() => {
          for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key.includes('-auth-token')) {
              const val = JSON.parse(localStorage.getItem(key));
              return {
                userId: val?.user?.id,
                hasSession: !!val?.session
              };
            }
          }
          return null;
        });
        console.log(`Captured session.user.id: ${authData?.userId}`);
        
      } else {
        console.log(`FAIL: Test A Real Signup. Ended up at ${currentUrl}`);
      }
    }

    // ------------------------------------------------------------------
    // TEST D — Logout (Done early so we can test Login next)
    // ------------------------------------------------------------------
    console.log('\n>>> TEST D: Logout');
    if (!has429 && page.url().includes('/chat')) {
      const logoutSuccess = await page.evaluate(() => {
        const btns = Array.from(document.querySelectorAll('button'));
        const logoutBtn = btns.find(b => b.textContent.includes('Sign Out') || b.textContent.includes('Log Out') || b.innerHTML.includes('LogOut') || b.querySelector('svg.lucide-log-out'));
        if (logoutBtn) {
          logoutBtn.click();
          return true;
        }
        return false;
      });
      
      await delay(2000);
      if (page.url().includes('/login') || page.url() === `${URL}/`) {
        console.log('PASS: Test D Logout (/login rendered)');
      } else {
        console.log(`FAIL: Test D Logout. URL is ${page.url()}`);
      }
    } else {
      console.log('BLOCKED: Cannot test Logout because Signup failed.');
    }

    // ------------------------------------------------------------------
    // TEST B — Real Login
    // ------------------------------------------------------------------
    console.log('\n>>> TEST B: Real Login');
    if (!has429) {
      await page.goto(`${URL}/login`, { waitUntil: 'networkidle2' });
      await page.type('input[type="email"]', TEST_EMAIL);
      await page.type('input[type="password"]', TEST_PASS);
      await page.click('button[type="submit"]');
      
      await delay(3000);
      if (page.url().includes('/chat')) {
        console.log('PASS: Test B Real Login (Chat Opens)');
      } else {
        console.log(`FAIL: Test B Real Login. URL is ${page.url()}`);
      }
    } else {
       console.log('BLOCKED: Cannot test Login for new user due to 429 rate limit on Signup.');
    }

    // ------------------------------------------------------------------
    // TEST C — Refresh
    // ------------------------------------------------------------------
    console.log('\n>>> TEST C: Refresh');
    if (!has429 && page.url().includes('/chat')) {
      await page.reload({ waitUntil: 'networkidle2' });
      await delay(2000);
      if (page.url().includes('/chat')) {
        console.log('PASS: Test C Refresh (No Redirect, Chat Accessible)');
      } else {
        console.log(`FAIL: Test C Refresh. Redirected to ${page.url()}`);
      }
    } else {
      console.log('BLOCKED: Cannot test Refresh due to previous failures.');
    }

    // ------------------------------------------------------------------
    // TEST E — Direct Route Access
    // ------------------------------------------------------------------
    console.log('\n>>> TEST E: Direct Route Access');
    
    // We are currently Authenticated (if no 429)
    if (!has429) {
      console.log('Testing Authenticated Routes:');
      for (const route of ['/chat', '/models', '/engines']) {
        await page.goto(`${URL}${route}`, { waitUntil: 'networkidle2' });
        await delay(1000);
        if (page.url().includes(route)) {
          console.log(`  PASS: ${route} loaded correctly.`);
        } else {
          console.log(`  FAIL: ${route} redirected to ${page.url()}`);
        }
      }
      
      // TEST F part 1: Admin route as non-admin
      console.log('\n>>> TEST F: Admin Route (Non-admin)');
      await page.goto(`${URL}/admin`, { waitUntil: 'networkidle2' });
      await delay(1000);
      if (page.url().includes('/chat')) {
        console.log('PASS: Non-admin redirected to /chat correctly.');
      } else {
        console.log(`FAIL: Non-admin ended up at ${page.url()}`);
      }
      
      // Logout to test unauthenticated
      await page.evaluate(() => { localStorage.clear(); });
      await page.reload();
      await delay(1000);
    }

    console.log('\nTesting Unauthenticated Routes:');
    let unauthPass = true;
    for (const route of ['/chat', '/models', '/engines', '/admin']) {
      await page.goto(`${URL}${route}`, { waitUntil: 'networkidle2' });
      await delay(1000);
      if (!page.url().includes('/login')) {
        console.log(`  FAIL: ${route} did not redirect to /login. Instead on ${page.url()}`);
        unauthPass = false;
      } else {
        console.log(`  PASS: ${route} correctly redirected to /login.`);
      }
    }
    if (unauthPass) console.log('PASS: Test E Unauthenticated Route Protection');
    
  } catch (err) {
    console.error('Test execution error:', err);
  } finally {
    await browser.close();
    console.log('\n--- FINAL VALIDATION SCRIPT END ---');
  }
}

run();
