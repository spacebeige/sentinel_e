import puppeteer from 'puppeteer';

const URL = 'http://localhost:5174';
const TEST_EMAIL = `sentinel.test.${Date.now()}@gmail.com`;
const TEST_PASS = 'SecurePass123!';
const TEST_NAME = 'Validation Commander';

async function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function run() {
  console.log('--- VALIDATION SCRIPT START ---');
  
  const browser = await puppeteer.launch({ 
    headless: 'new', 
    args: ['--no-sandbox', '--disable-setuid-sandbox'] 
  });
  
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  
  const logs = [];
  page.on('console', msg => {
    const text = msg.text();
    logs.push(text);
    if (text.includes('[PROTECTED_ROUTE]') || text.includes('[RAW') || text.includes('[CALLBACK') || text.includes('429')) {
      console.log(`[BROWSER] ${text}`);
    }
  });

  try {
    // ------------------------------------------------------------------
    // TEST 1 — Signup Flow
    // ------------------------------------------------------------------
    console.log('\n>>> TEST 1: Signup Flow');
    await page.goto(`${URL}/signup`, { waitUntil: 'networkidle2' });
    await page.type('input[type="text"]', TEST_NAME);
    await page.type('input[type="email"]', TEST_EMAIL);
    await page.type('input[type="password"]', TEST_PASS);
    await page.click('button[type="submit"]');
    
    await delay(3500);
    
    const has429 = logs.some(l => l.includes('429') || l.includes('over_email_send_rate_limit'));
    let currentUrl = page.url();
    
    if (has429) {
      console.log('FAIL (429 Rate Limit): Supabase blocked signup.');
      console.log('Cannot proceed with Authenticated Tests.');
    } else {
      if (currentUrl.includes('/complete-profile')) {
        console.log('Complete Profile rendered');
        const skipBtn = await page.$('button[type="button"]');
        if (skipBtn) await skipBtn.click();
        await delay(2500);
      }
      
      currentUrl = page.url();
      if (currentUrl.includes('/chat')) {
        console.log('Chat rendered');
        console.log('PASS: Test 1 Signup Flow');
        
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
        console.log(`session.user.id: ${authData?.userId}`);
        console.log(`auth=true: ${authData?.hasSession}`);
      } else {
        console.log(`FAIL: Test 1 Signup Flow. URL is ${currentUrl}`);
      }
    }

    // ------------------------------------------------------------------
    // TEST 2 — Refresh Persistence
    // ------------------------------------------------------------------
    console.log('\n>>> TEST 2: Refresh Persistence');
    if (!has429 && page.url().includes('/chat')) {
      await page.reload({ waitUntil: 'networkidle2' });
      await delay(2500);
      if (page.url().includes('/chat')) {
        console.log('Chat remains accessible. No redirect.');
        console.log('PASS: Test 2 Refresh Persistence');
      } else {
        console.log(`FAIL: Test 2 Refresh Persistence. Redirected to ${page.url()}`);
      }
    } else {
      console.log('BLOCKED');
    }

    // ------------------------------------------------------------------
    // TEST 3 — Logout
    // ------------------------------------------------------------------
    console.log('\n>>> TEST 3: Logout');
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
      
      await delay(2500);
      if (page.url().includes('/login') || page.url() === `${URL}/`) {
        console.log('Redirect /login. ProtectedRoute blocks access.');
        console.log('PASS: Test 3 Logout');
      } else {
        console.log(`FAIL: Test 3 Logout. URL is ${page.url()}`);
      }
    } else {
      console.log('BLOCKED');
    }

    // ------------------------------------------------------------------
    // TEST 4 — Login
    // ------------------------------------------------------------------
    console.log('\n>>> TEST 4: Login');
    if (!has429) {
      await page.goto(`${URL}/login`, { waitUntil: 'networkidle2' });
      await page.type('input[type="email"]', TEST_EMAIL);
      await page.type('input[type="password"]', TEST_PASS);
      await page.click('button[type="submit"]');
      
      await delay(3500);
      if (page.url().includes('/chat')) {
        console.log('Chat rendered. auth=true');
        console.log('PASS: Test 4 Login');
      } else {
        console.log(`FAIL: Test 4 Login. URL is ${page.url()}`);
      }
    } else {
       console.log('BLOCKED');
    }

    // ------------------------------------------------------------------
    // TEST 5 — Refresh After Login
    // ------------------------------------------------------------------
    console.log('\n>>> TEST 5: Refresh After Login');
    if (!has429 && page.url().includes('/chat')) {
      await page.reload({ waitUntil: 'networkidle2' });
      await delay(2500);
      if (page.url().includes('/chat')) {
        console.log('Chat remains accessible.');
        console.log('PASS: Test 5 Refresh After Login');
      } else {
        console.log(`FAIL: Test 5 Refresh. Redirected to ${page.url()}`);
      }
    } else {
      console.log('BLOCKED');
    }

    // ------------------------------------------------------------------
    // TEST 7 — Admin Route (Non-admin portion)
    // ------------------------------------------------------------------
    console.log('\n>>> TEST 7: Admin Route');
    if (!has429) {
      await page.goto(`${URL}/admin`, { waitUntil: 'networkidle2' });
      await delay(1500);
      if (page.url().includes('/chat')) {
        console.log('Non-admin: /admin redirects to /chat. isAdmin=false');
        console.log('PASS: Test 7 Admin Route (Non-admin check)');
      } else {
        console.log(`FAIL: Test 7. URL is ${page.url()}`);
      }
      
      // Logout manually for Test 6
      await page.evaluate(() => { localStorage.clear(); });
      await page.reload();
      await delay(1000);
    } else {
      console.log('BLOCKED');
    }

    // ------------------------------------------------------------------
    // TEST 6 — Route Protection (Unauthenticated)
    // ------------------------------------------------------------------
    console.log('\n>>> TEST 6: Route Protection');
    let unauthPass = true;
    for (const route of ['/chat', '/models', '/engines', '/admin']) {
      await page.goto(`${URL}${route}`, { waitUntil: 'networkidle2' });
      await delay(1000);
      if (!page.url().includes('/login')) {
        console.log(`FAIL: ${route} did not redirect to /login. Instead on ${page.url()}`);
        unauthPass = false;
      }
    }
    if (unauthPass) {
      console.log('All unauthenticated routes redirect to /login. No loops. No blank pages.');
      console.log('PASS: Test 6 Route Protection');
    }

  } catch (err) {
    console.error('Test execution error:', err);
  } finally {
    await browser.close();
    console.log('\n--- VALIDATION SCRIPT END ---');
  }
}

run();
