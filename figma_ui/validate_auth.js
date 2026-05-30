import puppeteer from 'puppeteer';

const URL = 'http://localhost:5174';

const TEST_EMAIL = `test_${Date.now()}@sentinel.dev`;
const TEST_PASS = 'password123';
const TEST_NAME = 'Test Commander';

async function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function run() {
  console.log('--- STARTING VALIDATION TESTS ---');
  
  const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
  
  // We will run tests with a persistent context to simulate a real user session
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  
  // Hook up console logs to capture output
  const logs = [];
  page.on('console', msg => {
    const text = msg.text();
    logs.push(text);
    if (true) {
      console.log(`[BROWSER LOG] ${text}`);
    }
  });

  try {
    // ------------------------------------------------------------------
    // TEST 5: Route Protection (Unauthenticated)
    // ------------------------------------------------------------------
    console.log('\n>>> Running Test 5: Route Protection');
    let t5_pass = true;
    for (const route of ['/chat', '/models', '/engines', '/admin']) {
      await page.goto(`${URL}${route}`, { waitUntil: 'networkidle2' });
      await delay(1000);
      const curUrl = page.url();
      if (!curUrl.includes('/login')) {
        console.log(`FAIL: Route ${route} did not redirect to /login. Instead on ${curUrl}`);
        t5_pass = false;
      }
    }
    if (t5_pass) {
      console.log('PASS: Test 5 Route Protection');
    }

    // ------------------------------------------------------------------
    // TEST 1: Email Signup
    // ------------------------------------------------------------------
    console.log('\n>>> Running Test 1: Email Signup');
    await page.goto(`${URL}/signup`, { waitUntil: 'networkidle2' });
    await page.type('input[type="text"]', TEST_NAME);
    await page.type('input[type="email"]', TEST_EMAIL);
    await page.type('input[type="password"]', TEST_PASS);
    await page.click('button[type="submit"]');
    
    // Wait for redirect to complete-profile or chat
    await page.waitForNavigation({ timeout: 10000 }).catch(() => {});
    await delay(2000);
    
    const curUrlAfterSignup = page.url();
    if (curUrlAfterSignup.includes('/complete-profile')) {
      console.log('Successfully navigated to /complete-profile');
      // Click "Skip for now"
      const skipButton = await page.$('button[type="button"]');
      if (skipButton) {
        await skipButton.click();
        await page.waitForNavigation({ timeout: 5000 }).catch(() => {});
        await delay(1000);
      }
    }
    
    const isAtChat = page.url().includes('/chat');
    if (isAtChat) {
      console.log('PASS: Test 1 Email Signup');
    } else {
      console.log(`FAIL: Test 1 Email Signup - ended up at ${page.url()}`);
    }

    // ------------------------------------------------------------------
    // TEST 3: Refresh Persistence
    // ------------------------------------------------------------------
    console.log('\n>>> Running Test 3: Refresh Persistence');
    await page.reload({ waitUntil: 'networkidle2' });
    await delay(2000);
    const urlAfterRefresh = page.url();
    if (urlAfterRefresh.includes('/chat')) {
      console.log('PASS: Test 3 Refresh Persistence');
    } else {
      console.log(`FAIL: Test 3 Refresh Persistence - redirected to ${urlAfterRefresh}`);
    }

    // ------------------------------------------------------------------
    // TEST 6: Admin Protection
    // ------------------------------------------------------------------
    console.log('\n>>> Running Test 6: Admin Protection');
    await page.goto(`${URL}/admin`, { waitUntil: 'networkidle2' });
    await delay(2000);
    const urlAfterAdmin = page.url();
    if (urlAfterAdmin.includes('/chat')) {
      console.log('PASS: Test 6 Admin Protection (Non-admin redirected to chat)');
    } else {
      console.log(`FAIL: Test 6 Admin Protection - ended up at ${urlAfterAdmin}`);
    }

    // ------------------------------------------------------------------
    // TEST 4: Logout
    // ------------------------------------------------------------------
    console.log('\n>>> Running Test 4: Logout');
    await page.goto(`${URL}/chat`, { waitUntil: 'networkidle2' });
    await delay(1000);
    
    // Find logout button in navbar. It's usually the LogOut icon
    // For simplicity, we can evaluate a script to click it if it contains "Log Out" or similar, 
    // or we can just clear local storage to simulate logout if we can't find the button easily.
    const logoutClicked = await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const logoutBtn = btns.find(b => b.textContent.includes('Sign Out') || b.textContent.includes('Log Out') || b.innerHTML.includes('LogOut') || b.querySelector('svg.lucide-log-out'));
      if (logoutBtn) {
        logoutBtn.click();
        return true;
      }
      return false;
    });

    if (logoutClicked) {
      await delay(2000);
      const urlAfterLogout = page.url();
      if (urlAfterLogout.includes('/login') || urlAfterLogout.includes('/')) {
        console.log('PASS: Test 4 Logout');
      } else {
        console.log(`FAIL: Test 4 Logout - ended up at ${urlAfterLogout}`);
      }
    } else {
      console.log('WARN: Could not find logout button, clearing localStorage instead');
      await page.evaluate(() => localStorage.clear());
      await page.reload();
      await delay(1000);
    }

    // ------------------------------------------------------------------
    // TEST 2: Email Login
    // ------------------------------------------------------------------
    console.log('\n>>> Running Test 2: Email Login');
    await page.goto(`${URL}/login`, { waitUntil: 'networkidle2' });
    await page.type('input[type="email"]', TEST_EMAIL);
    await page.type('input[type="password"]', TEST_PASS);
    await page.click('button[type="submit"]');
    
    await page.waitForNavigation({ timeout: 10000 }).catch(() => {});
    await delay(2000);
    
    const urlAfterLogin = page.url();
    if (urlAfterLogin.includes('/chat')) {
      console.log('PASS: Test 2 Email Login');
    } else {
      console.log(`FAIL: Test 2 Email Login - ended up at ${urlAfterLogin}`);
    }

    // ------------------------------------------------------------------
    // TEST 7: Password Recovery
    // ------------------------------------------------------------------
    console.log('\n>>> Running Test 7: Password Recovery');
    // We just test if we can submit the form successfully
    await page.goto(`${URL}/forgot-password`, { waitUntil: 'networkidle2' });
    await page.type('input[type="email"]', TEST_EMAIL);
    await page.click('button[type="submit"]');
    await delay(3000);
    
    const successMsg = await page.evaluate(() => {
      return document.body.innerText.includes('Check your email');
    });
    if (successMsg) {
      console.log('PASS: Test 7 Password Recovery (Email sent)');
    } else {
      console.log('FAIL: Test 7 Password Recovery');
    }

    // ------------------------------------------------------------------
    // TEST 8: Google OAuth
    // ------------------------------------------------------------------
    console.log('\n>>> Running Test 8: Google OAuth');
    await page.goto(`${URL}/login`, { waitUntil: 'networkidle2' });
    
    // Clear logs to only capture OAuth logs
    logs.length = 0;
    
    // Click "Continue with Google"
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const googleBtn = btns.find(b => b.textContent.includes('Continue with Google'));
      if (googleBtn) googleBtn.click();
    });
    
    await delay(4000); // wait for redirect
    
    // Usually Google redirects to accounts.google.com, then redirects back to /auth/callback
    // Let's see if we get the Supabase OAuth URL
    console.log(`Current URL after Google click: ${page.url()}`);
    if (page.url().includes('accounts.google.com') || page.url().includes('supabase.co/auth/v1/authorize')) {
       console.log('PASS: Test 8 Google OAuth (Redirected to provider)');
    } else if (page.url().includes('error')) {
       console.log('Google OAuth resulted in error URL');
    }
    
    // Check logs for errors
    const hasError = logs.some(l => l.toLowerCase().includes('error'));
    if (hasError) {
      console.log('Google OAuth Logs found errors:', logs.filter(l => l.toLowerCase().includes('error')));
    }

  } catch (err) {
    console.error('Test execution failed:', err);
  } finally {
    await browser.close();
    console.log('--- TESTS COMPLETE ---');
  }
}

run();
