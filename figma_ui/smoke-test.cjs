const puppeteer = require('puppeteer');

async function runSmokeTest() {
  const FRONTEND_URL = 'https://sentinel-e-evo.vercel.app';
  
  let browser;
  const results = {
    routing: [],
    console: [],
    auth: [],
    protected: [],
    assets: []
  };

  try {
    browser = await puppeteer.launch({ headless: 'new' });
    const page = await browser.newPage();
    
    // Capture console logs
    page.on('console', msg => {
      const text = msg.text();
      if (text.includes('Supabase auth is not configured') || 
          text.includes('REACT_APP_')) {
        results.console.push(`FAIL: Found legacy warning in console: ${text}`);
      } else {
        // Just checking if any error exists, but not tracking all
      }
    });

    console.log("Testing Assets...");
    const faviconRes = await page.goto(`${FRONTEND_URL}/favicon.ico`, { waitUntil: 'networkidle0' });
    if (faviconRes.status() === 200) results.assets.push("PASS: /favicon.ico returns 200");
    else results.assets.push(`FAIL: /favicon.ico returned ${faviconRes.status()}`);

    const imgRes = await page.goto(`${FRONTEND_URL}/sentinel-e(1).png`, { waitUntil: 'networkidle0' });
    if (imgRes.status() === 200) results.assets.push("PASS: /sentinel-e(1).png returns 200");
    else results.assets.push(`FAIL: /sentinel-e(1).png returned ${imgRes.status()}`);

    console.log("Testing Routing (Public)...");
    const publicRoutes = ['/', '/login', '/signup'];
    for (const route of publicRoutes) {
      const res = await page.goto(`${FRONTEND_URL}${route}`, { waitUntil: 'networkidle0' });
      if (res.status() === 404) {
        results.routing.push(`FAIL: Vercel 404 on ${route}`);
      } else {
        const title = await page.title();
        results.routing.push(`PASS: Route ${route} loaded successfully (Status: ${res.status()})`);
      }
    }

    console.log("Testing Protected Routes without Auth...");
    const protectedRoutes = ['/chat', '/profile', '/settings', '/admin'];
    for (const route of protectedRoutes) {
      await page.goto(`${FRONTEND_URL}${route}`, { waitUntil: 'networkidle0' });
      // It should redirect to /login
      const currentUrl = page.url();
      if (currentUrl.includes('/login')) {
        results.protected.push(`PASS: ${route} successfully redirected to /login`);
      } else {
        results.protected.push(`FAIL: ${route} did not redirect to /login (Current URL: ${currentUrl})`);
      }
    }
    
    // Summary
    console.log(JSON.stringify(results, null, 2));

  } catch (error) {
    console.error("Smoke test execution failed:", error);
  } finally {
    if (browser) await browser.close();
  }
}

runSmokeTest();
