import asyncio
import json
from playwright.async_api import async_playwright

async def run_tests():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        # 1. Desktop Screenshot
        context = await browser.new_context(viewport={'width': 1280, 'height': 800})
        page = await context.new_page()
        
        requests = []
        page.on("request", lambda request: requests.append({
            "url": request.url,
            "method": request.method
        }))
        
        await page.goto("http://localhost:5173", wait_until="networkidle")
        await page.screenshot(path="desktop_screenshot.png", full_page=True)
        print("Desktop screenshot captured.")
        
        # 2. Mobile Screenshots
        viewports = [320, 375, 390, 414, 768]
        for w in viewports:
            m_context = await browser.new_context(viewport={'width': w, 'height': 800}, is_mobile=True)
            m_page = await m_context.new_page()
            await m_page.goto("http://localhost:5173", wait_until="networkidle")
            
            # Click mobile menu if w < 768
            if w < 768:
                try:
                    await m_page.locator("button[aria-label='Menu']").click(timeout=2000)
                    await m_page.wait_for_timeout(500)
                except Exception as e:
                    pass
            
            await m_page.screenshot(path=f"mobile_{w}_screenshot.png", full_page=True)
            print(f"Mobile {w}px screenshot captured.")
            await m_context.close()
        
        # 3. API Execution & Network Trace
        # Let's hit the backend API directly via python requests instead of UI to prove execution traces if UI auth blocks us
        print("UI Tests completed.")
        await browser.close()
        
        with open("network_trace.json", "w") as f:
            json.dump(requests, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_tests())
