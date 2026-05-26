import re

files = [
    "/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/HeroSection.tsx",
    "/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/ChatPage.tsx",
    "/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/EnginesPage.tsx",
    "/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/PricingPage.tsx"
]

for file in files:
    with open(file, "r") as f:
        content = f.read()

    # Imports
    content = content.replace('import { useState, useEffect, useRef } from "react";', 'import { useState, useEffect, useRef } from "react";\nimport { useTheme } from "../context/ThemeContext";')
    content = content.replace('import { useState, useEffect } from "react";', 'import { useState, useEffect } from "react";\nimport { useTheme } from "../context/ThemeContext";')

    # React logic
    old_state1 = """  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    setIsDark(document.documentElement.classList.contains("dark"));
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);"""
    old_state2 = """  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    setIsDark(document.documentElement.classList.contains("dark"));
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);"""
    
    # We might just use regex to remove it
    new_state = """  const { isDark } = useTheme();"""
    content = re.sub(r'  const \[isDark, setIsDark\] = useState\(false\);\n*\s*useEffect\(\(\) => \{\n\s*setIsDark.*?observer\.disconnect\(\);\n\s*\}, \[\]\);', new_state, content, flags=re.DOTALL)

    with open(file, "w") as f:
        f.write(content)
        print(f"Updated {file}")
