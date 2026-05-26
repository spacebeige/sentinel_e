import re

with open("/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/HomePage.tsx", "r") as f:
    content = f.read()

# 1. Imports
content = content.replace('import { useState, useEffect, useRef } from "react";', 'import { useState, useEffect, useRef } from "react";\nimport { useTheme } from "../context/ThemeContext";')

# 2. State
old_state = """export default function HomePage() {
  const [isDark, setIsDark] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setIsDark(document.documentElement.classList.contains("dark"));
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);"""
new_state = """export default function HomePage() {
  const { isDark } = useTheme();
  const scrollRef = useRef<HTMLDivElement>(null);"""
content = content.replace(old_state, new_state)

with open("/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/HomePage.tsx", "w") as f:
    f.write(content)

print("HomePage updated")
