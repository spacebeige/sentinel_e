import re

with open('src/app/components/CrossAnalysisPanel.tsx', 'r') as f:
    content = f.read()

content = content.replace(
"""import {
  runCrossAnalysis,
  type CrossAnalysisResult,
  type CrossAnalysisModelProfile,
} from "../types";""",
"""import { runCrossAnalysis } from "@services/api";
import type { CrossAnalysisResult, CrossAnalysisModelProfile } from "../types";""")

with open('src/app/components/CrossAnalysisPanel.tsx', 'w') as f:
    f.write(content)
