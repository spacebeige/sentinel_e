import re

with open("src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

content = content.replace(
    'import { MODELS as AVAILABLE_MODELS, getModeConfig, ALL_RUNTIME_MODES } from "../config/runtime";\nconst proSubModes = ALL_RUNTIME_MODES.filter(m => m.isExperimental);',
    'import { MODELS as AVAILABLE_MODELS, getModeConfig, ALL_RUNTIME_MODES } from "../config/runtime";'
)

with open("src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)
