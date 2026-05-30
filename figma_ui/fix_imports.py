import re

with open("src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

content = content.replace(
    'import { AVAILABLE_MODELS } from "../../config/modelRegistry";',
    'import { AVAILABLE_MODELS, getModeConfig, proSubModes } from "../config/runtime";'
)

with open("src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)
