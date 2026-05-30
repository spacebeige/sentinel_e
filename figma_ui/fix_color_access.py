import re

with open("src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

# Make sure we import getModelConfig
content = content.replace(
    'import { MODELS as AVAILABLE_MODELS, getModeConfig, ALL_RUNTIME_MODES } from "../config/runtime";',
    'import { MODELS as AVAILABLE_MODELS, getModelConfig, getModeConfig, ALL_RUNTIME_MODES } from "../config/runtime";'
)

# Replace all instances of `proSubModes.find(m => m.id === selectedMode)!.color`
# and `proSubModes.find(m => m.id === selectedMode)!.placeholder`
content = content.replace(
    'proSubModes.find(m => m.id === selectedMode)!.color',
    'getModeConfig(selectedMode).color'
)
content = content.replace(
    'proSubModes.find(m => m.id === selectedMode)!.placeholder',
    'getModeConfig(selectedMode).placeholder'
)

# Fix `activeModel` safely, although activeModel itself might be used elsewhere.
# Actually, the user asked to replace `selectedModel.color` or `activeModel.color` if it's unsafe.
# Instead of activeModel, let's just make sure activeModel is safe.
content = content.replace(
    'const activeModel = AVAILABLE_MODELS.find(m => m.id === selectedModel) || AVAILABLE_MODELS[0];',
    'const activeModel = getModelConfig(selectedModel);'
)

with open("src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)
