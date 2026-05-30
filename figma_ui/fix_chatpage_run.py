import re

with open("src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

# Update runStandard call in handleSend
# From: response = await runStandard(userText, currentChatId || undefined, attachedFile || undefined, ac.signal);
# To: response = await runStandard(userText, activeModel.id, currentChatId || undefined, attachedFile || undefined, ac.signal);
content = content.replace(
    "response = await runStandard(userText, currentChatId || undefined, attachedFile || undefined, ac.signal);",
    "response = await runStandard(userText, activeModel.id, currentChatId || undefined, attachedFile || undefined, ac.signal);"
)

with open("src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)

