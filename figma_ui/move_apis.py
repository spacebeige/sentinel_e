import os
import re

legacy_api_path = 'src/legacy/services/api.js'
new_api_path = 'src/app/api.ts'

with open(new_api_path, 'r') as f:
    content = f.read()

# Extract functions
funcs_to_extract = [
    "getRootStatus", "getKernelStatus", "getSessionStats", 
    "getLearningSummary", "getLearningRiskProfiles", "getCurrentUser", 
    "submitAdminRequest", "getAdminRequestStatus"
]

to_append = "\n// --- Migrated from Figma UI ---\n"

for func in funcs_to_extract:
    # Match from "export async function <func>" up to the closing brace at the same indentation level
    pattern = r"export async function " + func + r"[\s\S]*?\n}\n"
    match = re.search(pattern, content)
    if match:
        func_body = match.group(0)
        # remove types
        func_body = re.sub(r': Promise<.*?>', '', func_body)
        func_body = re.sub(r'\(email: string\)', '(email)', func_body)
        func_body = re.sub(r'\(data: AdminRequestData\)', '(data)', func_body)
        func_body = func_body.replace('apiRequest<{ success: boolean; data: any }>', 'api.get')
        func_body = func_body.replace('apiRequest<any>', 'api.get')
        func_body = func_body.replace('apiRequest', 'api.get')
        func_body = func_body.replace('postJson', 'api.post')
        func_body = func_body.replace('getQuick', 'api.get')
        func_body = func_body.replace('if (res && res.success) {', 'if (res && res.data) { return res.data.data || res.data; }')
        
        # Simplify the return mappings since apiRequest structure was different
        to_append += "\n" + func_body

with open(legacy_api_path, 'r') as f:
    legacy = f.read()

# Append before export default api
legacy = legacy.replace("export default api;", to_append + "\nexport default api;")

with open(legacy_api_path, 'w') as f:
    f.write(legacy)
