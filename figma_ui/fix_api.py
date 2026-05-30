import re

with open("src/app/api.ts", "r") as f:
    content = f.read()

# Add imports if not present
if "MODEL_RUNTIME_MAP" not in content:
    content = content.replace(
        "import { adaptRunResponse",
        "import { MODEL_RUNTIME_MAP, ORCHESTRATION_MODE_MAP } from \"./config/runtime\";\nimport { adaptRunResponse"
    )

run_standard_target = """export async function runStandard(
  text: string,
  chatId?: string,
  file?: File,
  signal?: AbortSignal
): Promise<SentinelRunResponse> {
  const payload: Record<string, any> = {
    query: text,
    mode: "standard",
  };"""

run_standard_replacement = """export async function runStandard(
  text: string,
  modelId: string,
  chatId?: string,
  file?: File,
  signal?: AbortSignal
): Promise<SentinelRunResponse> {
  const mappedModel = MODEL_RUNTIME_MAP[modelId]?.model || modelId;
  const payload: Record<string, any> = {
    runtime: "single-model",
    model: mappedModel,
    mode: "standard",
    query: text,
  };"""
content = content.replace(run_standard_target, run_standard_replacement)

run_exp_target = """export async function runExperimental(
  text: string,
  subMode: string = "debate",
  rounds: number = 6,
  chatId?: string,
  killSwitch: boolean = false,
  file?: File,
  signal?: AbortSignal
): Promise<SentinelRunResponse> {
  const payload: Record<string, any> = {
    query: killSwitch ? "kill" : text,
    mode: "experimental",
    sub_mode: killSwitch ? "glass" : subMode,
  };"""

run_exp_replacement = """export async function runExperimental(
  text: string,
  subMode: string = "debate",
  rounds: number = 6,
  chatId?: string,
  killSwitch: boolean = false,
  file?: File,
  signal?: AbortSignal
): Promise<SentinelRunResponse> {
  const activeSubMode = killSwitch ? "glass" : subMode;
  const mappedMode = ORCHESTRATION_MODE_MAP[activeSubMode]?.mode || activeSubMode;
  const payload: Record<string, any> = {
    runtime: "mco",
    mode: mappedMode,
    orchestration: true,
    query: killSwitch ? "kill" : text,
  };"""
content = content.replace(run_exp_target, run_exp_replacement)

with open("src/app/api.ts", "w") as f:
    f.write(content)

