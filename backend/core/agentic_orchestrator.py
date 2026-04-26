import logging
import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger("AgenticOrchestrator")

class ToolCall(BaseModel):
    tool_name: str
    tool_input: Dict[str, Any]

class AgenticDecision(BaseModel):
    requires_tools: bool
    tools: List[ToolCall] = []
    reasoning_trace: str
    step_count: int
    # dependencies omitted for simplicity in MVP, but registry can handle it

class AgenticOrchestrator:
    def __init__(self, mco_bridge):
        self.mco_bridge = mco_bridge
        self.max_steps = 8
        self.tools = {
            "web_search": self._tool_web_search,
            "code_exec": self._tool_code_exec,
            "file_read": self._tool_file_read,
            "db_query": self._tool_db_query
        }

    async def run(self, user_id: str, chat_id: str, query: str, context: str) -> Dict[str, Any]:
        """Decompose query → subtask list → execute in order."""
        logger.info(f"Starting agentic turn for query: {query[:50]}...")
        
        step_logs = []
        current_context = context
        final_answer = ""
        
        for step in range(1, self.max_steps + 1):
            # 1. Decide next action
            decision = await self._get_decision(query, current_context, step)
            
            if not decision.requires_tools or not decision.tools:
                # If no more tools needed, we have the final answer in reasoning or need a final synthesis
                final_answer = decision.reasoning_trace
                break
            
            # 2. Execute tools in parallel
            start_time = datetime.utcnow()
            tasks = [self._execute_tool(tc) for tc in decision.tools]
            tool_results = await asyncio.gather(*tasks)
            end_time = datetime.utcnow()
            latency = (end_time - start_time).total_seconds() * 1000

            for i, tool_call in enumerate(decision.tools):
                log_entry = {
                    "step_id": step,
                    "tool_name": tool_call.tool_name,
                    "input": tool_call.tool_input,
                    "output": tool_results[i],
                    "latency_ms": latency / len(decision.tools)
                }
                step_logs.append(log_entry)
            
            # 3. Inject results back into context
            current_context += f"\n\n[Step {step} Results]:\n" + json.dumps(tool_results, indent=2)
            
        return {
            "formatted_output": final_answer,
            "step_logs": step_logs,
            "step_count": len(step_logs)
        }

    async def _get_decision(self, query: str, context: str, step: int) -> AgenticDecision:
        """Ask model to decide next tool call."""
        prompt = f"""
Current Query: {query}
Step: {step}
Context: {context}

Available Tools: {list(self.tools.keys())}

Decide if you need to use a tool to answer the query. 
If yes, provide tool_name and tool_input as JSON.
If no, provide the final answer.

Output format (JSON only):
{{
  "requires_tools": true/false,
  "tools": [{{ "tool_name": "...", "tool_input": {{}} }}],
  "reasoning_trace": "...",
  "step_count": {step}
}}
"""
        try:
            res = await self.mco_bridge.call_model("llama31-70b", prompt, "You are an agentic planner. Output raw JSON.")
            # Simple cleanup
            res = re.sub(r'```json\n|\n```|```', '', res).strip()
            data = json.loads(res)
            return AgenticDecision(**data)
        except Exception as e:
            logger.error(f"Failed to get agentic decision: {e}")
            return AgenticDecision(requires_tools=False, reasoning_trace="I encountered an error planning the next step.", step_count=step)

    async def _execute_tool(self, tool_call: ToolCall) -> Any:
        func = self.tools.get(tool_call.tool_name)
        if func:
            try:
                return await func(**tool_call.tool_input)
            except Exception as e:
                return f"Error executing {tool_call.tool_name}: {str(e)}"
        return f"Tool {tool_call.tool_name} not found."

    # Tool Implementations (Stubs for now)
    async def _tool_web_search(self, query: str) -> str:
        """Integrate Tavily search."""
        try:
            from compressed.search_engine import TavilySearch
            search = TavilySearch()
            if search.available:
                results = await search.search(query, num_results=5)
                return json.dumps(results, indent=2)
            return "Tavily search is not configured or unavailable."
        except Exception as e:
            return f"Search error: {str(e)}"

    async def _tool_code_exec(self, code: str) -> str:
        return "Code execution is restricted in this environment."

    async def _tool_file_read(self, filename: str) -> str:
        return f"Reading {filename}... [Access Denied]"

    async def _tool_db_query(self, query: str) -> str:
        return "Direct database querying is restricted."
