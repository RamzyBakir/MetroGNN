import asyncio
import json
import os
import sys
import httpx
from typing import List, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Models via OpenRouter
MODEL = "meta-llama/llama-3.3-70b-instruct"

def load_system_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "agent_prompt.md")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"\n[Warning] Could not find prompt file at {prompt_path}. Using fallback prompt.")
        return "You are the MetroGNN Architect and AI Assistant."

SYSTEM_PROMPT = load_system_prompt()

async def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("\n[Error] OPENROUTER_API_KEY environment variable is missing.")
        print("Please set it to your OpenRouter API key before running this client.\n")
        sys.exit(1)

    print("==========================================================")
    print("                 MetroGNN Agent Client                    ")
    print("==========================================================")
    print("Connecting to MetroGNN MCP Server...\n")

    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_server"],
        env=os.environ.copy()
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hello! I just started the client. Please check the system state and tell me what we can do, and ask me what I want to build or research."}
    ]

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List tools
            tools_response = await session.list_tools()
            
            openai_tools = []
            for tool in tools_response.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema
                    }
                })
            
            async with httpx.AsyncClient(timeout=120.0) as http_client:
                while True:
                    payload = {
                        "model": MODEL,
                        "messages": messages,
                    }
                    if openai_tools:
                        payload["tools"] = openai_tools

                    try:
                        response = await http_client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}"},
                            json=payload
                        )
                        response.raise_for_status()
                        response_data = response.json()
                    except httpx.HTTPStatusError as e:
                        print(f"\n[Agent Error]: HTTP Error {e.response.status_code}: {e.response.text}")
                        break
                    except Exception as e:
                        print(f"\n[Agent Error]: Failed to get completion from OpenRouter: {e}")
                        break
                    
                    msg = response_data["choices"][0]["message"]
                    
                    if msg.get("content"):
                        print(f"\n[Agent]: {msg['content']}")
                        
                    if msg.get("tool_calls"):
                        messages.append(msg)
                        for tool_call in msg["tool_calls"]:
                            tool_name = tool_call["function"]["name"]
                            tool_args = json.loads(tool_call["function"]["arguments"])
                            print(f"\n>> [Executing Tool: {tool_name}] with args: {tool_args}")
                            
                            try:
                                result = await session.call_tool(tool_name, tool_args)
                                result_text = "\n".join([c.text for c in result.content if hasattr(c, 'text')])
                                
                                messages.append({
                                    "role": "tool",
                                    "name": tool_name,
                                    "tool_call_id": tool_call["id"],
                                    "content": result_text
                                })
                            except Exception as e:
                                print(f">> [Tool Error]: {e}")
                                messages.append({
                                    "role": "tool",
                                    "name": tool_name,
                                    "tool_call_id": tool_call["id"],
                                    "content": f"Error: {str(e)}"
                                })
                        continue 

                    messages.append({"role": "assistant", "content": msg.get("content", "")})
                    
                    user_input = input("\n[You]: ")
                    if user_input.lower() in ['exit', 'quit']:
                        print("Exiting...")
                        break
                        
                    messages.append({"role": "user", "content": user_input})

if __name__ == "__main__":
    asyncio.run(main())
