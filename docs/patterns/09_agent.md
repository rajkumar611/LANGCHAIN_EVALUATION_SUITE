# 9. Agents (ReAct Pattern)

## What we're doing here

Agents use the **ReAct** (Reasoning + Acting) pattern to solve problems iteratively:
1. **Think**: LLM reasons about what action to take
2. **Act**: LLM calls a tool (from its available tools)
3. **Observe**: Receive tool output
4. **Repeat**: Continue until the goal is achieved or tool limit reached

This example uses LangGraph's built-in agent that combines reasoning with tool use. LangChain provides the LLM interface and tools; LangGraph provides the agentic orchestration.

## Why LangGraph?

- **Autonomous Decision Making**: LLM decides when and which tools to use
- **Error Recovery**: Gracefully handles tool errors and invalid calls
- **Multi-Step Reasoning**: Maintains state and memory across multiple steps
- **Tool Management**: Automatically manages tool availability and validation
- **Max Iterations**: Built-in limits to prevent infinite loops
- **Transparency**: Can log reasoning and actions for debugging
- **Flexibility**: Works with any set of tools
- **Production Ready**: Built-in safety guards and error handling
- **Streaming Support**: Can stream reasoning and actions in real-time

## Without LangGraph - Alternatives and Cons

### LangChain Only (Manual Orchestration)
```python
# Using LangChain without LangGraph - manual orchestration
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json
import re

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

def manual_agent_loop(question: str, max_iterations: int = 5):
    model = ChatAnthropic(model="claude-haiku-4-5-20251001")
    tools = [calculator]
    
    # Bind tools to model
    model_with_tools = model.bind_tools(tools)
    messages = [HumanMessage(content=question)]
    
    for i in range(max_iterations):
        # Get LLM response
        response = model_with_tools.invoke(messages)
        messages.append(response)
        
        # Check if model wants to use tools
        if not response.tool_calls:
            return response.content  # Final answer
        
        # Manually process each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            
            # Find and execute tool (fragile - no built-in routing)
            if tool_name == "calculator":
                result = calculator.invoke(tool_input)
            else:
                result = "Tool not found"
            
            # Add tool result to messages (manual message construction)
            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            ))
    
    return "Max iterations reached"
```

**Cons:**
- **Manual tool routing**: You must hardcode `if tool_name == "calculator"` for each tool
- **No built-in message validation**: Messages must be constructed correctly by hand
- **Error handling scattered**: Each tool call needs try/except; no centralized error recovery
- **State management burden**: You track `messages`, `iteration`, `tool_calls` manually
- **No streaming support**: Must wait for full response before processing
- **Fragile tool parsing**: If LLM output changes format, parsing breaks
- **No built-in stopping conditions**: Must implement your own "stop if no tools called" logic
- **Difficult to debug**: No visibility into agent decision-making; must log messages yourself
- **Token management**: Your responsibility to manage context window and truncate history
- **No memory persistence**: Conversation state lives only in the loop; no session support

### External Workflow Engine
Using a separate workflow system (Airflow, etc.) requires:
- Separate infrastructure
- Complex serialization of LLM calls
- Loss of tight integration with LLM reasoning
- Operational overhead
- No streaming support
