# 8. Agents (ReAct Pattern)

## What we're doing here

Agents use the **ReAct** (Reasoning + Acting) pattern to solve problems iteratively:
1. **Think**: LLM reasons about what action to take
2. **Act**: LLM calls a tool (from its available tools)
3. **Observe**: Receive tool output
4. **Repeat**: Continue until the goal is achieved or tool limit reached

This example uses LangGraph's built-in agent that combines reasoning with tool use.

## Why LangChain?

- **Autonomous Decision Making**: LLM decides when and which tools to use
- **Error Recovery**: Gracefully handles tool errors and invalid calls
- **Multi-Step Reasoning**: Maintains state and memory across multiple steps
- **Tool Management**: Automatically manages tool availability and validation
- **Max Iterations**: Built-in limits to prevent infinite loops
- **Transparency**: Can log reasoning and actions for debugging
- **Flexibility**: Works with any set of tools
- **Production Ready**: Built-in safety guards and error handling
- **Streaming Support**: Can stream reasoning and actions in real-time

## Without LangChain - Alternatives and Cons

### Manual ReAct Loop
```python
# Without LangChain
def agent_loop(goal, tools, max_iterations=5):
    history = [{"role": "user", "content": goal}]
    
    for i in range(max_iterations):
        # Get LLM reasoning
        response = llm_call(history)
        
        # Manually parse tool calls (fragile)
        tool_name = parse_tool_name(response)
        params = parse_params(response)
        
        # Manually execute tool
        if tool_name not in tools:
            tool_result = "Tool not found"
        else:
            tool_result = tools[tool_name](**params)
        
        # Manually update history
        history.append({"role": "assistant", "content": response})
        history.append({"role": "user", "content": f"Tool result: {tool_result}"})
    
    return history[-1]["content"]
```

**Cons:**
- Extremely complex state management
- Fragile parsing of LLM tool calls
- Manual error handling for each step
- Difficult to debug multi-step reasoning
- No built-in support for memory across conversations
- Hard to implement proper iteration limits
- Token management becomes complex with long histories
- Difficult to implement stopping conditions elegantly
- Tool validation is manual and error-prone
- Scaling beyond simple cases requires extensive code

### External Workflow Engine
Using a separate workflow system (Airflow, etc.) requires:
- Separate infrastructure
- Complex serialization of LLM calls
- Loss of tight integration with LLM reasoning
- Operational overhead
- No streaming support
