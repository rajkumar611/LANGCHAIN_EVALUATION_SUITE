# 5. Tool Calling (Function Calling)

## What we're doing here

Tool calling allows LLMs to request external tools to accomplish tasks they can't do alone. The LLM can:
1. **Recognize** when a tool is needed
2. **Call** the appropriate tool with the right parameters
3. **Receive** the tool's output
4. **Continue** reasoning with that information

In this example, the LLM can use tools like a calculator, weather service, or word counter to enhance its capabilities.

## Why LangChain?

- **Type Safety**: Pydantic models automatically validate tool inputs
- **Tool Binding**: Seamless integration between LLM and tools via `bind_tools()`
- **Parameter Schema**: Automatic generation of correct parameter schemas for the LLM
- **Tool Orchestration**: Handle multiple tools without manual if/else statements
- **Extensibility**: Add custom tools with just a function + docstring
- **Error Handling**: Built-in validation prevents invalid tool calls
- **Standardization**: Works consistently across different LLM providers
- **Agentic Loops**: Natural foundation for ReAct agents and autonomous workflows

## Without LangChain - Alternatives and Cons

### Manual Tool Dispatching
```python
# Without LangChain
tools_registry = {
    "calculator": calculator_fn,
    "weather": weather_fn,
    "word_count": word_count_fn
}

def use_tools(task):
    # You must manually parse LLM output to extract tool calls
    response = llm_call(f"Use tools to: {task}")
    
    # Manual parsing of tool names and parameters
    tool_name = extract_tool_name(response)  # regex/heuristics
    params = extract_params(response)  # custom parsing
    
    if tool_name in tools_registry:
        result = tools_registry[tool_name](**params)
        return result
    else:
        return "Unknown tool"
```

**Cons:**
- Manual parsing of LLM output is fragile (LLMs are unpredictable in formatting)
- No type validation of tool parameters
- No standardized schema for tool descriptions
- Hard to handle tool errors gracefully
- Difficult to scale beyond a few tools
- No built-in retry logic for tool failures
- Parameter passing is error-prone (wrong types, missing args)
- Hard to generate correct tool descriptions for the LLM
- Building agentic loops requires extensive custom code

### Raw API with Tool Definitions
Creating tool schemas manually (as JSON) requires:
- Deep knowledge of OpenAI/Claude function calling formats
- Manual validation of parameters
- Complex state management for multi-turn tool use
- No higher-level abstractions for common patterns
