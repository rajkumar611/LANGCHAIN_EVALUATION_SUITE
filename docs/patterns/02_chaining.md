# 2. Chaining (Sequential Workflows)

## What we're doing here

Chaining allows you to connect multiple steps together in a sequence, where the output of one step becomes the input to the next. This example demonstrates a 3-step pipeline:
1. **Translate** — translate text to a target language
2. **Summarize** — create a concise summary
3. **Format as JSON** — structure the output as JSON

## Why LangChain?

- **Composition**: Combine multiple operations into a single logical workflow
- **Error Handling**: Built-in error management across all steps
- **Automatic Piping**: Output automatically flows to the next step (via `|` operator)
- **Intermediate Inspection**: Easily log or inspect outputs between steps
- **Modularity**: Each step is independent and testable
- **Abstraction**: Hides complexity of manual state management
- **Debugging**: Track data flow through the entire chain

## Without LangChain - Alternatives and Cons

### Manual Sequential Calls
```python
# Without LangChain
def process_text(text, target_lang):
    translated = translate_with_llm(text, target_lang)
    summary = summarize_with_llm(translated)
    json_output = format_as_json_with_llm(summary)
    return json_output
```

**Cons:**
- Manual management of outputs and inputs between steps
- No standardized way to handle step failures
- Difficult to add logging or monitoring
- Hard to reuse individual steps in different chains
- No built-in support for conditional routing (if/else flows)
- Code becomes tangled with error handling for each step
- Difficult to parallelize or optimize chains
- State management becomes your responsibility

### Raw API Calls
Making direct API calls without a chain abstraction requires you to manually handle:
- Connection pooling
- Error retries
- Token management
- Output parsing
- Flow control
