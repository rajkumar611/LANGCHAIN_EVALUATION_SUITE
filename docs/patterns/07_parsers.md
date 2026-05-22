# 7. Output Parsing

## What we're doing here

Output parsing transforms unstructured LLM text into structured, usable data formats. This example demonstrates multiple parsers:
- **String Parser**: Extract plain text
- **JSON Parser**: Parse structured JSON
- **Comma-Separated List Parser**: Convert text into list items

Parsing ensures the LLM output is in the format your application expects.

## Why LangChain?

- **Format Validation**: Ensures output matches expected schema
- **Error Recovery**: Automatic retry with corrected prompts on parse failures
- **Multiple Formats**: Support for JSON, lists, key-value, enums, and more
- **Type Coercion**: Converts string output to proper types
- **Composability**: Chain parsers together
- **Prompt Fixing**: Can include failed output in a retry prompt to let LLM self-correct
- **Standardization**: Consistent interface across different LLM providers
- **Streaming**: Works with both batch and streaming outputs

## Without LangChain - Alternatives and Cons

### Manual JSON Parsing
```python
# Without LangChain
import json
import re

def extract_json(llm_output):
    # Fragile regex to extract JSON
    match = re.search(r'\{.*\}', llm_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {"error": "Invalid JSON"}
    return None
```

**Cons:**
- Fragile regex patterns that break with different LLM outputs
- No error recovery (fails completely on invalid JSON)
- Manual error handling for each format
- No validation against expected schema
- Difficult to implement retry logic
- LLM may return valid JSON in unexpected location/format
- No streaming support
- Hard to compose multiple parsers
- Whitespace and quote handling is error-prone

### Pydantic Validation Only
Using Pydantic directly still requires:
- Manual prompting to ensure LLM outputs valid JSON
- No built-in retry with corrected prompts
- Poor error messages to the LLM for fixing output
- Complex prompt engineering to achieve correct format

### Custom String Processing
Pattern matching with string operations:
- Extremely brittle and fragile
- Hard to handle variations in LLM output
- No schema validation
- Error messages aren't helpful to the LLM
- Scaling to complex schemas is impractical
