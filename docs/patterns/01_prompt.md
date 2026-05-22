# 1. Prompt Templates

## What we're doing here

Prompt templates allow you to create reusable message templates with dynamic variables. Instead of hardcoding entire prompts as strings, you define a template once and reuse it with different input values.

In this example, we take a user's topic and dynamically insert it into a prompt template to create a greeting message.

## Why LangChain?

- **Reusability**: Write the prompt once, use it multiple times with different inputs
- **Maintainability**: Change the prompt logic in one place, applies everywhere
- **Type Safety**: LangChain validates that all required variables are provided
- **Composability**: Templates can be easily chained together in larger workflows
- **Language Flexibility**: Supports both string and chat message formats
- **Input Validation**: Automatic validation of input variables using Pydantic models

## Without LangChain - Alternatives and Cons

### Manual String Formatting
```python
# Without LangChain
def greet(topic):
    return f"Provide a concise greeting for someone interested in {topic}"
```

**Cons:**
- Hard to maintain when prompts become complex
- Prone to breaking if you forget to pass required variables
- No validation of inputs
- Difficult to reuse the same prompt in different contexts
- Easy to introduce bugs with string formatting (f-strings, .format(), %)
- No built-in support for multi-part messages
- Scaling becomes difficult as you add more prompts

### Template Engines (Jinja2)
While you could use Jinja2, it adds another dependency and requires manual error handling for missing variables. LangChain provides a cleaner, more integrated approach with better error messages.
