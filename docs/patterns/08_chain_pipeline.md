# 8. Sequential Chain Pipeline

## What we're doing here

Sequential chain pipelines connect multiple LLM calls in sequence, where each step builds on the output of the previous one:
1. **Researcher**: Gathers information on a topic
2. **Writer**: Creates polished content based on research

This pattern combines the outputs of multiple steps to create more sophisticated solutions than a single LLM call could achieve.

## Why LangChain?

- **Clear Separation**: Each step focuses on a specific task
- **Pipeline Composition**: Easily chain step outputs to inputs
- **Specialization**: Can optimize prompts for each step's role
- **Quality Improvement**: Multiple passes improve output quality
- **Error Isolation**: Failures in one step don't necessarily break others
- **Monitoring**: Easy to log and inspect intermediate outputs
- **Flexibility**: Add, remove, or reorder steps easily
- **Extensibility**: Natural foundation for more complex workflows (with LangGraph)

## Without LangChain - Alternatives and Cons

### Sequential Manual Calls
```python
# Without LangChain
def research_and_write(topic):
    # First LLM call: research
    research_prompt = f"Research the topic: {topic}"
    research = llm_call(research_prompt)
    
    # Second LLM call: write
    write_prompt = f"Write a blog post based on: {research}"
    blog_post = llm_call(write_prompt)
    
    return blog_post
```

**Cons:**
- No abstraction or modularity (hard to reuse agents)
- Manual prompt engineering for each agent
- Difficult to add error handling between steps
- Hard to log intermediate results
- No standardized interface between agents
- Scaling to more agents becomes unmaintainable
- Difficult to test individual agents independently
- Error in one step breaks the entire workflow
- No built-in optimization or caching
- Hard to implement agent-specific configurations

### Independent Scripts
Running agents as separate scripts:
- Manual data passing between scripts
- Complex orchestration
- Difficult to maintain consistent state
- Poor error handling across processes
- Hard to implement timeouts and retries

### Raw Distributed System
Building your own multi-agent system requires:
- Message queuing infrastructure
- State management across agents
- Complex serialization
- Network communication overhead
- Difficult debugging and monitoring
