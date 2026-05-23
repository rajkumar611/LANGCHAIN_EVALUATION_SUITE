# 10. Workflows (Advanced State Machines)

## What we're doing here

LangGraph enables building complex, stateful AI workflows using a graph-based architecture:
1. **State Graph**: Define a shared state structure all nodes can access
2. **Nodes**: Functions that process state (researcher, writer, reviewer)
3. **Edges**: Transitions between nodes (including conditional routing)
4. **Cycles**: Support for iterative refinement (e.g., max 2 revision rounds)

This example demonstrates a content creation workflow with feedback loops and conditional routing.

## Why LangChain?

- **State Management**: Explicit, typed state that all nodes access
- **Graph Visualization**: Visualize your workflow as a graph
- **Conditional Routing**: Route based on state values (`if review_good then END else REVISE`)
- **Iteration Support**: Elegantly handle loops and multiple attempts
- **Type Safety**: Pydantic models define the state schema
- **Streaming**: Stream intermediate node outputs
- **Persistence**: Easy to add checkpointing and recovery
- **Debugging**: Clear visibility into state transitions
- **Production Ready**: Built for complex, reliable AI systems
- **Human-in-the-Loop**: Easy to add approval steps

## Without LangChain - Alternatives and Cons

### Manual State Management
```python
# Without LangChain
def workflow(topic):
    state = {
        "topic": topic,
        "research": None,
        "draft": None,
        "reviews": 0,
        "needs_revision": True
    }
    
    while state["needs_revision"] and state["reviews"] < 2:
        # Researcher node
        state["research"] = research_agent(state["topic"])
        
        # Writer node
        state["draft"] = writer_agent(state["research"])
        
        # Reviewer node
        review = reviewer_agent(state["draft"])
        state["reviews"] += 1
        state["needs_revision"] = review["needs_revision"]
    
    return state["draft"]
```

**Cons:**
- Manual state management becomes complex with multiple nodes
- Hard to visualize the workflow
- Difficult to implement proper conditional routing
- Manual loop handling is error-prone
- No type safety for state structure
- Debugging is painful (where did state get corrupted?)
- No built-in support for human-in-the-loop steps
- Scaling to complex workflows is impractical
- No checkpointing or recovery from failures
- Hard to add observability and logging
- Testing individual nodes is difficult

### Raw Graph Libraries (NetworkX, etc.)
Using a general graph library requires:
- Manual LLM integration
- Complex state passing between nodes
- No built-in streaming or async support
- Significant boilerplate code
- Loss of AI-specific optimizations

### Workflow Orchestration Tools (Airflow, Prefect)
While these work for traditional workflows:
- Heavy operational overhead
- Not designed for tight LLM integration
- Difficult to handle streaming outputs
- Complex deployment requirements
- Overkill for many use cases
