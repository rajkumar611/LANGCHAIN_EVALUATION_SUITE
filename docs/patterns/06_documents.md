# 6. Document Splitting & Text Chunking

## What we're doing here

Document splitting breaks large texts into manageable chunks for processing. Different splitting strategies suit different use cases:
- **Character Splitter**: Naive splitting by character count (fast but may break words)
- **Recursive Character Splitter**: Splits hierarchically by natural boundaries (paragraphs, sentences) while respecting token limits

This pattern is essential for RAG, indexing, and processing long documents.

## Why LangChain?

- **Multiple Strategies**: Different splitters for different document types
- **Overlap Support**: Configurable overlap to preserve context between chunks
- **Token-Aware**: Built-in support for token counting (not just character counting)
- **Recursive Splitting**: Intelligently preserves semantic structure
- **Separator Customization**: Choose how to split based on separators
- **Extensibility**: Easy to implement custom splitters
- **Performance**: Optimized splitting logic handles large documents efficiently
- **Format Agnostic**: Same API works for plain text, markdown, code, structured data

## Without LangChain - Alternatives and Cons

### Simple String Slicing
```python
# Without LangChain
def split_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks
```

**Cons:**
- Breaks mid-word or mid-sentence (poor semantics)
- No respect for document structure (paragraphs, sections, code blocks)
- No overlap handling (loses context between chunks)
- Difficult to implement token-aware splitting
- Can't easily adapt splitting strategy
- No support for overlapping windows
- Naive splitting can break code or structured data
- Difficult to maintain semantic coherence

### Recursive Splitting (Manual Implementation)
Building recursive splitting yourself requires:
- Complex recursive algorithms to track separators
- Manual token counting implementation
- Testing edge cases (empty documents, short texts)
- Handling different separator priorities
- Performance optimization for large documents

### Regular Expression Splitting
Regex-based splitting is brittle:
- Different document types need different patterns
- Hard to maintain regex patterns
- Poor performance on large texts
- Difficult to handle overlaps
