# Sample Query Guide

Upload **doc1** or **doc2** via the Upload button, then try the queries below.
Each section explains *why* that query exercises the specific strategy.

---

## Doc 1 — AI Research Report (`doc1_ai_research_report.txt`)

### Naive RAG
> Best for: simple, single-chunk factual lookups. Baseline comparison.

- "What faithfulness score did Athena-v2 achieve?"
- "What embedding model does Athena-v2 use?"
- "What is the default value of k in Reciprocal Rank Fusion?"

---

### Advanced RAG
> Best for: ambiguous or paraphrased questions where query rewriting changes which chunks are retrieved.

- "Why did the old system fail on long contracts?" *(paraphrase of truncation problem — rewrite finds the Verizon Capital incident)*
- "How does the system decide whether to use keyword or semantic search?" *(ambiguous — rewrite surfaces the query classifier logic)*
- "What happens after retrieval before the answer is generated?" *(vague — rewrite targets the reranking stage)*

---

### Hybrid RAG
> Best for: questions mixing exact terminology (acronyms, IDs, numbers) with conceptual meaning. BM25 catches the keywords; dense catches the semantics.

- "What is the NDCG@10 score achieved by RRF with k=60?" *(exact metric name + number — BM25 wins)*
- "How does the cross-encoder reranker improve result quality?" *(conceptual — dense wins)*
- "What was the hallucination rate before and after Athena-v2?" *(mixed: exact numbers + conceptual comparison)*

---

### Agentic RAG
> Best for: questions requiring multiple separate searches to piece together a complete answer. Watch the steps panel — you'll see multiple tool calls.

- "What are the three problems that motivated the RAG architecture, and how does each one map to a specific solution in Athena-v3?" *(requires: Section 2 for problems + Section 6 for solutions)*
- "Compare the latency of Athena-v2 on a 300-chunk corpus versus a 3,000-chunk corpus and explain why the difference is sub-linear." *(requires: benchmark section + FAISS explanation)*
- "What percentage of AEB-v2 failures were due to multi-hop reasoning, and what is Athena-v3's proposed fix?" *(requires: limitations section + roadmap section)*

---

### Graph RAG
> Best for: questions about sequential or causally connected concepts. Graph RAG does 2-hop BFS from seed chunks so it finds adjacent context.

- "Walk me through what happens to a document from the moment it is uploaded to when an answer is generated." *(spans all 5 pipeline stages sequentially)*
- "How does the ingestion stage influence retrieval quality?" *(ingestion → indexing → retrieval — 3 sequential hops)*
- "What limitation led to the reranking stage being added, and what did that fix downstream?" *(signal-to-noise → reranking → generation improvement)*

---

## Doc 2 — Product Operations Manual (`doc2_product_operations_manual.txt`)

### Naive RAG
> Simple policy lookups — single chunk has the full answer.

- "What is the monthly fee for the Professional tier?"
- "How long does a password reset link stay valid?"
- "What is the minimum data rows required before InsightEngine produces predictions?"

---

### Advanced RAG
> Paraphrased or vague questions — query rewriting finds the right policy section.

- "Can a client cancel their subscription mid-year?" *(paraphrase of downgrade/cancellation policy)*
- "What should I do if my pipeline keeps stopping on bad rows?" *(paraphrase of ERR-DB-001 and error policy)*
- "How do I get credit if the platform goes down?" *(paraphrase of SLA breach credit claim process)*

---

### Hybrid RAG
> Mix of exact codes/SKUs and conceptual policy questions.

- "What does error code ERR-DB-002 mean and how do I fix it?" *(exact code — BM25; resolution steps — dense)*
- "Which SKU do I order for API overage and how much does it cost?" *(SKU: CLX-API-OVR — exact keyword + pricing context)*
- "What is the SLA uptime guarantee for Enterprise clients?" *(exact tier name + percentage)*

---

### Agentic RAG
> Multi-part questions that require combining information from separate sections.

- "A Professional tier client sees INS-002 and also complains their dashboard is timing out. What are the exact steps to resolve both issues?" *(requires: InsightEngine section + Troubleshooting section)*
- "If a Starter tier client's platform is down 98% of the month, what credit do they receive and how do they claim it, and by when?" *(requires: SLA penalties section + credit claim procedure)*
- "What is the full escalation path for a P1 incident at an Enterprise client, and what response time should they expect at each level?" *(requires: incident classification + escalation path + Enterprise SLA)*

---

### Graph RAG
> Questions about end-to-end operational flows — adjacent sections connected sequentially.

- "Trace the full journey of a data row from ingestion through transformation to an InsightEngine prediction." *(DataBridge → transformation rules → InsightEngine — sequential flow)*
- "How does the error policy setting affect what happens downstream when a pipeline fails?" *(error policy → ERR-DB-001 → quarantine table → support action)*
- "What happens operationally from the moment a P2 incident is detected to when it is closed?" *(incident classification → escalation → PIR — sequential procedure)*

---

## Evaluation Queries (use /rag/evaluate after getting an answer)

After running any query above, use these fields in the Evaluate tab to score the answer:

| Field | What to put |
|---|---|
| Question | The query you asked |
| Answer | The answer the RAG returned |
| Contexts | Paste the retrieved chunk text shown in the response |
| Ground Truth (optional) | The actual correct answer from the document |

**Good evaluation test:** Run the same question through Naive RAG and Advanced RAG, then evaluate both answers. Compare faithfulness and context utilisation scores — Advanced should score higher for the ambiguous queries.
