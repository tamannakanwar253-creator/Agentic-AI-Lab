from langchain_ollama import ChatOllama


# ============================================================
# EXPERIMENT 03
# PROMPT CHAINING FOR SUMMARIZATION
# ============================================================

print("=" * 70)
print("        PROMPT CHAINING FOR SUMMARIZATION")
print("=" * 70)


# ============================================================
# 1. LOAD LOCAL LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ============================================================
# SAMPLE DOCUMENT
# ============================================================

document = """
Artificial Intelligence is becoming an important technology across
many industries. Modern AI systems can analyze large amounts of data,
understand natural language, generate content, and assist people with
complex tasks.

Large Language Models are a major development in AI. These models are
trained on large collections of text and can perform tasks such as
question answering, summarization, translation, classification, and
content generation.

Retrieval-Augmented Generation combines information retrieval with
language generation. A RAG system retrieves relevant information from
external knowledge sources and provides that information to a language
model as context.

Agentic AI extends these capabilities by allowing AI systems to use
tools, maintain memory, make plans, and perform multiple actions.
Agents can interact with APIs, databases, search systems, and other
software tools.

Multi-agent systems use multiple specialized agents that collaborate
to solve complex problems. For example, one agent can perform research,
another can evaluate the information, and another can generate the
final response.

These technologies are increasingly being used in education, healthcare,
finance, software development, customer support, and business automation.
"""


# ============================================================
# STEP 1 — EXTRACT KEY POINTS
# ============================================================

print("\n[STEP 1] Extracting key points...")
print("-" * 70)

prompt_1 = f"""
You are an information extraction assistant.

Read the following document:

{document}

Extract the most important ideas from the document.

Rules:
- Provide 5 to 8 key points.
- Each point should be concise.
- Do not add information that is not present in the document.
"""

response_1 = llm.invoke(prompt_1)

key_points = response_1.content

print(key_points)


# ============================================================
# STEP 2 — ORGANIZE THE INFORMATION
# ============================================================

print("\n[STEP 2] Organizing key points...")
print("-" * 70)

prompt_2 = f"""
You are an information organization assistant.

The following key points were extracted from a document:

{key_points}

Organize these points into logical categories.

Use categories such as:

1. Artificial Intelligence
2. Large Language Models
3. RAG
4. Agentic AI
5. Multi-Agent Systems
6. Applications

Only use information contained in the key points.
Do not invent new facts.
"""

response_2 = llm.invoke(prompt_2)

organized_points = response_2.content

print(organized_points)


# ============================================================
# STEP 3 — GENERATE FINAL SUMMARY
# ============================================================

print("\n[STEP 3] Generating final summary...")
print("-" * 70)

prompt_3 = f"""
You are a professional summarization assistant.

Use the following organized information:

{organized_points}

Write a clear final summary of the original document.

Requirements:

- Write 1 to 2 paragraphs.
- Keep the important technical concepts.
- Use simple academic language.
- Do not add information that is not present.
"""

response_3 = llm.invoke(prompt_3)

final_summary = response_3.content

print(final_summary)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("PROMPT CHAINING COMPLETED")
print("=" * 70)

print("\nFinal Summary:")
print(final_summary)