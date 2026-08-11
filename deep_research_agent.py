from langchain_ollama import ChatOllama


# ============================================================
# EXPERIMENT 07
# DEEP RESEARCH AGENT WORKFLOW
# ============================================================

print("=" * 70)
print("             DEEP RESEARCH AGENT WORKFLOW")
print("=" * 70)


# ============================================================
# 1. LOAD LOCAL LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ============================================================
# 2. PLANNING AGENT
# ============================================================

def planning_agent(topic):

    print("\n[AGENT 1] PLANNING")
    print("-" * 70)

    prompt = f"""
You are a research planning agent.

Research topic:
{topic}

Create a research plan for this topic.

Break the topic into exactly 5 important research questions.

For each question provide:
- Question
- Why it is important

Rules:
- Keep the questions focused.
- Avoid duplicate questions.
- Do not answer the questions yet.
"""

    response = llm.invoke(prompt)

    print(response.content)

    return response.content


# ============================================================
# 3. RESEARCH AGENT
# ============================================================

def research_agent(topic, research_plan):

    print("\n[AGENT 2] RESEARCH")
    print("-" * 70)

    prompt = f"""
You are a research analysis agent.

Research topic:
{topic}

Research plan:
{research_plan}

For each research question:

1. Provide a concise finding.
2. Explain the important concept.
3. Mention important advantages or limitations where relevant.

Important:
- Use general knowledge available to the language model.
- Do not invent citations, URLs, papers, statistics, or named sources.
- Clearly state when information is uncertain.
- Keep the findings factual and academic.
"""

    response = llm.invoke(prompt)

    print(response.content)

    return response.content


# ============================================================
# 4. REFLECTION AGENT
# ============================================================

def reflection_agent(topic, research_plan, findings):

    print("\n[AGENT 3] REFLECTION")
    print("-" * 70)

    prompt = f"""
You are a reflection and quality-control agent.

Research topic:
{topic}

Research plan:
{research_plan}

Research findings:
{findings}

Review the research carefully.

Check:

1. Did all research questions receive an answer?
2. Are there missing important concepts?
3. Are there contradictions?
4. Are there unsupported or overly confident claims?
5. What should be improved?

Produce:

COMPLETENESS: HIGH / MEDIUM / LOW

STRENGTHS:
- ...

GAPS:
- ...

IMPROVEMENTS:
- ...

Do not invent facts or citations.
"""

    response = llm.invoke(prompt)

    print(response.content)

    return response.content


# ============================================================
# 5. WRITER AGENT
# ============================================================

def writer_agent(topic, findings, reflection):

    print("\n[AGENT 4] REPORT WRITER")
    print("-" * 70)

    prompt = f"""
You are an academic report writer.

Topic:
{topic}

Research findings:
{findings}

Reflection:
{reflection}

Write a clear final research report.

Use this structure:

# Research Topic

## 1. Introduction

## 2. Key Concepts

## 3. Detailed Findings

## 4. Advantages

## 5. Limitations

## 6. Future Scope

## 7. Conclusion

Requirements:

- Use clear academic language.
- Organize information logically.
- Incorporate useful improvements identified during reflection.
- Do not invent citations or references.
- Do not claim that web research was performed.
"""

    response = llm.invoke(prompt)

    print(response.content)

    return response.content


# ============================================================
# 6. COMPLETE DEEP RESEARCH WORKFLOW
# ============================================================

def deep_research(topic):

    # Step 1
    research_plan = planning_agent(topic)

    # Step 2
    findings = research_agent(
        topic,
        research_plan
    )

    # Step 3
    reflection = reflection_agent(
        topic,
        research_plan,
        findings
    )

    # Step 4
    final_report = writer_agent(
        topic,
        findings,
        reflection
    )

    return final_report


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    topic = input(
        "\nEnter a research topic: "
    )

    print("\nStarting deep research workflow...")

    final_report = deep_research(topic)

    print("\n" + "=" * 70)
    print("FINAL RESEARCH REPORT")
    print("=" * 70)

    print(final_report)

    print("\n" + "=" * 70)
    print("DEEP RESEARCH WORKFLOW COMPLETED")
    print("=" * 70)