import sqlite3
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent


# ============================================================
# CONFIGURATION
# ============================================================

DB_NAME = "company.db"


# ============================================================
# 1. CREATE DATABASE
# ============================================================

connection = sqlite3.connect(DB_NAME)
cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary INTEGER NOT NULL,
    experience INTEGER NOT NULL
)
""")

# Reset sample data
cursor.execute("DELETE FROM employees")

employees = [
    (1, "Rahul", "IT", 60000, 2),
    (2, "Priya", "HR", 50000, 3),
    (3, "Arjun", "IT", 75000, 5),
    (4, "Sneha", "Finance", 65000, 4),
    (5, "Kiran", "IT", 55000, 1),
    (6, "Ananya", "HR", 58000, 4)
]

cursor.executemany("""
INSERT INTO employees
(id, name, department, salary, experience)
VALUES (?, ?, ?, ?, ?)
""", employees)

connection.commit()


# ============================================================
# 2. DATABASE TOOLS
# ============================================================

@tool
def get_database_schema() -> str:
    """
    Returns the schema of the employees database.
    Use this before writing SQL queries when database
    structure is needed.
    """

    return """
Table: employees

Columns:
- id: employee ID
- name: employee name
- department: department name
- salary: annual salary
- experience: years of experience
"""


@tool
def execute_sql(query: str) -> str:
    """
    Executes a READ-ONLY SQLite SELECT query and returns
    the database results.
    """

    query_clean = query.strip().rstrip(";")
    query_upper = query_clean.upper()

    # Safety check
    if not query_upper.startswith("SELECT"):
        return "ERROR: Only SELECT queries are allowed."

    forbidden = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "REPLACE"
    ]

    for command in forbidden:
        if command in query_upper:
            return f"ERROR: {command} operation is not allowed."

    try:
        cursor.execute(query_clean)

        rows = cursor.fetchall()

        if not rows:
            return "No records found."

        column_names = [
            description[0]
            for description in cursor.description
        ]

        output = [str(column_names)]

        for row in rows:
            output.append(str(row))

        return "\n".join(output)

    except sqlite3.Error as error:
        return f"SQL Error: {error}"


# ============================================================
# 3. INITIALIZE LOCAL LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ============================================================
# 4. CREATE AGENT
# ============================================================

tools = [
    get_database_schema,
    execute_sql
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a SQL database agent.

Your job is to answer questions about the employees database.

Follow this process:

1. Understand the user's question.
2. Use get_database_schema when you need the database structure.
3. Create a valid SQLite SELECT query.
4. Use execute_sql to run the query.
5. Analyze the returned result.
6. If the query fails, correct the query and try again.
7. Give the user a clear final answer.

Important rules:

- Only use the employees table.
- Only perform read-only SELECT operations.
- Never modify or delete database data.
- Do not invent database results.
- Use the tools whenever database information is required.
"""
)


# ============================================================
# 5. RUN AGENT
# ============================================================

print("=" * 65)
print("             SQL AGENT WITH TOOL USE")
print("=" * 65)

print("\nDatabase: company.db")
print("Table: employees")

print("\nExample questions:")
print("1. Who has the highest salary?")
print("2. How many employees work in IT?")
print("3. What is the average salary of IT employees?")
print("4. Who has more than 3 years of experience?")

question = input("\nEnter your question: ")


# ============================================================
# 6. INVOKE AGENT
# ============================================================

print("\nAgent is working...")
print("-" * 65)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    }
)


# ============================================================
# 7. DISPLAY AGENT MESSAGES
# ============================================================

print("\n" + "=" * 65)
print("AGENT EXECUTION")
print("=" * 65)

for message in result["messages"]:

    message_type = type(message).__name__

    print(f"\n[{message_type}]")

    if hasattr(message, "content") and message.content:
        print(message.content)

    if hasattr(message, "tool_calls") and message.tool_calls:
        for tool_call in message.tool_calls:
            print("\nTool selected:")
            print("Tool:", tool_call["name"])
            print("Arguments:", tool_call["args"])


# ============================================================
# 8. FINAL ANSWER
# ============================================================

final_message = result["messages"][-1]

print("\n" + "=" * 65)
print("FINAL ANSWER")
print("=" * 65)

print(final_message.content)

print("\n" + "=" * 65)
print("SQL AGENT EXPERIMENT COMPLETED")
print("=" * 65)


# ============================================================
# 9. CLOSE DATABASE
# ============================================================

connection.close()
