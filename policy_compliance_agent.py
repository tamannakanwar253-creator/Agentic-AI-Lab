from langchain_ollama import ChatOllama
import json


# ============================================================
# EXPERIMENT 06
# POLICY COMPLIANCE AGENT
# ============================================================

print("=" * 70)
print("              POLICY COMPLIANCE AGENT")
print("=" * 70)


# ============================================================
# 1. LOAD LOCAL LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ============================================================
# 2. POLICY RULES
# ============================================================

POLICY_RULES = {
    "rule_1": {
        "name": "Maximum transaction amount",
        "description": "Transactions above $10,000 require manual review."
    },

    "rule_2": {
        "name": "International transaction",
        "description": "International transactions require additional verification."
    },

    "rule_3": {
        "name": "Restricted category",
        "description": "Transactions involving restricted categories must be rejected."
    },

    "rule_4": {
        "name": "Missing customer information",
        "description": "Transactions with missing customer information require manual review."
    }
}


# ============================================================
# 3. DISPLAY POLICY
# ============================================================

def display_policy():

    print("\nPOLICY RULES")
    print("-" * 70)

    for rule_id, rule in POLICY_RULES.items():

        print(
            f"{rule_id}: "
            f"{rule['name']} - "
            f"{rule['description']}"
        )


# ============================================================
# 4. RULE-BASED EVALUATION
# ============================================================

def rule_based_evaluation(transaction):

    reasons = []

    decision = "APPROVED"

    # --------------------------------------------------------
    # Rule 1
    # --------------------------------------------------------

    if transaction["amount"] > 10000:

        reasons.append(
            "Transaction amount exceeds $10,000."
        )

        decision = "REVIEW"


    # --------------------------------------------------------
    # Rule 2
    # --------------------------------------------------------

    if transaction["international"]:

        reasons.append(
            "International transaction requires additional verification."
        )

        decision = "REVIEW"


    # --------------------------------------------------------
    # Rule 3
    # --------------------------------------------------------

    if transaction["category"].lower() == "restricted":

        reasons.append(
            "Transaction belongs to a restricted category."
        )

        decision = "REJECTED"


    # --------------------------------------------------------
    # Rule 4
    # --------------------------------------------------------

    if not transaction["customer_name"].strip():

        reasons.append(
            "Customer information is missing."
        )

        if decision != "REJECTED":
            decision = "REVIEW"


    return decision, reasons


# ============================================================
# 5. AI COMPLIANCE ANALYSIS
# ============================================================

def ai_compliance_analysis(transaction, rule_decision, reasons):

    prompt = f"""
You are a Policy Compliance Agent.

Evaluate the following synthetic transaction using the supplied
company policy.

TRANSACTION:

Customer: {transaction['customer_name']}
Amount: ${transaction['amount']}
Category: {transaction['category']}
International: {transaction['international']}

POLICY:

1. Transactions above $10,000 require manual review.
2. International transactions require additional verification.
3. Restricted categories must be rejected.
4. Missing customer information requires manual review.

RULE-BASED RESULT:

Decision: {rule_decision}

Reasons:
{reasons}

Your task:

1. Review the transaction.
2. Explain whether the rule-based decision is appropriate.
3. Identify which policy rules apply.
4. Give a final decision.

The final decision MUST be one of:

APPROVED
REVIEW
REJECTED

Do not invent information.
Do not create additional policies.
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# 6. COMPLETE COMPLIANCE AGENT
# ============================================================

def compliance_agent(transaction):

    print("\n" + "=" * 70)
    print("TRANSACTION EVALUATION")
    print("=" * 70)

    print("\nTransaction:")
    print(json.dumps(transaction, indent=4))

    # --------------------------------------------------------
    # Rule engine
    # --------------------------------------------------------

    print("\n[STEP 1] Rule-Based Evaluation")
    print("-" * 70)

    rule_decision, reasons = rule_based_evaluation(
        transaction
    )

    print("Rule-Based Decision:", rule_decision)

    if reasons:

        print("\nReasons:")

        for reason in reasons:
            print("-", reason)

    else:

        print("No policy violations detected.")


    # --------------------------------------------------------
    # LLM agent
    # --------------------------------------------------------

    print("\n[STEP 2] AI Compliance Analysis")
    print("-" * 70)

    ai_result = ai_compliance_analysis(
        transaction,
        rule_decision,
        reasons
    )

    print(ai_result)


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPLIANCE EVALUATION COMPLETED")
    print("=" * 70)


# ============================================================
# 7. SYNTHETIC TEST DATA
# ============================================================

transactions = [

    {
        "customer_name": "Rahul Sharma",
        "amount": 5000,
        "category": "software",
        "international": False
    },

    {
        "customer_name": "Priya Rao",
        "amount": 15000,
        "category": "software",
        "international": False
    },

    {
        "customer_name": "Arjun Kumar",
        "amount": 8000,
        "category": "software",
        "international": True
    },

    {
        "customer_name": "Sneha Patel",
        "amount": 3000,
        "category": "restricted",
        "international": False
    },

    {
        "customer_name": "",
        "amount": 4000,
        "category": "software",
        "international": False
    }
]


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    display_policy()

    print("\n")
    print("=" * 70)
    print("TESTING SYNTHETIC TRANSACTIONS")
    print("=" * 70)

    for index, transaction in enumerate(
        transactions,
        start=1
    ):

        print(f"\n\nTEST CASE {index}")

        compliance_agent(transaction)