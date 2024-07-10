import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.environ.get("openai_api_key")
)

llm_default_params = llm._default_params
print(f"\n=== Chat Completion LLM Parameters ===\n")
for k,v in llm_default_params.items():
    print(f"{k:20}{v}")
print()

messages=[
    (
        "system", "You are a comedian who tells jokes."
    ),
    ("Tell me a joke!")
]

ai_msg = llm.invoke(messages)
print(f"=== Funny Lines ===\n{ai_msg.content}")

messages=[
    (
        "system", "You are a thespian."
    ),
    ("Give me a quote from Shakespeare that you like.")
]

print()

ai_msg = llm.invoke(messages)
print(f"=== The Bard Sayeth ===\n{ai_msg.content}")

messages=[
    (
        "system", "You are a money changer."
    ),
    ("What is the exchange rate for Japanese Yen to US Dollar today? ")
]

print()

ai_msg = llm.invoke(messages)
print(f"=== Yen:USD Exchange ===\n{ai_msg.content}")

messages=[
    (
        "system", "Ruby is a technology entrepreneur. She''s keen to make her start up a success."
    ),
    ("Tell me what her typical working looks like.")
]

print()

ai_msg = llm.invoke(messages)
print(f"=== Tech Entrepreneur ===\n{ai_msg.content}")