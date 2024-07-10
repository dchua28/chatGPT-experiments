import os
from langchain_openai import OpenAI

# By default the OpenAI module uses gpt-3.5-turbo-instruct
# for this text completion task. It was trained up until 
# Sep 2021. See the Yen-to-USD prompt output to confirm this.

llm = OpenAI( api_key=os.environ.get("OPENAI_API_KEY"), max_tokens=512)
llm_default_params = llm._default_params
print(f"\n=== Text Completion LLM Parameters ===\n")
for k,v in llm_default_params.items():
    print(f"{k:20}{v}")
print()

print(f"=== Funny Lines ===\t{llm.invoke('Tell me a joke!')}\n")
print(f"=== The Bard Sayeth ===\t{llm.invoke('Give me a quote from Shakespeare that you like.')}\n")
print(f"=== Yen:USD Exchange ===\t{llm.invoke('What is the exchange rate for Japanese Yen to US Dollar today?')}\n")
print(f"=== Tech Entrepreneur ===\t{llm.invoke('Ruby is a technology entrepreneur. She''s keen to make her start up a success. Tell me what her typical working looks like.')}\n")
