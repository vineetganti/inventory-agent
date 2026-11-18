"""
Inventory Planner  v0  -  one prompt, one answer          (by Shivank Agarwal)
================================================================================
NEW IN THIS VERSION: the simplest possible AI app.
    We type the numbers ourselves, put them in a prompt, and ask the model
    "should I reorder?". No UI, no database, no tools.

RUN:  python inventoryplanner_v0.py
"""
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import llm as provider  # Groq or Claude, chosen by LLM_PROVIDER (see llm.py)
load_dotenv()   # reads the API keys from the .env file

# The model is chosen in llm.py (Groq gpt-oss-120b or Claude Haiku 5.5); the course used gpt-6-luna with reasoning off.
llm = provider.make_llm()

# A prompt with {blanks} that we fill in below
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an inventory management expert. Analyze inventory situations "
               "and give clear reorder recommendations."),
    ("human", """Should I reorder {product}? If so, how much?

Current stock: {current_stock} units
Average demand: {average_demand} units per day
Lead time: {lead_time} days

Give your recommendation and reasoning."""),
])

chain = prompt | llm    # "pipe": fill the prompt, then send it to the model

response = chain.invoke({
    "product": "toothpaste",
    "current_stock": 20,
    "average_demand": 5,
    "lead_time": 4,
})
print(response.content)
