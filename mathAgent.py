from groq import Groq
from sympy import sympify
from dotenv import load_dotenv
import os
import streamlit as st
import time
import math

load_dotenv()

groq = Groq(
    api_key= os.getenv("GROQ_API")
)

def calculate(expression):
    try:
        # Safer eval with access to only essential built-ins
        return eval(expression, {"__builtins__": None}, {
            "sum": sum,
            "range": range,
            "math": __import__("math"),
            "__name__": "__main__"
        })
    except Exception:
        try:
            return float(sympify(expression).evalf())
        except Exception as e:
            return f"Error: {str(e)}"



def ask_model(history):

        messages = [
            {
                "role": "system",
                "content": """You are a math expression generator. Given a question involving arithmetic or symbolic math, output a pure Python expression that returns the exact numeric result when evaluated using Python's `eval()` or SymPy's `sympify()`.

Your expression **must be accurate**, avoid estimating or rounding, and avoid explanations or surrounding text.

Use only:
- Basic arithmetic: `+`, `-`, `*`, `/`, `**`
- Built-in functions: `sum()`, `range()`
- For square roots or cube roots, use exponentiation (e.g. `x ** 0.5`, `x ** (1/3)`)

✅ Examples:

Q: cube root of 999  
➡️ `999 ** (1/3)`

Q: sum of first 10 natural numbers  
➡️ `sum(range(1, 11))`

Q: square root of sum of first 5 prime numbers  
➡️ `sum([2, 3, 5, 7, 11]) ** 0.5`

**Only return the raw expression** — nothing else. Do not use list comprehensions, lambdas, or advanced Python syntax.

"""
            }
        ]
        messages.extend(history)

        completion =  groq.chat.completions.create(
            messages= messages,
            model = 'llama3-8b-8192'
        )

        return completion.choices[0].message.content

def response_generator(response):
        for c in response:
                yield c
                time.sleep(0.001)

if st.session_state.get("messages") is None:
        st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Enter your query:")
if query:
        with st.chat_message("user"):
                st.markdown(query)
        st.session_state.messages.append({
                "role" : "user",
                "content" : f'{query}'
        }) 
        MAX_TURNS = 10
        if len(st.session_state.messages) > MAX_TURNS:
            st.session_state.messages = st.session_state.messages[-MAX_TURNS:]

        expression = ask_model(st.session_state.messages)

        answer = calculate(expression)

        st.session_state.messages.append(
              {
              "role" : "assistant",
              "content" : f"The final answer is {answer}"
              }
        )

        with st.chat_message("system"):
                st.write_stream(response_generator(f'The final answer is {answer}'))



    