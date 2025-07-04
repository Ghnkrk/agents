from groq import Groq
from dotenv import load_dotenv
import streamlit as st
import os
from sympy import sympify
import re
import time
import math

load_dotenv()

groq = Groq(
    api_key=os.getenv("GROQ_API")
)

def calculate(expression):
    try:
        
        return eval(expression, {"__builtins__": None}, {
            "sum": sum,
            "range": range,
            "sqrt": math.sqrt,
            "math": __import__("math"),
            "__name__": "__main__"
        })
    except Exception:
        try:
            return float(sympify(expression).evalf())
        except Exception as e:
            return f"Error: {str(e)}"
        
def unit_convert(value, from_unit, to_unit):
    try:
        value = float(value)
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        
        length_factors = {
            "mm": 0.001,
            "cm": 0.01,
            "m": 1.0,
            "km": 1000.0,
            "in": 0.0254,
            "ft": 0.3048,
            "yard": 0.9144
        }

        weight_factors = {
            "mg": 0.001,
            "g": 1.0,
            "kg": 1000.0,
            "tonne": 1_000_000.0,
            "lbs": 453.592,
            "oz": 28.3495
        }

        temperature = {
            ("celsius", "fahrenheit"): lambda x: (x * 9/5) + 32,
            ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9
        }

       
        if (from_unit, to_unit) in temperature:
            return temperature[(from_unit, to_unit)](value)

       
        if from_unit in length_factors and to_unit in length_factors:
            meters = value * length_factors[from_unit]
            return meters / length_factors[to_unit]

       
        if from_unit in weight_factors and to_unit in weight_factors:
            grams = value * weight_factors[from_unit]
            return grams / weight_factors[to_unit]

        return f"Conversion from '{from_unit}' to '{to_unit}' not supported."

    except Exception as e:
        return f"Error: {str(e)}"

def factorial(n):
    n = int(n)
    if n == 0:
        return 1
    if n < 0:
        return "Invalid number. Negative number for factorial is not possible."
    else:
        return n * factorial(n - 1)

tools = {
    "calculate": calculate,
    "unit_convert": unit_convert,
    "factorial": factorial
}

def extract_tool_call(text):
    """
    Extracts the tool name and arguments from model output like:
    TOOL:tool_name("arg1", "arg2", ...)
    Returns: (tool_name, [arg1, arg2, ...]) or None if not a tool call
    """
    pattern = r'TOOL\s*:\s*(\w+)\((.*?)\)'
    match = re.search(pattern, text)

    if not match:
        return None, None

    tool_name = match.group(1).strip()
    raw_args = match.group(2)

    
    args = []
    if raw_args.strip():
        # Handle both quoted and unquoted arguments
        arg_pattern = r'"([^"]*)"|\b([^,\s]+)\b'
        matches = re.findall(arg_pattern, raw_args)
        for match in matches:
            args.append(match[0] if match[0] else match[1])

    return tool_name, args

def word_generator(txt):
    for word in txt:
        yield word
        time.sleep(0.01)

def thought_generator(thoughts):
    for thought in thoughts:
        for char in thought:
            yield char
            time.sleep(0.002)
        yield "\n\n"

def ask_model(messages):
    completion = groq.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192"
    )
    return completion.choices[0].message.content

MAX_STEPS = 10


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "thought_log" not in st.session_state:
    st.session_state["thought_log"] = []

# Display chat history
for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
       
        if msg["role"] == "assistant" and i < len(st.session_state["thought_log"]):
            thoughts = st.session_state["thought_log"][i]
            if thoughts:
                with st.expander("💭 Thought Process"):
                    for thought in thoughts:
                        st.text(thought)


query = st.chat_input("Enter your query:")
if query:
  
    with st.chat_message("user"):
        st.markdown(query)
    
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })
    

    conversation_messages = [
        {
            "role": "system",
            "content": """You are a multi-step math and unit assistant. Solve tasks by reasoning step-by-step and using tools when needed.

### Your Response Options:
- TOOL:tool_name("arg1", "arg2", ...) → to use a tool
- FINAL_ANSWER: your final result → when you're done
- Any reasoning text (you can provide reasoning before using tools)

Only use tools when necessary. Don't estimate or guess — compute precisely.

### Available Tools:
- calculate(expression): Evaluate a math expression (e.g., "sqrt(144) + 5")
- unit_convert(value, from_unit, to_unit): Convert units (e.g., "100", "km", "m")
- factorial(n): Compute factorial of a number (e.g., "5")

### Important Rules:
1. **Always use a tool** for non-trivial expressions (e.g., involving factorials, square roots, logs, or unit math).
2. Do **not** estimate or manually solve complex math — especially chained operations like `sqrt(factorial(10))`. Always break it down and use tools.
3. Only output TOOL:tool_name(args) on its own line when using a tool.
4. After using a tool, you'll receive the result and can continue reasoning.
5. When you have the final answer, output FINAL_ANSWER: [your answer]
6. Be concise but clear in your reasoning.

🔒 Precision is critical. If there's *any* doubt about accuracy, use the tool.
"""
        },
        {
            "role": "user",
            "content": query
        }
    ]
    
    thought_log = []
    final_answer = ""
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            for step in range(MAX_STEPS):
                response = ask_model(conversation_messages)
                
                
                if "FINAL_ANSWER:" in response:
                    final_answer = response.split("FINAL_ANSWER:")[1].strip()
                    thought_log.append(f"Step {step + 1}: {response}")
                    break
                
               
                tool_name, args = extract_tool_call(response)
                if tool_name and tool_name in tools:
                    try:
                        tool_result = tools[tool_name](*args)
                        thought_log.append(f"Step {step + 1}: {response}")
                        thought_log.append(f"Tool Result: {tool_result}")
                        
                      
                        conversation_messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        conversation_messages.append({
                            "role": "user",
                            "content": f"Tool result: {tool_result}. Please continue or provide the final answer."
                        })
                    except Exception as e:
                        thought_log.append(f"Step {step + 1}: Tool Error: {str(e)}")
                        conversation_messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        conversation_messages.append({
                            "role": "user",
                            "content": f"Tool error: {str(e)}. Please try a different approach."
                        })
                else:
                   
                    thought_log.append(f"Step {step + 1}: {response}")
                    conversation_messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    conversation_messages.append({
                        "role": "user",
                        "content": "Please continue with your reasoning or provide the final answer."
                    })
            
          
            if not final_answer:
                final_answer = "Could not determine the final answer within the step limit."
        
        
        if thought_log:
            with st.expander("💭 Thought Process", expanded=True):
                st.spinner("Thinking...")
                st.write_stream(thought_generator(thought_log))
        
       
        st.markdown(f"**Answer:** {final_answer}")
    
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"**Answer:** {final_answer}"
    })
    st.session_state.thought_log.append(thought_log)
