import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def ask(prompt, temperature=0.7):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content


def chat(messages):
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content


# --- Completions API ---

# API Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)
print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Total tokens:", response.usage.total_tokens)

# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    print(f"\nTemperature {temp}:", ask(prompt, temp))

# Higher temperature gives more varied and unusual answers.
# I would use 0 for consistent output, but it does not guarantee identical results.

# API Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
for i, choice in enumerate(response.choices, 1):
    print(f"\nFact {i}:", choice.message.content)

# API Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)
print("\nShort response:", response.choices[0].message.content)
# The answer gets cut off at 15 tokens. max_tokens helps limit cost and length.

# --- System Messages ---

# System Q1
question = "I don't understand what a list comprehension is."
personas = [
    "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement.",
    "You are a strict, no-nonsense senior engineer. You answer briefly and technically."
]
for persona in personas:
    messages = [
        {"role": "system", "content": persona},
        {"role": "user", "content": question}
    ]
    print("\nPersona:", persona)
    print(chat(messages))
# The system message changes the tone, length, and level of detail.

# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
print("\nMemory test:", chat(messages))
# It knows the name because we send the whole history in this request.

# --- Prompt Engineering ---

# Prompt Q1
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]
task = "Classify the sentiment of the review as positive, negative, or mixed."

print("\nZero-shot")
for i, review in enumerate(reviews, 1):
    print(f"Review {i}:", ask(f'{task}\nReview: "{review}"', 0))

# Prompt Q2
one_example = '''Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed'''

print("\nOne-shot")
for i, review in enumerate(reviews, 1):
    print(f"Review {i}:", ask(f'{task}\n{one_example}\n\nReview: "{review}"\nSentiment:', 0))
# # The one-shot example made the answers shorter, with just the label.

three_examples = '''Examples:
Review: "I love how easy this is to use."
Sentiment: positive
Review: "It stopped working after one day."
Sentiment: negative
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed'''

print("\nFew-shot")
for i, review in enumerate(reviews, 1):
    print(f"Review {i}:", ask(f'{task}\n{three_examples}\n\nReview: "{review}"\nSentiment:', 0))
# Zero-shot: simple tasks. One-shot: to show a format.
# Few-shot: when I need consistency or have several categories.

# Prompt Q4
prompt = """A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
Show your reasoning step by step, then label the result "Final Answer:"."""
print("\nChain of thought:", ask(prompt, 0))
# Solving one step at a time reduces skipped steps and math mistakes.

# Prompt Q5
review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = f"""Analyze the review in triple backticks.
Return only valid JSON with these keys: sentiment, confidence (float from 0 to 1),
reason (one sentence). Do not use markdown.
```{review}```"""

raw = ask(prompt, 0)
print("\nRaw response:", raw)

try:
    data = json.loads(raw)
    print("Sentiment:", data["sentiment"])
    print("Confidence:", data["confidence"])
    print("Reason:", data["reason"])
except json.JSONDecodeError:
    print("Not valid JSON:", raw)

# Prompt Q6
user_texts = [
    "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice.",
    "The weather was calm today, and the park was full of people."
]

for user_text in user_texts:
    prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
    print("\nDelimiter test:", ask(prompt, 0))
# Delimiters separate user text from my instructions.
# They help reduce prompt injection, but do not fully prevent it.

# --- Ollama ---

# Ollama Q1
print("\nOpenAI:", ask("Explain what a large language model is in two sentences."))

"""
Ollama output:
A large language model is a type of AI that can understand and generate human-like text, trained on vast amounts of data to process and understand language effectively. It's used for tasks like writing, answering questions, or even complex tasks, demonstrating strong ability to handle multiple languages and tasks.
"""
# Differences: OpenAI gave more detail, including neural networks and examples like translation.
# Ollama used simpler wording and repeated words like "tasks" and "understand".
# Advantage of local: data can stay on my computer, with no per-request API fee.
# Disadvantage: it uses my hardware, and small models are usually less accurate.