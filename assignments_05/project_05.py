import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


# --- Task 1 ---
def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """You are a job application coach helping a career changer.
Only help with job application materials, such as resumes, cover letters, and interviews.
If asked about something else, politely redirect.
Do not invent experience, skills, numbers, or credentials.
When you give advice or drafts, remind the user to review and edit before submitting.
Say that you may not know their industry's norms, so they should use their own judgment."""
# I told it not to invent facts because false resume claims can hurt the user.


# --- Task 2 ---
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list, with no markdown or other text.
    Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    reply = get_completion(messages, temperature=0.3)

    try:
        results = json.loads(reply)
        for item in results:
            print("\nOriginal:", item["original"])
            print("Improved:", item["improved"])
        return results
    except json.JSONDecodeError:
        print("Could not read the response as JSON:", reply)
        return []


# --- Task 3 ---
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.
    Use only facts from the person's background. Do not copy facts from the examples.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics bootcamp and want to apply those
    skills to healthcare questions. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of retail banking, seeing how technology
    affects daily work. That experience turned into curiosity, and after two years of
    teaching myself Python, I'm ready to help build those tools. I'm applying to [Company]
    to combine my banking knowledge with my technical skills.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages)


# I used two different career changes to show how to connect old experience to a new role.
# Few-shot helps control tone, length, and structure.


# --- Task 4 ---
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    if flagged:
        print("Sorry, I can't process that message. Please rephrase it.")
    return not flagged


# --- Task 5 ---
def run_chatbot():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.")
    print("Please review and edit everything before submitting. I may not know your")
    print("industry's norms, so use your own judgment.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            if raw_bullets and is_safe("\n".join(raw_bullets)):
                results = rewrite_bullets(raw_bullets)
                if results:
                    messages.append({"role": "user", "content": user_input})
                    messages.append({"role": "assistant", "content": json.dumps(results)})
                    print("\nPlease review and edit these before submitting.")

        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            if job_title and background and is_safe(f"{job_title}\n{background}"):
                letter = generate_cover_letter(job_title, background)
                print("\n" + letter)
                print("\nPlease review and edit this before submitting.")
                messages.append({"role": "user", "content": f"Cover letter for {job_title}. Background: {background}"})
                messages.append({"role": "assistant", "content": letter})

        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print("\nJob Application Helper:", reply, "\n")
            messages.append({"role": "assistant", "content": reply})


def run_tests():
    # Task 2 test
    rewrite_bullets([
        "Helped customers with their problems",
        "Made reports for the management team",
        "Worked with a team to finish the project on time"
    ])
    # These bullets are vague and do not show specific actions or results.
    # Model changes: stronger action verbs and more detail, but it added unsupported claims.
    # For example, "on time" became "ahead of schedule", and it added "cross-functional".
    
    # Task 3 test
    print("\n" + generate_cover_letter(
        "Junior Data Engineer",
        "Five years of experience as a middle school math teacher; recently completed "
        "a Python course and built data pipelines using Pandas."
    ))

    # Task 4 test
    print("\nSafe test:", is_safe("Can you help me improve my resume?"))
    print("Flagged test:", is_safe("I am going to hurt you."))


if __name__ == "__main__":
    run_tests()
    run_chatbot()


# --- Task 6: Ethics Reflection ---
# Option A: Comment block
# The model may favor communication styles or backgrounds that are common in its
# training data. It might push direct, self-promoting language that does not fit
# every culture or industry.
# If someone submits output without reviewing it, it could include false or
# exaggerated claims. That could cost them an offer or damage their credibility.
# As a guardrail, I would require users to confirm every claim is true before
# they copy or export a draft.
# In testing, the model added claims that were not in my input, even though my prompts limited this.