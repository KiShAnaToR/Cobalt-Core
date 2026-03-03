import re
import openai


def analyze(transcript: str) -> dict:
    """Returns feedback such as repeated words, grammar issues, english level.

    Leverages the OpenAI model to provide a richer analysis alongside simple
    statistics.
    """
    words = re.findall(r"\w+", transcript.lower())
    repeats = {w: words.count(w) for w in set(words) if words.count(w) > 2}
    grammar_issues = []
    if " i " in transcript:
        grammar_issues.append("Consider capitalizing 'I'")

    ai_text = ""
    try:
        system_msg = "You are an English tutor who critiques transcripts."
        user_prompt = (
            f"Please analyze the following transcript for grammar mistakes, "
            "usage issues, and give a short assessment of the speaker's "
            "English proficiency level (e.g., beginner, intermediate, advanced).""
            f"Transcript:\n{transcript}"
        )
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
        )
        ai_text = resp.choices[0].message.content.strip()
    except Exception as e:
        ai_text = f"[AI analysis error: {e}]"

    return {
        "repeated_words": repeats,
        "grammar_issues": grammar_issues,
        "ai_analysis": ai_text,
    }
