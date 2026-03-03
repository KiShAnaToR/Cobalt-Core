from typing import List, Dict
import openai
import io
import re

# AI interaction helpers

def next_question(role: str, resume_data: dict, history: List[Dict]) -> str:
    """Generate a follow‑up question using the AI model.

    The prompt supplies the desired job role, parsed resume details and
    conversation history so the model can ask something relevant.
    """
    try:
        system_msg = "You are a helpful technical interviewer."
        user_prompt = (
            f"Role: {role}\n"
            f"Resume data: {resume_data}\n"
            f"Conversation history: {history}\n"
            "Based on the above, ask the candidate one appropriate interview question."
        )
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=150,
        )
        question = resp.choices[0].message.content.strip()
    except Exception as e:
        question = f"[error generating question: {e}]"
    return question




def evaluate_answer(answer: str, history: List[Dict]) -> dict:
    """Ask the AI model to critique the candidate's answer.

    The returned dictionary includes a score, detailed feedback, and
    may include suggestions for improvement or vocabulary hints.
    """
    try:
        system_msg = "You are an experienced interviewer who provides constructive feedback."
        user_prompt = (
            f"The candidate answered the following:\n{answer}\n"
            f"Conversation history: {history}\n"
            "Please provide a brief score between 1 and 10, plus written feedback "
            "highlighting strengths, weaknesses, grammar, and vocabulary."
        )
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=250,
        )
        text = resp.choices[0].message.content.strip()
        # naive parse: first line maybe score
        lines = text.splitlines()
        score = None
        feedback_text = text
        if lines:
            m = re.match(r"score[:\s]*(\d+)", lines[0].lower())
            if m:
                score = int(m.group(1))
                feedback_text = "\n".join(lines[1:])
        return {"score": score, "feedback": feedback_text}
    except Exception as e:
        # fallback to simple logic
        score = len(answer.split())
        fb = "Good" if score > 5 else "Try to elaborate more"
        return {"score": score, "feedback": f"AI error: {e}; {fb}"}


def transcribe_and_evaluate(audio_bytes: bytes, history: List[Dict]):
    """Use OpenAI to transcribe audio then evaluate the text. Returns transcript and evaluation."""
    try:
        audio_file = io.BytesIO(audio_bytes)
        # Whisper API expects a filename attribute on file-like objects
        audio_file.name = "audio.wav"
        resp = openai.Audio.transcribe("whisper-1", audio_file)
        transcription = resp.get("text", "")
    except Exception as e:
        transcription = f"[transcription error: {e}]"
    evaluation = evaluate_answer(transcription, history)
    return {"transcript": transcription, "evaluation": evaluation}
