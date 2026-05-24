"""
llm.py
------
Handles LLM calls via Groq (free LLaMA3-70b).
Falls back to OpenAI if OPENAI_API_KEY is set instead.

Get your FREE Groq API key at: https://console.groq.com
"""

import os
from groq import Groq

# ── Config ─────────────────────────────────────────────────────────────────
# GROQ_MODEL   = "llama3-70b-8192"      # free, fast, high quality
GROQ_MODEL   = "llama-3.3-70b-versatile"      # free, fast, high quality
MAX_TOKENS   = 1024
TEMPERATURE  = 0.2                    # low = factual, consistent answers

# SYSTEM_PROMPT = """You are a helpful medical information assistant powered by \
# verified healthcare data from PubMed and NIH sources.

# Your role:
# - Answer medical questions clearly and accurately based ONLY on the provided context
# - Cite which source (PubMedQA, MedQuAD) the information comes from
# - If the context does not contain enough information to answer, say so honestly
# - Always end your response with: "⚠️ This is for educational purposes only. \
# Please consult a qualified healthcare professional for personal medical advice."
# Do NOT make up information. Do NOT go beyond what the context provides."""

# SYSTEM_PROMPT = """You are a helpful, friendly medical information assistant.

# Your job:
# - Answer the user's health question in clear, simple, easy-to-understand language
# - Use the provided context to support your answer, but also use your general medical knowledge to fill gaps
# - Give practical, direct answers — like a knowledgeable friend explaining health topics
# - Structure your answer with bullet points or short paragraphs when helpful
# - If the context is from research papers and too technical, translate it into plain English
# - Never say "the context does not contain enough information" — instead give the best answer you can using both the context AND your knowledge
# - Always end with the disclaimer below

# End every response with:
# ⚠️ This is for educational purposes only. Please consult a qualified healthcare professional for personal medical advice."""


SYSTEM_PROMPT = """You are a knowledgeable, friendly health assistant — like a doctor friend who explains things clearly.

When answering health questions:
- Give comprehensive, well-structured answers using your medical knowledge
- Use the provided context as supporting evidence, but DO NOT limit yourself to it
- Format answers with emojis, headers, and bullet points to make them easy to read
- Include: what it is, symptoms, causes, treatments, home remedies, medications, when to see a doctor
- Use simple everyday language — avoid overly technical jargon
- Be direct and practical — give real actionable advice
- Never say "the context doesn't have enough info" — you are a knowledgeable assistant, use your knowledge
- For general health questions, give full helpful answers like a medical encyclopedia would

Structure your answers like this when relevant:
- Brief intro (1-2 lines)
- 🔍 Key facts / symptoms
- 💊 Treatments & medications  
- 🏠 Home remedies & lifestyle tips
- ⚠️ When to see a doctor

Always end with:
⚠️ This is for educational purposes only. Please consult a qualified healthcare professional for personal medical advice."""


# ── LLM caller ────────────────────────────────────────────────────────────────
class MedicalLLM:
    """
    Wraps Groq API. Call .generate(query, context) to get an answer.
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found.\n"
                "1. Get a free key at https://console.groq.com\n"
                "2. Set it: export GROQ_API_KEY=your_key_here"
            )
        self.client = Groq(api_key=api_key)
        print(f"LLM ready: {GROQ_MODEL} via Groq")

    def generate(self, query: str, context: str) -> str:
        """
        Generate an answer given the user query and retrieved context.

        Args:
            query:   the user's question
            context: formatted string of retrieved chunks (from retriever.py)

        Returns:
            LLM-generated answer as a string
        """
        user_message = f"""Here is relevant medical information to answer the question:

{context}

---

Question: {query}

Please answer based on the context above."""

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        return response.choices[0].message.content

    def generate_with_history(
        self,
        query: str,
        context: str,
        chat_history: list[dict],
    ) -> str:
        """
        Multi-turn version — includes previous messages for conversation continuity.

        Args:
            chat_history: list of {"role": "user"/"assistant", "content": str}
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add last 6 turns of history (to stay within token limits)
        messages.extend(chat_history[-6:])

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        })

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        return response.choices[0].message.content


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    llm = MedicalLLM()

    test_context = """[Source 1 | MEDQUAD | relevance: 0.91]
Question: What is diabetes?
Answer: Diabetes is a chronic disease that occurs when the pancreas does not produce
enough insulin, or when the body cannot effectively use the insulin it produces.
Insulin is a hormone that regulates blood sugar."""

    answer = llm.generate("What is diabetes?", test_context)
    print(answer)
