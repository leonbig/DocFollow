from openai import OpenAI
from typing import Optional

def get_text_theme(text: str, client: OpenAI, eval_model) -> Optional[str]:
    # Input validation
    # if not text or not text.strip():
    #     return None

    # Asks the model to read the text and summarize the core theme briefly.
    prompt = f"""
    Your sole task is to identify and output the single, most representative main topic from the text provided below.
    Adhere strictly to these non-negotiable output rules:
    1.  Your response **MUST** consist *only* of the extracted topic itself.
        * **DO NOT** include any introductory phrases (e.g., "The topic is:", "Here is the topic:", "Okay, the main topic is:").
        * **DO NOT** include any explanations, apologies, summaries, or conversational filler.
        * **DO NOT** include any labels or quotation marks around the topic unless they are part of the topic itself as found in the text.
        * The output must be a direct string of words representing the topic.
    2.  The topic **MUST** be between 5 and 15 words in length, inclusive. This word count range is strict.
    3.  The topic **MUST** be maximally comprehensive, accurately encapsulating the core subject and primary themes of the *entire* text, while strictly adhering to the specified word count range. Prioritize full text coverage within the brevity constraint.

    Text to analyze:
    {text}"""
    try:
        response = client.chat.completions.create(
            model=eval_model, 
            messages=[
                {"role": "system", "content": "You are a theme analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9
        )
        theme = response.choices[0].message.content.strip()
        if theme:
            return theme
        else:
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None