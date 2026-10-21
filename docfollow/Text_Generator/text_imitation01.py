def text_imitation01(client, para_model, original_text, new_theme):
    prompt_template = f"""
    # Role Definition
    You are an expert in text imitation.

    # Task Overview
    Analyze the original text and create a new text on a different topic while follow the following constrains.

    # Input
    Original Text:
    {original_text}

    Target topic:
    {new_theme}

    # Imitation Constrains

    1. Length Constrain:
       - The imitated text MUST contain exactly the same number of sentences as the original
       - Match the word count of each individual sentence as closely as possible to its corresponding sentence in the original
       - The total word count of the imitated text MUST match the original text exactly

    2. Topic Constrain:
       - The imitated content must focus entirely on the target topic
       - Ensure logical coherence and natural expression
       - Appropriately adjust technical terms or topic-specific vocabulary to fit the target topic

    # Output Requirements
        Output only the imitated text without explaining your analysis process or creative approach. Ensure the output text is complete, clearly formatted, and ready for immediate use.
"""

    messages = [{"role": "user", "content": prompt_template}]
    response = client.chat.completions.create(
        model=para_model, 
        messages=messages,
        temperature=0.9,
    )
    content = response.choices[0].message.content.strip()
    return content
