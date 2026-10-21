import os
import argparse
from typing import Dict, List, Optional
from openai import OpenAI

# Prompt templates for each level
PROMPT_TEMPLATES: Dict[int, str] = {
    1: """
        Here is the original theme Q0:
        "{theme}"

        Task: 
            Identify the one or two most central content words in Q0 (e.g. key nouns or adjectives). 
            Replace each with either an antonym (to invert) or a near-synonym (to refine) so that the new theme Q1 
                (a) preserves Q0's structure and rhythm, 
                (b) subtly shifts its meaning, and 
                (c) would still almost—but not quite—fit the original text.
                (d) has the exact same number of words as Q0.
            Output only the transformed theme Q1. Do not include any explanations, reasoning, or additional text.""",
    
    2: """
        Here is the original theme Q0:
        "{theme}"

        Task: 
            1. Decompose Q0 into two slots—Phenomenon (P) and Context (C). 
            2. Choose one of these slots (P or C) and swap it for a parallel concept within the same domain (e.g. replace 'temperature drop' with 'salinity increase' in a marine context). 
            3. Draft a new theme Q1 that 
                (a) keeps the other slot unchanged, 
                (b) mirrors Q0's syntax exactly, and 
                (c) renders the original text's examples inapplicable without extensive rework.
                (d) has the exact same number of words as Q0.
            Output only the transformed theme Q1. Do not provide any explanations or additional text.""",
    
    3: """
        Here is the original theme Q0:
        "{theme}"

        Task: 
            1. Abstract Q0's core argumentative frame (e.g. 'change in X → response in Y'). 
            2. Select a distant but analogous domain and map X→X* and Y→Y* accordingly. 
            3. Reproduce Q0's syntactic pattern and rhetorical rhythm to form a new theme Q1 about X*'s effect on Y* in that new context. Ensure Q1 
                (a) fully breaks topical overlap with the original, yet 
                (b) preserves its logical skeleton for imitation.
                (c) has the exact same number of words as Q0.
            Output only the transformed theme Q1. Do not include any explanations, steps, or additional text."""
}

def validate_inputs(client: OpenAI, eval_model: str, orig_theme: str, orig_text: str, theme_level: int) -> None:
    """
    Validate all input parameters.
    
    Args:
        client: OpenAI client instance
        eval_model: Model name to use
        orig_theme: Original theme text
        orig_text: Original text
        theme_level: Theme transformation level
        
    Raises:
        ValueError: If any input is invalid
        TypeError: If types are incorrect
    """
    if not isinstance(client, OpenAI):
        raise TypeError("client must be an instance of OpenAI")
    
    if not isinstance(eval_model, str) or not eval_model.strip():
        raise ValueError("eval_model must be a non-empty string")
    
    if not isinstance(orig_theme, str) or not orig_theme.strip():
        raise ValueError("orig_theme must be a non-empty string")
    
    if not isinstance(orig_text, str):
        raise ValueError("orig_text must be a string")
    
    if not isinstance(theme_level, int) or theme_level not in PROMPT_TEMPLATES:
        raise ValueError(f"theme_level must be one of {list(PROMPT_TEMPLATES.keys())}")

def build_messages(orig_theme: str, orig_text: str, theme_level: int) -> List[Dict[str, str]]:
    """
    Build the messages for the API call.
    
    Args:
        orig_theme: Original theme text
        orig_text: Original text
        theme_level: Theme transformation level
        
    Returns:
        List of message dictionaries
    """
    messages = [
        {"role": "system", "content": "You are a theme transformation assistant."}
    ]
    
    # Include original text for all levels
    if orig_text.strip():
        messages.append({
            "role": "user",
            "content": f"Here is the text T:\n{orig_text}"
        })
    
    # Add the theme transformation prompt
    prompt = PROMPT_TEMPLATES[theme_level].format(theme=orig_theme)
    messages.append({"role": "user", "content": prompt})
    
    return messages

def generate_target_theme(client: OpenAI, eval_model: str, orig_theme: str, orig_text: str, theme_level: int) -> str:
    """
    Generate a target theme based on the original theme and text.
    
    Args:
        client: OpenAI client instance
        eval_model: Model name to use
        orig_theme: Original theme text
        orig_text: Original text
        theme_level: Theme transformation level (1-3)
        
    Returns:
        str: Generated target theme
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If API call fails
    """
    # Validate inputs
    validate_inputs(client, eval_model, orig_theme, orig_text, theme_level)
    
    try:
        # Build messages
        messages = build_messages(orig_theme, orig_text, theme_level)
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=eval_model,
            messages=messages,
            temperature=0
        )
        
        # Extract and validate response
        result = response.choices[0].message.content.strip()
        if not result:
            raise RuntimeError("Empty response from API")
            
        return result
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate target theme: {str(e)}")

