from pathlib import Path
from jinja2.exceptions import TemplateError
from typing import List, Dict, Union, Tuple, Callable
from file_manager import FileManager
import json
import numpy as np
from transformers import PreTrainedTokenizer
from scat_utils import get_scat_prompt, MAX_TEMPS
from completion_hf import MODELS

# Define our own temperature grid parameters
EPS_GRID = 0.2  # This will create temperatures spaced 0.1 apart

ALL_EXAMPLES = [
    ("Letter: C\nCategory: Countries", "China"),
    ("Letter: V\nCategory: Instruments", "Violin"),
    ("Letter: A\nCategory: Animals", "Alligator"),
    ("Letter: B\nCategory: Things you find in a bathroom", "Brush"),
    ("Letter: P\nCategory: U.S. Cities", "Philadelphia"),
    ("Letter: S\nCategory: Superheroes", "Superman"),
    ("Letter: M\nCategory: Brands of cars", "Mazda"),
    ("Letter: T\nCategory: Things you can eat for breakfast", "Toast"),
    ("Letter: F\nCategory: Famous people", "Franklin"),
    ("Letter: D\nCategory: Hobbies", "Drawing"),
]

# Create a registry for prompt functions
PROMPT_REGISTRY: Dict[str, Callable[[str, str, PreTrainedTokenizer], str]] = {}

def register_prompt(name: str):
    """Decorator to register a prompt function"""
    def decorator(func: Callable[[str, str, PreTrainedTokenizer], str]):
        PROMPT_REGISTRY[name] = func
        return func
    return decorator

# Register the default prompt from scat_utils at module level
@register_prompt("default")
def default_prompt(letter: str, category: str, tokenizer: PreTrainedTokenizer) -> str:
    return get_scat_prompt(letter, category, tokenizer)

@register_prompt("gemini1")
def gemini1_prompt(letter: str, category: str, tokenizer: PreTrainedTokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a Scattegories master. Provide a single, concise word or short phrase that fits the given letter and category. No explanations, no extra words."},
        {"role": "user", "content": 'Let\'s play Scattegories! I\'ll give you a letter and a category. You respond with a valid answer that starts with that letter and fits the category. For example, if I say "Fruit" and "A," you could respond with "Apple" or "Apricot."'},
        {"role": "assistant", "content": "Understood. I'm ready."},
    ]
    for q, a in ALL_EXAMPLES[:2]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": f"Letter: {letter}\nCategory: {category}"})
    return apply_template(messages, tokenizer)

@register_prompt("chatgpt1")
def chatgpt1_prompt(letter: str, category: str, tokenizer: PreTrainedTokenizer) -> str:
    messages = [
            {"role": "system", "content": "You are a concise and clever word generator. Respond with a single word or short phrase that fits the request. No explanations."},
            {"role": "user", "content": 'We are playing a word game. I’ll give you a letter and a category. You respond with something that starts with that letter and fits the category. For example, if I say "Fruit" and "A," your answer could be "Apple" or "Avocado."'},
            {"role": "assistant", "content": "Got it."},
            ]
    for q, a in ALL_EXAMPLES[2:4]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": f"Letter: {letter}\nCategory: {category}"})
    return apply_template(messages, tokenizer)

@register_prompt("claude1")
def claude1_prompt(letter: str, category: str, tokenizer: PreTrainedTokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a Scattegories expert. Provide single-word or very brief answers that start with the specified letter. Be creative, original, and avoid common responses."},
        {"role": "user", "content": "We're playing Scattegories. I'll give you a letter and category, and you'll respond with an interesting word or short phrase starting with that letter. For example, if I say 'Letter: A, Category: Fruit,' you might answer 'Ackee' or 'Asian pear' rather than the more obvious 'Apple.'"},
        {"role": "assistant", "content": "Ready to play!"},
    ]
    for q, a in ALL_EXAMPLES[4:6]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": f"Letter: {letter}\nCategory: {category}"})
    return apply_template(messages, tokenizer)

@register_prompt("grok1")
def grok1_prompt(letter: str, category: str, tokenizer: PreTrainedTokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a creative assistant. Provide a single word or short phrase in response, no explanations."},
        {"role": "user", "content": "We're playing Scattergories! I'll give you a letter and a category. Respond with a word or short phrase starting with that letter, fitting the category. For example, 'Fruit' and 'B' could be 'Banana' or 'Blueberry.'"},
        {"role": "assistant", "content": "Got it! Ready to play."},
    ]
    for q, a in ALL_EXAMPLES[6:8]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": f"Letter: {letter}\nCategory: {category}"})
    return apply_template(messages, tokenizer)

@register_prompt("deepseek1")
def deepseek1_prompt(letter: str, category: str, tokenizer: PreTrainedTokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a competitive Scattergories champion. Respond with only valid answers - single words or very short phrases that perfectly match the category and start with the given letter. No explanations, no apologies."},
        {"role": "user", "content": "Let's play Scattergories! I'll give you a letter and category, and you'll respond with the first valid answer that comes to mind. Quick, creative, and strictly following the rules. For example:"},
        {"role": "assistant", "content": "Ready to play."},
    ]
    for q, a in ALL_EXAMPLES[8:]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": f"⚡ Lightning Round ⚡\nLetter: {letter.upper()}\nCategory: {category}\nGO:"})
    return apply_template(messages, tokenizer)

def apply_template(messages: List[Dict[str, str]], tokenizer: PreTrainedTokenizer) -> str:
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
    except TemplateError:
        messages = messages[1:]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
    return str(text)

# @register_prompt("var1")
# def var1_prompt(letter: str, category: str, tokenizer: PreTrainedTokenizer) -> str:
    # messages = [
            # {"role": "system", "content": "You are a helpful assistant. Answer in as few words as possible, with no explanations."},
            # {"role": "user", "content": "You are going to help me play Scattergories."},
            # {"role": "assistant", "content": "I understand."},
            # ]
    # SCAT_EXAMPLES = [
            # ("Letter: C\nCategory: Countries", "China"),
            # ("Letter: V\nCategory: Instruments", "Violin"),
            # ]
    # for q, a in SCAT_EXAMPLES:
        # messages.append({"role": "user", "content": q})
        # messages.append({"role": "assistant", "content": a})
    # messages.append({"role": "user", "content": f"Letter: {letter}\nCategory: {category}"})
    # try:
        # text = tokenizer.apply_chat_template(
            # messages,
            # tokenize=False,
            # add_generation_prompt=True
            # )
    # except TemplateError:
        # messages = messages[1:]
        # text = tokenizer.apply_chat_template(
            # messages,
            # tokenize=False,
            # add_generation_prompt=True
            # )
    # return str(text)

def get_temps_clean(max_temp: float) -> list[float]:
    """Generate a list of temperatures from 0 to max_temp in EPS_GRID increments"""
    temps = np.arange(0, max_temp + EPS_GRID, EPS_GRID)
    return [round(x, 3) for x in temps]

def generate_model_configs(
    models: List[str],
    temp_ranges: Dict[str, Tuple[float, float]],
    base_dir: Union[str, Path] = "."
) -> None:
    """
    Generate model configuration files for each combination of model, prompt function, and temperature.
    
    Args:
        models: List of model names (e.g., ['llama3.2', 'gpt4'])
        temp_ranges: Dictionary mapping model names to (min_temp, max_temp) tuples
        base_dir: Base directory for the project
    """
    fm = FileManager.from_base(base_dir)
    
    # Generate configs for each model
    for model in models:
        if model not in temp_ranges:
            print(f"Warning: No temperature range specified for model {model}")
            continue
            
        min_temp, max_temp = temp_ranges[model]
        temperatures = [round(t, 2) for t in np.linspace(min_temp, max_temp, 4)[1:]]
            
        # Generate configs for each temperature
        for temp in temperatures:
            if temp == 0.0:
                continue
            # Generate configs for each prompt function
            for prompt_name in PROMPT_REGISTRY.keys():
                config = {
                    "id": f"{model}-temp{temp:.1f}-{prompt_name}",
                    "model": model,
                    "temperature": temp,
                    "prompt_function": prompt_name,
                    "description": f"{model} with temperature {temp:.1f} using {prompt_name} prompt"
                }
                
                # Write the configuration
                fm.write_model_config(config["id"], config)
                print(f"Created config for {config['id']}")

def main():
    # Use models from completion_hf
    models = list(MODELS.keys())
    
    # Create temp ranges using MAX_TEMPS from scat_utils
    temp_ranges = {
        model: (0.0, MAX_TEMPS[MODELS[model]]) 
        for model in models 
        if MODELS[model] in MAX_TEMPS and model not in ('llama3.1', 'qwen2.5')
    }
    
    generate_model_configs(models, temp_ranges)

# Example of how to use the registry when loading configs
def load_and_use_config(config_id: str, letter: str, category: str, tokenizer: PreTrainedTokenizer) -> str:
    """Example function showing how to load a config and use its prompt function"""
    fm = FileManager.from_base(".")
    config = fm.load_model_config(config_id)
    
    # Get the prompt function from the registry
    prompt_fn = PROMPT_REGISTRY[config["prompt_function"]]
    
    # Use the prompt function
    return prompt_fn(letter, category, tokenizer)

if __name__ == "__main__":
    main() 