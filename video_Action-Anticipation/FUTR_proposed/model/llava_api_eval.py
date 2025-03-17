import os
import sys
import torch
llava_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../awesome-llm/llava/"))
sys.path.append(llava_model_path)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


def get_fine_grained_labels(video_frames, human_prompt):
    """
    Generate fine-grained labels using llava
    
    Args:
        video_features (list): Encoded features of video sequences (for context).
        coarse_labels (list): Coarse-level labels for the video sequence.
        fine_label_options (list): List of fine-grained label options.
    
    Returns:
        str: llava-generated fine-grained labels.
    """
    model_path = "liuhaotian/llava-v1.5-7b"

    # Define arguments for eval_model
    args = type('Args', (), {
        "model_path": model_path,               # Path to the LLaVA model
        "model_base": None,                     # Base model, if any
        "model_name": get_model_name_from_path(model_path),  # Get model name dynamically
        "query": human_prompt,                        # User prompt
        "conv_mode": None,                      # Conversation mode
        "image_file": video_frames,               # Input image file (URL or path)
        "sep": ",",                             # Separator for outputs
        "temperature": 0,                       # Sampling temperature
        "top_p": None,                          # Top-p sampling (optional)
        "num_beams": 1,                         # Number of beams for beam search
        "max_new_tokens": 512                   # Maximum number of new tokens
    })()

    return eval_model(args)
