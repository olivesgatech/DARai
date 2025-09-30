import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import inspect , traceback , torch
from PIL import Image
from torchvision.transforms import ToPILImage
from transformers import *
from huggingface_hub import snapshot_download
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor , Qwen2VLForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers.modeling_utils import get_parameter_device



def load_Llava_model_and_processor(model_name="llava-hf/LLaVA-NeXT-Video-7B-hf"):
    """
    Loads the LlavaNextVideo model and its processor.
    Returns:
        model, processor: The loaded model and processor.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = LlavaNextVideoProcessor.from_pretrained(model_name)
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        "llava-hf/LLaVA-NeXT-Video-7B-hf",
        quantization_config=quantization_config,
        # device_map='auto'
        device_map={"": "cuda:0"}
    )
    return model, processor


def load_LLaMA3_model_and_processor(model_name="DAMO-NLP-SG/VideoLLaMA3-2B"):
    """
    Loads the VideoLLaMA3-2B model and its processor.
    Returns:
        model, processor: The loaded model and processor
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # device_map="auto",
        # device_map="balanced",
        device_map= "balanced_low_0",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    log_path = "llama3_parameter_devices.txt"
    with open(log_path, "w") as f:
        f.write("--- Model Parameters and Their Devices ---\n")
        for name, param in model.named_parameters():
            f.write(f"{name:60} --> {param.device}\n")

    return model, processor
# ____________________ Custome device map _________________________#
# def load_LLaMA3_model_and_processor(model_name="DAMO-NLP-SG/VideoLLaMA3-2B"):
#     """
#     Loads the VideoLLaMA3-2B model and its processor with multi-GPU support using Accelerate.
#     Returns:
#         model, processor: The loaded model and processor
#     """
#     processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
#     local_model_path = snapshot_download(repo_id=model_name)
#
#     with init_empty_weights():
#         model = AutoModelForCausalLM.from_pretrained(
#             local_model_path,
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True,
#             low_cpu_mem_usage=True,
#         )
#
#     # Manually define device map
#     device_map = {
#         "model.embed_tokens": 0,
#         "model.norm": 1,
#         "lm_head": 0,
#         "model.vision_encoder.embeddings": 0,
#         "model.vision_encoder.encoder.final_layer_norm": 1,
#         "model.vision_encoder.post_layernorm": 1,
#         "model.mm_projector.readout.0": 1,
#         "model.mm_projector.readout.2": 1,
#     }
#
#     # Text layers
#     for i in range(0, 14):
#         device_map[f"model.layers.{i}"] = 0
#     for i in range(14, 28):
#         device_map[f"model.layers.{i}"] = 1
#
#     # Vision encoder layers
#     for i in range(0, 14):
#         device_map[f"model.vision_encoder.encoder.layers.{i}"] = 0
#     for i in range(14, 27):
#         device_map[f"model.vision_encoder.encoder.layers.{i}"] = 1
#
#     # Tie weights BEFORE dispatching
#     model.tie_weights()
#
#     # Dispatch model across GPUs
#     model = load_checkpoint_and_dispatch(
#         model,
#         checkpoint=local_model_path,
#         device_map=device_map,
#         dtype=torch.bfloat16,
#         offload_folder="offload"
#     )
#
#     log_file = "llama3_parameter_devices.txt"
#     with open(log_file, "w") as f:
#         f.write("Module Name -> Device Mapping:\n\n")
#         for name, param in model.named_parameters():
#             f.write(f"{name}: {param.device}\n")
#
#     return model, processor

def load_instruct_blip_model_and_processor(model_name="Salesforce/instructblip-vicuna-7b"):
    '''
        Loads the InstructBlipVideo model and its processor.
        Returns:
            model, processor: The loaded model and processor
    '''

    model = InstructBlipVideoForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b",
                                                                      load_in_8bit=True, device_map = "cuda:0")
                                                                      # load_in_8bit=True, device_map='auto')
    # print(f"______________Model_______________ \n {model}")
    # for name, param in model.named_parameters():
    #     print(f"{name} is on {param.device}")
    # print(f"_____________________Model Config____________________ \n {model.config}")
    # print(f"Number of query tokens: {model.config.num_query_tokens}")

    # Update the number of query tokens
    model.config.num_query_tokens = 128   # 32 * 4

    processor = InstructBlipVideoProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    processor.num_query_tokens = model.config.num_query_tokens
    return model, processor

def load_Qwen2VL_model_and_processor(model_name= "Qwen/Qwen2-VL-7B-Instruct", quantize = False, quant_bits = 8):
    """
    Loads the Qwen2VL model and its processor.
    Returns:
        model (Qwen2VLForConditionalGeneration), processor (AutoProcessor)
    """
    quant_config = None
    if quantize:
        if quant_bits == 4:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        elif quant_bits == 8:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        else:
            raise ValueError("quant_bits must be 4 or 8")

    processor = AutoProcessor.from_pretrained(model_name , trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        **({"quantization_config": quant_config} if quantize else {})
    )
        # attn_implementation="flash_attention_2",

    return model, processor

def LLaVa_NeXT_generate_answer(instance, question, processor, model, max_new_tokens=100):
    """
    Generates an answer from the LlavaNextVideo model based on a given question and a video sample.
    Returns:
        str: The cleaned answer (only the text after 'ASSISTANT:').
    """
    # Unpack the sample
    frames, activity, camera, (subject_id, session_id) = instance

    # Convert each NumPy frame to a PIL Image
    to_pil = ToPILImage()
    frame_images = [to_pil(frame) for frame in frames]

    prompt = (
        f"USER: <video>\n"
        f"Question: {question}\n"
        f"ASSISTANT:"
    )

    # Note: both videos and text must be wrapped in lists.
    inputs = processor(
        videos=[frame_images],
        text=[prompt],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate the answer from the model
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode the generated tokens into a string
    full_answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Keep only the content after 'ASSISTANT:'
    marker = "ASSISTANT:"
    idx = full_answer.find(marker)
    cleaned_answer = full_answer if idx == -1 else full_answer[idx + len(marker):].strip()

    return cleaned_answer


def LLaMA3_generate_answer(instance, question, model, processor, max_tokens):
    """
    Generates an answer for a given video sample.
    """

    # Video conversation
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": instance, "fps": 1, "max_frames": 7}},
                {"type": "text", "text": question},
            ]
        },
    ]

    inputs = processor(conversation=conversation, return_tensors="pt")

    # inputs = {
    #     k: v.to(get_parameter_device(model)) if isinstance(v, torch.Tensor) else v
    #     for k, v in inputs.items()}

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

    text_device = model.model.embed_tokens.weight.device
    vision_device = model.model.vision_encoder.embeddings.patch_embedding.weight.device

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if k == "pixel_values":
                inputs[k] = v.to(vision_device, dtype=torch.bfloat16)
            else:
                inputs[k] = v.to(text_device)

    output_ids = model.generate(**inputs, max_new_tokens=max_tokens )

    #______________ using AMP/// not working - Do not try _______________#
    # with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    #     output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    #____________________________________________________________________#
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(response)
    return response


def instruct_blip_generate_answer(instance, question, processor, model, max_new_tokens):
    """
    Generates an answer from the InstructBlipVideo model based on a given question and a video sample of 4 image frames.
    Returns: the cleaned answer
    """
    # Unpack the sample
    frames, activity, camera, (subject_id, session_id) = instance

    # Convert each NumPy frame to a PIL Image
    # to_pil = ToPILImage()
    # frame_images = [to_pil(frame) for frame in frames]

    # #reduce frame_images to 4 random frames
    # frame_images = random.sample(frame_images, 4)

    prompt = "In this video, " + question
    # prompt = question
    inputs = processor(
        text=prompt,
        images=frames,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=max_new_tokens,
        repetition_penalty=1.5,
        length_penalty=1.0,
    )
    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    cleaned_answer = answer.replace(prompt, "")
    print(f"prompt: {prompt} \n answer: {cleaned_answer}")
    return cleaned_answer


def Qwen2VL_generate_answer(instance, question: str, processor, model, max_new_tokens: int = 100) -> str:
    """
    Generates an answer from Qwen2VL for a single video‐question pair.

    Args:
        instance: a tuple (frames, activity, camera, (subject_id, session_id))
                  where `frames` is a list/array of video frames (e.g. NumPy arrays).
        question:  the natural‐language question to ask.
        processor: the AutoProcessor for Qwen2VL.
        model:     the Qwen2VLForConditionalGeneration model.
        max_new_tokens: maximum tokens to generate.

    Returns:
        The model’s answer (text only), with any leading prompt trimmed off.
    """
    # frame_images = [ Image.fromarray(instance[0][i]) for i in range(instance[0].shape[0])]
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video", "video": instance[-1],
                    "fps": 1.0,
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    # Tokenize with the chat template
    try:
        inputs = processor.apply_chat_template(
            conversation,
            video_fps=1.0,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,  # <— explicitly enable padding
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
    except Exception as e:
        print("ERROR in apply_chat_template:", e, flush=True)
        traceback.print_exc()
        raise
    try:
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    except Exception as e:
        print("ERROR in model.generate:", e, flush=True)
        traceback.print_exc()
        raise

    generated_ids = [out_ids[len(inp_ids):] for inp_ids, out_ids in zip(inputs.input_ids, output_ids)]
    full_answer = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    return full_answer.strip()
