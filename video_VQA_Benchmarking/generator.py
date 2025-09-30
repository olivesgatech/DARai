import csv
import os
import json
import torch
from dataset import VideoFrameDataset ,Video_Dataset
from models import (load_Llava_model_and_processor , load_LLaMA3_model_and_processor , load_instruct_blip_model_and_processor ,
                    instruct_blip_generate_answer , LLaMA3_generate_answer , LLaVa_NeXT_generate_answer, Qwen2VL_generate_answer , load_Qwen2VL_model_and_processor)
from utils import set_seed
from torchvision.transforms import ToPILImage

def LLaVa_NeXT_Video_generator(config_filename):
    """
    Generates answers for multiple questions from a dataset using LLaVA-NeXT-Video.
    Saves results incrementally to avoid data loss.
    """

    # Load configuration
    with open(config_filename, "r") as f:
        cfg = json.load(f)

    dataset_root = cfg.get("dataset_root", "/mnt/Data1/RGB_sd")
    sequence_length = cfg.get("sequence_length", 16)
    csv_filename = cfg.get("csv_filename", "output_answers.csv")
    questions = cfg.get("questions", ["What is the person doing in this video?"])  # List of questions
    max_new_tokens = cfg.get("max_new_tokens", 100)

    # Load the dataset
    dataset = VideoFrameDataset(root_dir=dataset_root, sequence_length=sequence_length)
    print(f"Loaded dataset from {dataset_root} with {len(dataset)} samples.")

    # Load the model and processor
    model, processor = load_Llava_model_and_processor()
    print("Loaded model and processor.")

    # Prepare the CSV file: Write header if file does not exist
    csv_exists = os.path.exists(csv_filename)
    if not csv_exists:
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f,
                                    fieldnames=["activity", "camera", "subject_id", "session_id", "question", "answer"])
            writer.writeheader()

    # Loop through the dataset
    for idx in range(len(dataset)):
        try:
            instance = dataset[idx]
            _, activity, camera, (subject_id, session_id) , _ = instance

            for question in questions:  # Loop through all questions
                try:
                    answer = LLaVa_NeXT_generate_answer(instance, question, processor, model, max_new_tokens=max_new_tokens)
                    cleaned_answer = answer.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in answer else answer.strip()

                    output_entry = {
                        "activity": activity,
                        "camera": camera,
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "question": question,
                        "answer": cleaned_answer
                    }

                    # Append the answer to the CSV file
                    with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["activity", "camera", "subject_id", "session_id",
                                                               "question", "answer"])
                        writer.writerow(output_entry)

                    print(f"Processed sample {idx}, Question: {question}")

                except Exception as qe:
                    print(f"Error processing question '{question}' for sample {idx}: {qe}")
                    continue  # Continue to the next question

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue  # Skip to the next sample

    print("Saved answers to", csv_filename)
    return csv_filename


def LLaMA3_Video_generator(config_filename):
    """
    Generates answers for multiple questions from the dataset using the VideoLLaMA3-2B model.
    Saves results incrementally to avoid data loss.

    Args:
        config_filename (str): Path to the JSON configuration file.
    """
    # Load configuration file
    with open(config_filename, "r") as f:
        cfg = json.load(f)

    csv_filename = cfg.get("csv_filename", "output_answers.csv")

    # Create dataset instance
    dataset = Video_Dataset(
        root_dir=cfg["dataset_root"],
        sequence_length=cfg.get("sequence_length", 32),
        output_video_dir=cfg.get("output_video_dir", "./video_outputs"),
        fps=cfg.get("fps", 1)
    )

    # Load model and processor
    model_name = cfg["model_name"]
    model, processor = load_LLaMA3_model_and_processor(model_name)

    csv_exists = os.path.exists(csv_filename)
    if not csv_exists:
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f,
                                    fieldnames=["activity", "camera", "subject_id", "session_id", "question", "answer"])
            writer.writeheader()

    for idx in range(len(dataset)):
        try:
            video_path, activity, camera, ids = dataset[idx]
            subject_id, session_id = ids
            questions = cfg["question"]
            max_tokens = cfg.get("max_new_tokens", 300)
            for question in questions:
                answer = LLaMA3_generate_answer(video_path, question, model, processor, max_tokens)
                output_entry = {
                    # "video_path": video_path,
                    "activity": activity,
                    "camera": camera,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "question": question,
                    "answer": answer
                }
                # print("___________Checkpoint____________")
                # print(output_entry)
                # Append result to CSV
                with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["activity", "camera", "subject_id", "session_id","question", "answer"])
                    writer.writerow(output_entry)

                print(f"Processed sample {idx+1}/{len(dataset)}: {question} \t {answer}")
        except Exception as e:
            print(f"Error processing sample {idx+1}: {e}")

    print("Saved answers to", csv_filename)
    return csv_filename

#Example Usage
#LLaMA3_Video_generator("LLaMA3_Video.json")



def Qwen2VL_generator(config_filename: str) -> str:
    """
    Runs through an entire dataset, asks each question via Qwen2VL_generate_answer,
    and appends all results to a CSV (incrementally, to avoid data loss).

    Returns:
        The path to the final CSV file.
    """
    # --- Load config ---
    with open(config_filename, "r") as f:
        cfg = json.load(f)

    dataset_root = cfg.get("dataset_root", "/mnt/data-tmp/ghazal/DARai_DATA/RGB_sd")
    sequence_length = cfg.get("sequence_length", 16)
    csv_filename = cfg.get("csv_filename", "output_answers_qwen.csv")
    questions = cfg.get("questions", ["What is this person doing?"])
    max_new_tokens = cfg.get("max_new_tokens", 200)
    model_name = cfg.get("model_name", "Qwen/Qwen2-VL-7B-Instruct")


    dataset = VideoFrameDataset(root_dir=dataset_root, sequence_length=sequence_length)
    # print(f"Loaded dataset from {dataset_root} ({len(dataset)} samples).")

    model, processor = load_Qwen2VL_model_and_processor(model_name, quantize=True, quant_bits=8)
    # print("Model and processor loaded.")


    if not os.path.exists(csv_filename):
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "activity", "camera", "subject_id", "session_id", "question", "answer"
            ])
            writer.writeheader()

    # --- Loop & generate ---
    for idx, instance in enumerate(dataset):
        _, activity, camera, (subject_id, session_id) , _ = instance
        for question in questions:
            try:
                answer = Qwen2VL_generate_answer(
                    instance, question, processor, model, max_new_tokens
                )
                entry = {
                    "activity": activity,
                    "camera": camera,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "question": question,
                    "answer": answer
                }
                with open(csv_filename, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=entry.keys())
                    writer.writerow(entry)

                print(f"[Sample {idx}] Q: {question} → {answer}")
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

    print("Saved answers to", csv_filename)
    return csv_filename