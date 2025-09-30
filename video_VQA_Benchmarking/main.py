import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from utils import set_seed
from generators import LLaVa_NeXT_Video_generator , LLaMA3_Video_generator , Instruct_Blip_Video_generator, Qwen2VL_generator


# To-Do: Add more generators in the future and import them here

def main():
    parser = argparse.ArgumentParser(
        description="Select which generator to run and specify a config file."
    )
    parser.add_argument(
        "--generator", type=str, default="llava-next-video", help="Name of the generator to run")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to the configuration JSON file")
    parser.add_argument("--output_file" , type=str, default="output_answers.csv", help="Path to the output file")

    parser.add_argument("--seed", type=int, default=1000, help="Seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)

    # Map generator names to their corresponding functions.
    generators = {
        "llava-next-video": LLaVa_NeXT_Video_generator,
        "llama3_video": LLaMA3_Video_generator,
        "instruct-blip-video": Instruct_Blip_Video_generator,
        "qwen2vl": Qwen2VL_generator,
        # To-Do: "other": OtherModel_generator,
    }

    gen_key = args.generator.lower()
    if gen_key in generators:
        # Pass the config file path to the generator function.
        csv_filename = generators[gen_key](args.config)
        print(f"Generator '{gen_key}' finished. Results saved to: {csv_filename}")
    else:
        print(f"Unknown generator '{args.generator}'. Available options: {', '.join(generators.keys())}")

if __name__ == "__main__":
    main()
