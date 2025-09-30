VQA_DARai_Benchmarking is a benchmarking framework designed for evaluating Visual Question Answering (VQA) models on the DARai dataset. This repository provides tools to generate answers for video-based datasets and store results efficiently.


## Current Model
- **LLaVA-NeXT-Video**
- **Qwen2-VL-7B-Instruct**
- **VideoLLaMA3-2B**
  
## Installation
To use this benchmarking tool, clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone "address"
cd VQA_DARai_Benchmarking

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Running the Benchmarking Script
To generate and save answers from a dataset, use:

```
python main.py --generator "model name" --config "Path to the configuration JSON file" 

```
## Configuration files
- Edit the dataset path according to your system path files
- Adjust output tokens to your preferences. 
## Handling Errors and Resuming Processing
- The script **saves results incrementally** to avoid data loss in case of errors.
- If interrupted, it **resumes processing** from where it left off.
- Error logs are printed to help debug failed instances.

## Output Format
The results are saved in a CSV file with the following columns:

```plaintext
| Activity | Camera | Subject ID | Session ID | Question | Answer |
|----------|--------|------------|------------|----------|---------|
| Running  | cam1  | 001        | 02         | Q1       | MODEL ANSWER |
```

License

This work is licensed under CC BY-SA 4.0 .
