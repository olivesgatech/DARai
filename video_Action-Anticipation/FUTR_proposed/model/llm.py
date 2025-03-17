import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class VideoToLabelLLM(nn.Module):
    def __init__(self, llm_model_name='"meta-llama/Llama-2-7b-hf"', embedding_dim=512, max_length=900):
        super(VideoToLabelLLM, self).__init__()

        # Load pre-trained LLM (e.g., GPT-2 or similar)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

        # Linear layer to project video embedding E to LLM's embedding size
        self.embedding_projector = nn.Linear(embedding_dim, self.llm.config.n_embd)  # Match LLM embedding size

    def forward(self, video_embedding, human_prompt):
        """
        Args:
            video_embedding: Encoded image sequence vector (batch_size x seq_len x embedding_dim).
            human_prompt: Textual prompt provided by the user (list of strings).
        Returns:
            Generated labels from the LLM (detached output).
        """
        video_embedding = video_embedding.unsqueeze(0)
        # Project video embeddings to LLM input dimension
        projected_embedding = self.embedding_projector(video_embedding)  # Shape: (batch_size, seq_len, LLM embedding size)

        # Average pooling to condense seq_len into a single embedding (optional)
        #pooled_embedding = projected_embedding.mean(dim=1, keepdim=True)  # Shape: (batch_size, LLM embedding size)

        # Tokenize the human prompt
        tokenized_prompt = self.tokenizer(
            human_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Embed the tokenized prompt using LLM's embedding layer
        with torch.no_grad():  # Prevent gradients from flowing through LLM
            prompt_input_ids = tokenized_prompt["input_ids"].to(self.llm.device)
            prompt_attention_mask = tokenized_prompt["attention_mask"].to(self.llm.device)
            prompt_embeddings = self.llm.transformer.wte(prompt_input_ids)  # Shape: (batch_size, prompt_seq_len, embedding_size)

            # Concatenate the video embedding with the prompt embeddings
            combined_embeddings = torch.cat([projected_embedding, prompt_embeddings], dim=1)  # Combine video + prompt

            extended_attention_mask = torch.cat(
                [
                    torch.ones((prompt_attention_mask.size(0), projected_embedding.size(1)), device=prompt_attention_mask.device),
                    prompt_attention_mask,
                ],
                dim=1,
            )

            # Forward pass through the LLM
            outputs = self.llm(inputs_embeds=combined_embeddings, attention_mask=extended_attention_mask)

        # Decode the generated text (fine-grained labels)
        generated_texts = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)

        return generated_texts