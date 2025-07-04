from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_pth: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoints found in the specified path."
            chk_path = checkpoints[0]
            print(f"Loading checkpoint from {chk_path}...")
            checkpoint = torch.load(chk_path, map_location=device)
            print(f'Checkpoint loaded in {time.time() - prev_time:.2f} seconds.')
            prev_time = time.time()

        with open(Path(checkpoints_dir) / 'params.json', 'r') as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_pth)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f'Loaded state dict in {time.time() - prev_time:.2f} seconds.')

        return LLaMA(model, tokenizer, model_args)
    
    def text_completion(self, prompts: list[str], temperature: float=0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.Encode(prompt, out_type=int, add_bos = True, add_eos = False) for prompt in prompts]
        # make sure batch size is not too large
        batch_size = len(prompt_tokens)
        assert self.args.max_batch_size >= batch_size, 'Batch size is large'

        # Make sure that prompt len is not larger than max sequence length 
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert self.args.max_seq_len >= max_prompt_len

        total_len = min(self.args.max_seq_len, max_prompt_len + max_gen_len)

        # create a list that will contain gen tokens plus initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)

        for k,t in enumerate(prompt_tokens):
            # Populate the initial tokens with prompt tokens
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_token_mask = tokens != pad_id # True if token is prompt token, False otherwise

        for cur_pos in tqdm(range(1, total_len), desc='Generating tokens'):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                # The temperature is applied before softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the next token with max probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace the token if it is a padding token
            next_token = torch.where(prompt_token_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            # EOS reached only if we found an EOS token for padding position
            eos_reached |= (~prompt_token_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut the EOS token, if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]

            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.Decode(current_prompt_tokens))

        return (out_tokens, out_text)

    def _sample_top_p(self, probs, p):
        # When sorting probs the actual indices gets mixed up which causes difficulty to decode in later steps hence
        # we also extract probs index so that each token gets its position back after applying top p strategy
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        return next_token

if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot prompt
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Umar Jamil
        Decision: 
        """
    ]

    model = LLaMA.build(
        checkpoints_dir='llama-2-7b',
        tokenizer_pth='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device
    )

    # Inference the model
    out_tokens, out_text = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_text) == len(prompts)
    for i in range(len(out_text)):
        print(f'{out_text[i]}')
        print('-'*50)
