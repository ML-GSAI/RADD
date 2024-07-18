import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
from sampling import OrderedSampler,DiffusionSampler


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="JingyangOu/radd-lambda-dce", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--method", type=str, default="tweedie") # euler, tweedie
    parser.add_argument("--strategy", type=str, default="direct") # direct, top_p, top_k
    parser.add_argument("--strategy_para", type=int, default=0.8) # p for top_p, k for top_k, no use when direct 
    parser.add_argument("--prefix", type=str, default="I have a  ")
    parser.add_argument("--suffix", type=str, default=" in a joint framework.")
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')

    prefix_ids = tokenizer(args.prefix).input_ids
    suffix_ids = tokenizer(args.suffix).input_ids
    input_ids = prefix_ids + suffix_ids
    input_locs = list(range(len(prefix_ids))) + list(range(1024-len(suffix_ids), 1024))

    # more generaly commands can be defined with something like below:
    # input_ids = [0, 1, 512, 8080, 50256, 20000]
    # input_locs = [5, 6, 19, 20, 1000, 10001]


    input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(args.batch_size, 1)

    def proj_fun(x):
        x[:, input_locs] = input_ids
        return x
    
    device = torch.device('cuda')
    model, noise = load_model(args.model_path, device)
    token_dim = model.config.tokens + 1
    

    if args.method == 'euler' or args.method == 'tweedie':
        sampler = DiffusionSampler(args.method, model, noise, (args.batch_size, args.length), token_dim, args.strategy, args.strategy_para, device=device)
    else:
        raise ValueError(f"Method {args.method} is not valid.")

    samples = sampler.sample(args.steps, proj_fun=proj_fun)


    text_samples = tokenizer.batch_decode(samples)
    for i in text_samples:
        print(i)
        print("=================================================")

if __name__=="__main__":
    main()