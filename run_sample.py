import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
from sampling import OrderedSampler,DiffusionSampler


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="JingyangOu/radd-t-dce", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--method", type=str, default="tweedie") # ordered, euler, tweedie
    parser.add_argument("--strategy", type=str, default="direct") # direct, top_p, top_k
    parser.add_argument("--strategy_para", type=float, default=0.8) # p for top_p, k for top_k, no use when direct 



    args = parser.parse_args()

    
    device = torch.device('cuda')
    model, noise = load_model(args.model_path, device)
    token_dim = model.config.tokens + 1
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
    order =  torch.arange(0,1024)
    if args.method == 'ordered':
        sampler = OrderedSampler(model, (args.batch_size, args.length), token_dim, args.strategy, args.strategy_para, order, device=device)
    elif args.method == 'euler' or args.method == 'tweedie':
        sampler = DiffusionSampler(args.method, model,  noise, (args.batch_size, args.length),token_dim, args.strategy, args.strategy_para, device=device)
    else:
        raise ValueError(f"Method {args.method} is not valid.")
    

    samples = sampler.sample(args.steps)
    text_samples = tokenizer.batch_decode(samples)

    for i in text_samples:
        print(i)
        print("=================================================")

if __name__=="__main__":
    main()