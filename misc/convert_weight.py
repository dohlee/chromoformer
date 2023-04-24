import argparse
import torch
import os

from copy import deepcopy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu")
    new_ckpt = deepcopy(ckpt)

    net = ckpt["net"]

    for k, v in net.items():
 
        if "lin_proj_c" in k:
            new_ckpt["net"][k.replace("lin_proj_c", "lin_proj_pcre")] = net[k]
            new_ckpt["net"].pop(k)

    net = new_ckpt['net']
    new_ckpt = deepcopy(new_ckpt)
    
    for k, v in net.items():

        if "transformer2000" in k:
            new_ckpt["net"][k.replace("transformer2000", "regulation.2000.transformer")] = net[k]
            new_ckpt["net"].pop(k)
            continue

        if "transformer500" in k:
            new_ckpt["net"][k.replace("transformer500", "regulation.500.transformer")] = net[k]
            new_ckpt["net"].pop(k)
            continue

        if "transformer100" in k:
            new_ckpt["net"][k.replace("transformer100", "regulation.100.transformer")] = net[k]
            new_ckpt["net"].pop(k)
            continue

        if "embed2000_a" in k:
            new_ckpt["net"][k.replace("embed2000_a", "embed.2000")] = net[k]
            new_ckpt["net"].pop(k)
            continue

        if "embed500_a" in k:
            new_ckpt["net"][k.replace("embed500_a", "embed.500")] = net[k]
            new_ckpt["net"].pop(k)
            continue

        if "embed100_a" in k:
            new_ckpt["net"][k.replace("embed100_a", "embed.100")] = net[k]
            new_ckpt["net"].pop(k)
            continue

        if "embed2000_b" in k:
            new_ckpt["net"][k.replace("embed2000_b", "pairwise_interaction.2000")] = net[k]
            new_ckpt["net"].pop(k)
            continue

        if "embed500_b" in k:
            new_ckpt["net"][k.replace("embed500_b", "pairwise_interaction.500")] = net[k]
            new_ckpt["net"].pop(k)
            continue

        if "embed100_b" in k:
            new_ckpt["net"][k.replace("embed100_b", "pairwise_interaction.100")] = net[k]
            new_ckpt["net"].pop(k)
            continue

        if "embed" in k:
            new_ckpt["net"][k.replace("embed", "embed.")] = net[k]
            new_ckpt["net"].pop(k)
            continue

        if "pw_int" in k:
            new_ckpt["net"][k.replace("pw_int", "pairwise_interaction.")] = net[k]
            new_ckpt["net"].pop(k)
            continue

        if "reg" in k:
            new_ckpt["net"][k.replace("reg", "regulation.")] = net[k]
            new_ckpt["net"].pop(k)
            continue


    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    torch.save(new_ckpt, args.output)


if __name__ == "__main__":
    main()
