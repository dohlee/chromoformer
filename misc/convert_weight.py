import argparse
import torch

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
        if "embed" in k:
            new_ckpt["net"][k.replace("embed", "embed.")] = net[k]
            new_ckpt["net"].pop(k)
        if "pw_int" in k:
            new_ckpt["net"][k.replace("pw_int", "pairwise_interaction.")] = net[k]
            new_ckpt["net"].pop(k)
        if "reg" in k:
            new_ckpt["net"][k.replace("reg", "regulation.")] = net[k]
            new_ckpt["net"].pop(k)

    torch.save(new_ckpt, args.output)


if __name__ == "__main__":
    main()
