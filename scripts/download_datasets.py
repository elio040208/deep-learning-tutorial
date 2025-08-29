import argparse
from pathlib import Path

from torchvision import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "mnist"], help="Dataset to download")
    parser.add_argument("--root", type=str, default="data", help="Root directory")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "cifar10":
        datasets.CIFAR10(root=str(root), train=True, download=True)
        datasets.CIFAR10(root=str(root), train=False, download=True)
    elif args.dataset == "mnist":
        datasets.MNIST(root=str(root), train=True, download=True)
        datasets.MNIST(root=str(root), train=False, download=True)

    print(f"Downloaded {args.dataset} to {root}")


if __name__ == "__main__":
    main()
