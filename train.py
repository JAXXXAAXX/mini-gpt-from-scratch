import torch


def main():
    print("Mini GPT project started.")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())


if __name__ == "__main__":
    main()