import argparse

parser = argparse.ArgumentParser(description='Finetune existing model with actor critic')
parser.add_argument('--learning-rate', '-lr', type=float, help='Learning rate')
parser.add_argument('--epoch', '-e', type=float, help='Learning rate')
parser.add_argument('--metrics', '-m', type=str, help='Metrics for reward module', choices=['f1', 'recall', 'precision'])
parser.add_argument('--batch_size', '-bs', type=int, help='Batch size for both training and validating')
parser.add_argument('--checkpoint', '-c', type=str, help='Relative checkpoint path, should be pth.tar file')
args = parser.parse_args()
print(args)
print(args.checkpoint)