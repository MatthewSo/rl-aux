# Use argparse to get gpu id and run id
import argparse
parser = argparse.ArgumentParser(description='WAMAL CIFAR100-20 MAXL Training')
# REQUIRED GPU FIELD, NO DEFAULT
parser.add_argument('--gpu', type=int, required=True, help='GPU device ID to use (required)')
# REQUIRED RUN_ID FIELD, NO DEFAULT
parser.add_argument('--run_id', type=int, required=True, help='Run ID for the experiment (required)')

args = parser.parse_args()
GPU = args.gpu
RUN_ID = args.run_id
