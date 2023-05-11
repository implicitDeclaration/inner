import argparse
import sys

args = None

def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Robust Repair",
                                     epilog="End of Parameters")

    # Parameters for training
    parser.add_argument(
        "--mode", help="train models and probes, repair model, or test models", type=str, default='train',
    )
    parser.add_argument(
        "--continue_train", help="train bd models from a trained model", action='store_true',
    )
    parser.add_argument(
        "--exchange", help="exchange fault neuron", action='store_true',
    )
    parser.add_argument(
        "--save_dir", help="where to save the trained models", type=str, default='./checkpoints/cifar10',
    )
    parser.add_argument(
        "--rep_dir", help="where to save the repaired models", type=str, default='./checkpoints/repaired/',
    )
    parser.add_argument(
        "--set", help="dataset", default='cifar10', type=str
    )
    parser.add_argument(
        "--arch", help="architecture of model, support resnet, vggnet", default='res18_dense', type=str
    )
    parser.add_argument(
        "--rep_type", help="repair adversarial (adv), backdoor (bd) or misclassification (mis) samples", type=str,
    )
    parser.add_argument(
        "--probe_train_num", help="number of samples to train the probe", type=int, default=1000,
    )
    parser.add_argument(
        "--ratio", help="ratio of bd samples to identify the mask, the greater ratio, the greater clean sample num", type=float, default=0.0
    )
    parser.add_argument(
        "--repair_sample_num", help="number of total samples to repair", type=int,
    )
    parser.add_argument(
        "--rep_layer_num", help="number of layers repair", type=int, default=3
    )
    parser.add_argument(
        "--neuron_num", help="number of neurons to repair", type=int, default=100
    )
    parser.add_argument(
        "--attack_type", help="use which adversarial attack, fgsm, cw", type=str, default='fgsm',
    )
    parser.add_argument(
        "--gpu", help="use which gpu", type=int, default=3,
    )

    args = parser.parse_args()

    return args


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()