import os
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset

from modules.af_extractor import AppearanceFeatureExtractor
from modules.kp_detector import CanonicalKeypointDetector
from modules.he_estimator import HeadExpressionEstimator
from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator

import torch
import torch.distributed as dist

from train import train

MODELS = {
    'af_extractor': AppearanceFeatureExtractor,
    'kp_detector': CanonicalKeypointDetector,
    'he_estimator': HeadExpressionEstimator,
    'generator': OcclusionAwareGenerator,
    'discriminator': MultiScaleDiscriminator
}


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0,1,2,3", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    # For Distributed Data Parallel training
    parser.add_argument("--local_rank", type=int, default=0, metavar='N', help='local process rank')

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    with open(args.config) as f:
        config = yaml.load(f)

    # For Distributed Data Parallel training
    # dist.init_process_group(backend='nccl', init_method='env://')
    # torch.cuda.set_device(args.local_rank)

    args.is_main_process = args.local_rank == 0

    if args.is_main_process:
        if args.checkpoint is not None:
            log_dir = os.path.join(*os.path.split(args.checkpoint)[:-1])
        else:
            log_dir = os.path.join(args.log_dir, os.path.basename(args.config).split('.')[0])
            log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    if args.is_main_process:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            copy(args.config, log_dir)

    af_extractor = AppearanceFeatureExtractor(**config['model_params']['af_extractor'],
                                              **config['model_params']['common_params'])

    if torch.cuda.is_available():
        af_extractor.cuda(args.local_rank)
    if args.verbose:
        print(af_extractor)

    kp_detector = CanonicalKeypointDetector(**config['model_params']['kp_detector'],
                                            **config['model_params']['common_params'])

    if torch.cuda.is_available():
        kp_detector.cuda(args.local_rank)
    if args.verbose:
        print(kp_detector)

    he_estimator = HeadExpressionEstimator(**config['model_params']['he_estimator'],
                                           **config['model_params']['common_params'])

    if torch.cuda.is_available():
        he_estimator.cuda(args.local_rank)
    if args.verbose:
        print(he_estimator)

    generator = OcclusionAwareGenerator(**config['model_params']['generator'],
                                        **config['model_params']['common_params'])

    if torch.cuda.is_available():
        generator.cuda(args.local_rank)
    if args.verbose:
        print(generator)

    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator'],
                                            **config['model_params']['common_params'])

    if torch.cuda.is_available():
        discriminator.cuda(args.local_rank)
    if args.verbose:
        print(discriminator)

    dataset = FramesDataset(is_train=(args.mode == 'train'), **config['dataset_params'])
    
    if args.mode == 'train':
        train(config, af_extractor, kp_detector, he_estimator, generator, discriminator, log_dir, dataset, args)
