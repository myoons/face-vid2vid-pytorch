import os
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset

from modules.appearance_feature_extractor import AppearanceFeatureExtractor
from modules.canonical_keypoint_detector import CanonicalKeypointDetector
from modules.head_expression_estimator import HeadExpressionEstimator
from modules.occlusion_aware_generator import OcclusionAwareGenerator
from modules.multi_scale_discriminator import MultiScaleDiscriminator

import torch

from train import train

MODELS = {
    'appearance_feature_extractor': AppearanceFeatureExtractor,
    'canonical_keypoint_detector': CanonicalKeypointDetector,
    'head_expression_estimator': HeadExpressionEstimator,
    'occlusion_aware_generator': OcclusionAwareGenerator,
    'multi_scale_discriminator': MultiScaleDiscriminator
}


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    return parser.parse_args()


def init_model(model_name, configs):
    model = MODELS[model_name](**configs['model_params'][model_name],
                               **configs['model_params']['common_params'])

    if torch.cuda.is_available():
        model = model.cuda()

    if args.verbose:
        print(model)

    return model


if __name__ == '__main__':

    args = arg_parse()
    with open(args.config) as f:
        config = yaml.load(f)

    if args.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(args.checkpoint)[:-1])
    else:
        log_dir = os.path.join(args.log_dir, os.path.basename(args.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    models = dict()
    for model_ in MODELS.keys():
        models[model_] = init_model(model_, config)

    dataset = FramesDataset(is_train=(args.mode == 'train'), **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
        copy(args.config, log_dir)

    if args.mode == 'train':
        print("Training...")
        train(config=config,
              checkpoint=args.checkpoint,
              log_dir=log_dir,
              dataset=dataset,
              device_ids=args.device_ids,
              **models)
