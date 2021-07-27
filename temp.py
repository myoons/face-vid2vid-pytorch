import torch
import torchvision
from torch.utils.data import DataLoader

from tqdm import tqdm
from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel
from modules.pretrained_hopenet import Hopenet
from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater


def load_pretrained_hopenet(config):
    model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    model.load_state_dict(torch.load(config['pretrained_hopenet_dir']))

    for param in model.parameters():
        param.requires_grad = False

    return model


def train(config, appearance_feature_extractor, canonical_keypoint_detector, head_expression_estimator,
          occlusion_aware_generator, multi_scale_discriminator, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_appearance_feature_extractor = torch.optim.Adam(appearance_feature_extractor.parameters(),
                                                              lr=train_params['lr_appearance_feature_extractor'],
                                                              betas=[0.5, 0.999])
    optimizer_canonical_keypoint_detector = torch.optim.Adam(canonical_keypoint_detector.parameters(),
                                                             lr=train_params['lr_canonical_keypoint_detector'],
                                                             betas=[0.5, 0.999])
    optimizer_head_expression_estimator = torch.optim.Adam(head_expression_estimator.parameters(),
                                                           lr=train_params['lr_head_expression_estimator'],
                                                           betas=[0.5, 0.999])
    optimizer_occlusion_aware_generator = torch.optim.Adam(occlusion_aware_generator.parameters(),
                                                           lr=train_params['lr_occlusion_aware_generator'],
                                                           betas=[0.5, 0.999])
    optimizer_multi_scale_discriminator = torch.optim.Adam(multi_scale_discriminator.parameters(),
                                                           lr=train_params['lr_multi_scale_discriminator'],
                                                           betas=[0.5, 0.999])

    optimizers = [optimizer_appearance_feature_extractor,
                  optimizer_canonical_keypoint_detector,
                  optimizer_head_expression_estimator,
                  optimizer_occlusion_aware_generator,
                  optimizer_multi_scale_discriminator]

    if checkpoint is not None:
        pass
        # TODO : Logger.load_cpk()
    else:
        start_epoch = 0

    scheduler_appearance_feature_extractor = MultiStepLR(optimizer_appearance_feature_extractor,
                                                         train_params['epoch_milestones'],
                                                         gamma=0.1,
                                                         last_epoch=start_epoch - 1)

    scheduler_canonical_keypoint_detector = MultiStepLR(optimizer_canonical_keypoint_detector,
                                                        train_params['epoch_milestones'],
                                                        gamma=0.1,
                                                        last_epoch=start_epoch - 1)

    scheduler_head_expression_estimator = MultiStepLR(optimizer_head_expression_estimator,
                                                      train_params['epoch_milestones'],
                                                      gamma=0.1,
                                                      last_epoch=start_epoch - 1)

    scheduler_occlusion_aware_generator = MultiStepLR(optimizer_occlusion_aware_generator,
                                                      train_params['epoch_milestones'],
                                                      gamma=0.1,
                                                      last_epoch=start_epoch - 1)

    scheduler_multi_scale_discriminator = MultiStepLR(optimizer_multi_scale_discriminator,
                                                      train_params['epoch_milestones'],
                                                      gamma=0.1,
                                                      last_epoch=start_epoch - 1)

    schedulers = [scheduler_appearance_feature_extractor,
                  scheduler_canonical_keypoint_detector,
                  scheduler_head_expression_estimator,
                  scheduler_occlusion_aware_generator,
                  scheduler_multi_scale_discriminator]

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    pretrained_hopenet = load_pretrained_hopenet(config)
    generator_full = GeneratorFullModel(appearance_feature_extractor,
                                        canonical_keypoint_detector,
                                        head_expression_estimator,
                                        pretrained_hopenet,
                                        occlusion_aware_generator,
                                        multi_scale_discriminator,
                                        train_params)
    discriminator_full = DiscriminatorFullModel(occlusion_aware_generator,
                                                multi_scale_discriminator,
                                                train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:

        for epoch in tqdm(start_epoch, total=train_params['num_epochs']):
            for x in dataloader:
                with torch.no_grad():
                    losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                # loss.backward()

                for optimizer in optimizers[:-1]:
                    # optimizer.step()
                    optimizer.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    with torch.no_grad():
                        losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    # loss.backward()
                    # optimizers[-1].step()
                    optimizers[-1].zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            for scheduler in schedulers:
                scheduler.step()

            # TODO : Logger.log_epoch()


if __name__ == '__main__':
    import os
    import sys
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
            model.to(args.device_ids[0])

        if args.verbose:
            print(model)

        return model

    args = arg_parse()
    with open(args.config) as f:
        config = yaml.load(f)

    if args.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(args.checkpoint)[:-1])
    else:
        log_dir = os.path.join(args.log_dir, os.path.basename(args.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    models = []
    for model in MODELS.keys():
        models.append(init_model(model, config))

    input()
    dataset = FramesDataset(is_train=(args.mode == 'train'), **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
        copy(args.config, log_dir)

    if args.mode == 'train':
        print("Training...")
        train(config, *models, args.checkpoint, log_dir, dataset, args.device_ids)
