import warnings

warnings.simplefilter("ignore", UserWarning)

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from varname import nameof
from tqdm import trange, tqdm
from torch.optim.lr_scheduler import MultiStepLR
from modules.models import GeneratorFullModel, DiscriminatorFullModel

from logger import Logger
from frames_dataset import DatasetRepeater


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

    optimizers_g = {'appearance_feature_extractor': optimizer_appearance_feature_extractor,
                    'canonical_keypoint_detector': optimizer_canonical_keypoint_detector,
                    'head_expression_estimator': optimizer_head_expression_estimator,
                    'occlusioin_aware_generator': optimizer_occlusion_aware_generator}

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(appearance_feature_extractor,
                                      canonical_keypoint_detector,
                                      head_expression_estimator,
                                      occlusion_aware_generator,
                                      multi_scale_discriminator,
                                      optimizer_appearance_feature_extractor,
                                      optimizer_canonical_keypoint_detector,
                                      optimizer_head_expression_estimator,
                                      optimizer_occlusion_aware_generator,
                                      optimizer_multi_scale_discriminator)
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

    generator_full = GeneratorFullModel(appearance_feature_extractor=appearance_feature_extractor,
                                        canonical_keypoint_detector=canonical_keypoint_detector,
                                        head_expression_estimator=head_expression_estimator,
                                        occlusion_aware_generator=occlusion_aware_generator,
                                        multi_scale_discriminator=multi_scale_discriminator,
                                        train_params=train_params)
                                        
    discriminator_full = DiscriminatorFullModel(multi_scale_discriminator=multi_scale_discriminator,
                                                generator_output_channels=occlusion_aware_generator.output_channels,
                                                train_params=train_params)

    if torch.cuda.is_available():
        print('uploading model to gpus')
        generator_full = DataParallel(generator_full)
        discriminator_full = DataParallel(discriminator_full)

    global_step = 0
    with Logger(log_dir=log_dir, checkpoint_freq=train_params['checkpoint_freq'], visualizer_params=config['visualizer_params']) as logger:

        for epoch in trange(start_epoch, train_params['num_epochs']):
            learning_rates = {key: value.param_groups[0]['lr'] for (key, value) in optimizers_g.items()}
            learning_rates['multi_scale_discriminator'] = optimizer_multi_scale_discriminator.param_groups[0]['lr']

            for x in tqdm(dataloader, total=len(dataloader)):
                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()

                for optimizer in optimizers_g.values():
                    optimizer.step()
                    optimizer.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_multi_scale_discriminator.step()
                    optimizer_multi_scale_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}

                global_step += 1
                logger.log_iter(losses=losses, learning_rates=learning_rates, step=global_step)

            for scheduler in schedulers:
                scheduler.step()

            logger.log_epoch(epoch=epoch,
                            models={
                                'appearance_feature_extractor': appearance_feature_extractor,
                                'canonical_keypoint_detector': canonical_keypoint_detector,
                                'head_expression_estimator': head_expression_estimator,
                                'occlusion_aware_generator': occlusion_aware_generator,
                                'multi_scale_discriminator': multi_scale_discriminator,
                                'optimizer_appearance_feature_extractor': optimizer_appearance_feature_extractor,
                                'optimizer_canonical_keypoint_detector': optimizer_canonical_keypoint_detector,
                                'optimizer_head_expression_estimator': optimizer_head_expression_estimator,
                                'optimizer_occlusion_aware_generator': optimizer_occlusion_aware_generator,
                                'optimizer_multi_scale_discriminator': optimizer_multi_scale_discriminator},
                            inp=x,
                            out=generated)
