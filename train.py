import warnings
warnings.simplefilter('ignore', UserWarning)

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

from modules.models import GeneratorFullModel, DiscriminatorFullModel
from logger import Logger


def train(config, af_extractor, kp_detector, he_estimator, generator, discriminator, log_dir, dataset, args):
    train_params = config['train_params']

    optimizer_af_extractor = Adam(af_extractor.parameters(), lr=train_params['lr_af_extractor'], betas=(0.5, 0.999))
    optimizer_kp_detector = Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_he_estimator = Adam(he_estimator.parameters(), lr=train_params['lr_he_estimator'], betas=(0.5, 0.999))
    optimizer_generator = Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))

    if args.checkpoint is not None:
        start_epoch = Logger.load_cpk(af_extractor, kp_detector, generator, generator, discriminator,
                                      optimizer_af_extractor, optimizer_kp_detector, optimizer_he_estimator,
                                      optimizer_generator, optimizer_discriminator)
    else:
        start_epoch = 0

    epoch_milestones = train_params['epoch_milestones']
    scheduler_af_extractor = MultiStepLR(optimizer_af_extractor, epoch_milestones, gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, epoch_milestones, gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_he_estimator = MultiStepLR(optimizer_he_estimator, epoch_milestones, gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_generator = MultiStepLR(optimizer_generator, epoch_milestones, gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, epoch_milestones, gamma=0.1, last_epoch=start_epoch - 1)

    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=train_params['batch_size'], drop_last=True, num_workers=16, pin_memory=True)

    generator_full = GeneratorFullModel(af_extractor, kp_detector, he_estimator, generator, discriminator, train_params, args, train=True)
    discriminator_full = DiscriminatorFullModel(discriminator, generator.output_channels, train_params, args)

    if torch.cuda.is_available():
        generator_full = generator_full.to(args.local_rank)
        generator_full = DDP(generator_full, device_ids=[args.local_rank])

        discriminator_full = discriminator_full.to(args.local_rank)
        discriminator_full = DDP(discriminator_full, device_ids=[args.local_rank])

    global_step = 0
    if args.is_main_process:
        logger = Logger(log_dir, train_params['checkpoint_freq'], config['visualizer_params'])

    for epoch in range(start_epoch, train_params['num_epochs']):
        
        optimizer_af_extractor.zero_grad()
        optimizer_kp_detector.zero_grad()
        optimizer_he_estimator.zero_grad()
        optimizer_generator.zero_grad()
        optimizer_discriminator.zero_grad()

        sampler.set_epoch(epoch)

        if args.is_main_process:
            pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch : {epoch}/{train_params['num_epochs']}")
        else:
            pbar = dataloader

        for (source, driving) in pbar:

            source, driving = source.to(args.local_rank), driving.to(args.local_rank)
            x = {'source': source, 'driving': driving}

            losses_generator, generated = generator_full(x)

            loss_values = [val.mean() for val in losses_generator.values()]
            loss = sum(loss_values)
            loss.backward()

            optimizer_af_extractor.step()
            optimizer_kp_detector.step()
            optimizer_he_estimator.step()
            optimizer_generator.step()
            
            optimizer_af_extractor.zero_grad()
            optimizer_kp_detector.zero_grad()
            optimizer_he_estimator.zero_grad()
            optimizer_generator.zero_grad()

            if train_params['loss_weights']['generator_gan'] != 0:                
                losses_discriminator = discriminator_full(x, generated)

                loss_values = [val.mean() for val in losses_discriminator.values()]
                loss = sum(loss_values)
                loss.backward()

                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()
            else:
                losses_discriminator = {}

            losses_generator.update(losses_discriminator)
            losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}

            if args.is_main_process:
                logger.log_iter(losses, global_step)

            global_step += 1

        scheduler_af_extractor.step()
        scheduler_kp_detector.step()
        scheduler_he_estimator.step()
        scheduler_generator.step()
        scheduler_discriminator.step()

        if args.is_main_process:

            logger.log_epoch(epoch=epoch,
                             models={
                                 'af_extractor': af_extractor,
                                 'kp_detector': kp_detector,
                                 'he_estimator': he_estimator,
                                 'generator': generator,
                                 'discriminator': discriminator,
                                 'optimizer_af_extractor': optimizer_af_extractor,
                                 'optimizer_kp_detector': optimizer_kp_detector,
                                 'optimizer_he_estimator': optimizer_he_estimator,
                                 'optimizer_generator': optimizer_generator,
                                 'optimizer_discriminator': optimizer_discriminator},
                             inp=x,
                             out=generated)
