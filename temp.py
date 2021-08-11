import yaml

from modules.appearance_feature_extractor import AppearanceFeatureExtractor
from modules.canonical_keypoint_detector import CanonicalKeypointDetector
from modules.head_expression_estimator import HeadExpressionEstimator
from modules.occlusion_aware_generator import OcclusionAwareGenerator
from modules.multi_scale_discriminator import MultiScaleDiscriminator

import torch


MODELS = {
    'appearance_feature_extractor': AppearanceFeatureExtractor,
    'canonical_keypoint_detector': CanonicalKeypointDetector,
    'head_expression_estimator': HeadExpressionEstimator,
    'occlusion_aware_generator': OcclusionAwareGenerator,
    'multi_scale_discriminator': MultiScaleDiscriminator
}


def init_model(model_name, configs):
    model = MODELS[model_name](**configs['model_params'][model_name],
                               **configs['model_params']['common_params'])
	
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model


if __name__ == '__main__':

    with open('config/vox.yaml') as f:
        config = yaml.load(f)

    models = dict()
    for model_ in MODELS.keys():
        models[model_] = init_model(model_, config)

    from modules.models import GeneratorFullModel, DiscriminatorFullModel

    generator_full = GeneratorFullModel(**models,
                                        train_params=config['train_params'])
    discriminator_full = DiscriminatorFullModel(multi_scale_discriminator=models['multi_scale_discriminator'],
                                                generator_output_channels=models['occlusion_aware_generator'].output_channels,
                                                train_params=config['train_params'])

    def count_model_params(model, prefix=""):
        n_params = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            n_params += nn

        print(f'{prefix} PARAMS : {n_params / 1e6:.2f} M')

    count_model_params(generator_full, 'GENERATOR FULL')
    count_model_params(discriminator_full, 'DISCRIMINATOR FULL')

   #  for param in generator_full.parameters():
   #     param.requires_grad = False
   #
   #  for param in discriminator_full.parameters():
   #     param.requires_grad = False

    x = {'source': torch.rand((1, 3, 256, 256)).cuda(),
         'driving': torch.rand((1, 3, 256, 256)).cuda()}
    # with torch.no_grad():
    g_loss, generated = generator_full(x)
    # print('temp')
