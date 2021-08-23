import yaml
import imageio
import numpy as np
from argparse import ArgumentParser
from skimage import io, img_as_float32

from modules.appearance_feature_extractor import AppearanceFeatureExtractor
from modules.canonical_keypoint_detector import CanonicalKeypointDetector
from modules.head_expression_estimator import HeadExpressionEstimator
from modules.occlusion_aware_generator import OcclusionAwareGenerator
from modules.multi_scale_discriminator import MultiScaleDiscriminator

from modules.models import GeneratorFullModel
import torch
import torch.nn.functional as F

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
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint to load")
    parser.add_argument("--source", required=True, help="path to source image")
    parser.add_argument("--driving", required=True, help="path to driving image")

    return parser.parse_args()


def init_model(model_name, configs):
    model = MODELS[model_name](**configs['model_params'][model_name],
                               **configs['model_params']['common_params'])

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def load_cpk(checkpoint_path, appearance_feature_extractor, canonical_keypoint_detector,
             head_expression_estimator, occlusion_aware_generator, multi_scale_discriminator):

    checkpoint = torch.load(checkpoint_path)

    appearance_feature_extractor.load_state_dict(checkpoint['appearance_feature_extractor'])
    canonical_keypoint_detector.load_state_dict(checkpoint['canonical_keypoint_detector'])
    head_expression_estimator.load_state_dict(checkpoint['head_expression_estimator'])
    occlusion_aware_generator.load_state_dict(checkpoint['occlusion_aware_generator'])

    if multi_scale_discriminator is not None:
        try:
            multi_scale_discriminator.load_state_dict(checkpoint['discriminator'])
        except Exception as e:
            print('No discriminator in the state-dict. Discriminator will be randomly initialized')


if __name__ == '__main__':

    args = arg_parse()
    with open(args.config) as f:
        config = yaml.load(f)

    models = dict()
    for m in MODELS.keys():
        models[m] = init_model(m, config)

    load_cpk(args.checkpoint, **models)

    for model in models.values():
        for param in model.parameters():
            param.requires_grad = False

    generator = GeneratorFullModel(**models, train_params=config['train_params'])
    source = torch.from_numpy(np.array(img_as_float32(io.imread(args.source)), dtype='float32')).permute(2, 0, 1).unsqueeze(0).cuda()
    driving = torch.from_numpy(np.array(img_as_float32(io.imread(args.driving)), dtype='float32')).permute(2, 0, 1).unsqueeze(0).cuda()
    x = {'source': source.cuda(), 'driving': driving.cuda()}

    with torch.no_grad():
        _, generated = generator(x)

    prediction = generated['prediction'].data.cpu().numpy()
    prediction = np.transpose(prediction[0], [1, 2, 0])
    imageio.imsave('prediction.png', prediction)

    occlusion_map = generated['occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
    occlusion_map = F.interpolate(occlusion_map, size=source.shape[2:]).numpy()
    occlusion_map = np.transpose(occlusion_map[0], [1, 2, 0])
    imageio.imsave('occlusion_map.png', occlusion_map)