import yaml
import imageio
import numpy as np
import face_alignment
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from skimage import io
from skimage.draw import circle
from skimage.transform import resize

from modules.af_extractor import AppearanceFeatureExtractor
from modules.kp_detector import CanonicalKeypointDetector
from modules.he_estimator import HeadExpressionEstimator
from modules.generator import OcclusionAwareGenerator
from modules.pretrained_models import Hopenet

import torch
import torch.nn.functional as F
from torchvision import transforms


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint to load")
    parser.add_argument("--source", required=True, help="path to source image")
    parser.add_argument("--driving", required=True, help="path to driving image")
    parser.add_argument("--img_size", default=256, type=int, help="height, width of image")

    return parser.parse_args()


def get_head_pose(images, face_align, hopenet, he):

    transform_hopenet = transforms.Compose([transforms.Resize(size=(224, 224)),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])

    _, _, h, w = images.shape
    preds = face_align.face_detector.detect_from_batch(images * 255)

    aligned_images = []
    for idx, pred in enumerate(preds):
        if len(pred) == 1:
            lt_x, lt_y, rb_x, rb_y, _ = pred[0]
            bw = rb_x - lt_x
            bh = rb_y - lt_y

            x_min = int(max(lt_x - 2 * bw / 4, 0))
            x_max = int(min(rb_x + 2 * bw / 4, w - 1))
            y_min = int(max(lt_y - 3 * bh / 4, 0))
            y_max = int(min(rb_y + bh / 4, h - 1))

            if y_min == y_max or x_min == x_max:
                img = images[idx]
                img = transform_hopenet(img)
                aligned_images.append(img)
            else:
                img = images[idx, :, y_min:y_max, x_min:x_max]
                img = transform_hopenet(img)
                aligned_images.append(img)
        else:
            img = images[idx]
            img = transform_hopenet(img)
            aligned_images.append(img)

    aligned_tensor = torch.stack(aligned_images)
    yaw, pitch, roll = hopenet(aligned_tensor)

    he['yaw'] = yaw
    he['pitch'] = pitch
    he['roll'] = roll


def load_cpk(ckpt):

    afe = AppearanceFeatureExtractor(**config['model_params']['af_extractor'],
                                     **config['model_params']['common_params'])
    if torch.cuda.is_available():
        afe = afe.cuda()
    afe.load_state_dict(ckpt['af_extractor'])
    afe.eval()

    kpd = CanonicalKeypointDetector(**config['model_params']['kp_detector'],
                                    **config['model_params']['common_params'])
    if torch.cuda.is_available():
        kpd = kpd.cuda()
    kpd.load_state_dict(ckpt['kp_detector'])
    kpd.eval()

    hee = HeadExpressionEstimator(**config['model_params']['he_estimator'],
                                  **config['model_params']['common_params'])
    if torch.cuda.is_available():
        hee = hee.cuda()
    hee.load_state_dict(ckpt['he_estimator'])
    hee.eval()

    g = OcclusionAwareGenerator(**config['model_params']['generator'],
                                **config['model_params']['common_params'])
    if torch.cuda.is_available():
        g = g.cuda()
    g.load_state_dict(ckpt['generator'])
    g.eval()

    hn = Hopenet(requires_grad=False, weight_dir='pretrained/hopenet.pkl')
    if torch.cuda.is_available():
        hn = hn.cuda()
    hn.eval()

    if torch.cuda.is_available():
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:0')
    else:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

    return afe, kpd, hee, g, hn, fa


def pred_to_degree(pred, idx_tensor):
    degree = torch.sum(torch.softmax(pred, dim=-1) * idx_tensor, 1) * 3 - 99
    degree = degree / 180 * 3.14
    return degree


def get_rotation_matrix(yaw, pitch, roll):
    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    roll_mat = torch.cat([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll),
                          torch.zeros_like(roll), torch.cos(roll), -torch.sin(roll),
                          torch.zeros_like(roll), torch.sin(roll), torch.cos(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    pitch_mat = torch.cat([torch.cos(pitch), torch.zeros_like(pitch), torch.sin(pitch),
                           torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
                           -torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw),
                         torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
                         torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    rotation_matrix = torch.einsum('bij,bjk,bkm->bim', roll_mat, pitch_mat, yaw_mat)
    return rotation_matrix


def get_keypoint(he_out, kp_canonical):
    value = {}

    kp = kp_canonical['keypoints']
    yaw, pitch, roll = he_out['yaw'], he_out['pitch'], he_out['roll']
    translation = he_out['translation']
    deformation = he_out['deformation']

    idx_tensor = torch.arange(66, dtype=torch.float32)
    if torch.cuda.is_available():
        idx_tensor = idx_tensor.cuda()

    yaw = pred_to_degree(yaw, idx_tensor)
    pitch = pred_to_degree(pitch, idx_tensor)
    roll = pred_to_degree(roll, idx_tensor)

    rotation_matrix = get_rotation_matrix(yaw, pitch, roll)

    kp_rotated = torch.einsum('bmp,bkp->bkm', rotation_matrix, kp)
    kp_translated = kp_rotated + translation
    kp_deformed = kp_translated + deformation
    value['keypoints'] = kp_deformed

    if 'jacobian' in kp_canonical:
        jacobian = kp_canonical['jacobian']  # (bs, k ,3, 3)
        jacobian_rotated = torch.einsum('bmp,bkps->bkms', rotation_matrix, jacobian)
        value['jacobian'] = jacobian_rotated

    return value


def draw_image_with_kp(image, kp_array):
    image = np.copy(image)
    spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
    kp_array = spatial_size * (kp_array + 1) / 2

    colormap = plt.get_cmap('gist_rainbow')
    num_kp = kp_array.shape[0]
    for kp_idx, kp in enumerate(kp_array):
        rr, cc = circle(kp[0], kp[1], 3, shape=image.shape[:2])
        image[rr, cc] = np.array(colormap(kp_idx / num_kp))[:3]

    return image


def create_image_column_with_kp(images, kp):
    image_array = np.array([draw_image_with_kp(v, k) for v, k in zip(images, kp)])

    return create_image_column(image_array)


def create_image_column(images):
    images = np.copy(images)
    images[:, :, [0, -1]] = (1, 1, 1)
    return np.concatenate(list(images), axis=0)


def visualize(source, driving, out):
    images = []

    # Source image with keypoints
    source = source.data.cpu()
    source = np.transpose(source, [0, 2, 3, 1])
    kp_source = out['kp_source']['keypoints'].data[..., :2].cpu().numpy()
    images.append((source, kp_source))

    # Equivariance visualization
    if 'transformed_frame' in out:
        transformed = out['transformed_frame'].data.cpu().numpy()
        transformed = np.transpose(transformed, [0, 2, 3, 1])
        transformed_kp = out['transformed_kp']['keypoints'].data[..., :2].cpu().numpy()
        images.append((transformed, transformed_kp))

    # Driving image with keypoints
    kp_driving = out['kp_driving']['keypoints'].data[..., :2].cpu().numpy()
    driving = driving.data.cpu().numpy()
    driving = np.transpose(driving, [0, 2, 3, 1])
    images.append((driving, kp_driving))

    # Result with and without keypoints
    prediction = out['prediction'].data.cpu().numpy()
    prediction = np.transpose(prediction, [0, 2, 3, 1])
    if 'kp_norm' in out:
        kp_norm = out['kp_norm']['keypoints'].data.cpu().numpy()
        images.append((prediction, kp_norm))
    images.append(prediction)

    # Occlusion map
    if 'occlusion_map' in out:
        occlusion_map = out['occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
        occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
        occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
        images.append(occlusion_map)

    image = create_image_grid(*images)
    image = (255 * image).astype(np.uint8)
    return image


def create_image_grid(*args):
    out = []
    for arg in args:
        if type(arg) == tuple:
            out.append(create_image_column_with_kp(arg[0], arg[1]))
        else:
            out.append(create_image_column(arg))

    return np.concatenate(out, axis=1)


if __name__ == '__main__':

    args = arg_parse()
    with open(args.config) as f:
        config = yaml.load(f)

    if torch.cuda.is_available():
        checkpoint = torch.load(args.checkpoint)
    else:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))

    af_extractor, kp_detector, he_estimator, generator, hopenet, face_align = load_cpk(checkpoint)

    source = torch.from_numpy(resize(io.imread(args.source), (args.img_size, args.img_size)).astype(np.float32))\
        .permute(2, 0, 1).unsqueeze(0)
    driving = torch.from_numpy(resize(io.imread(args.driving), (args.img_size, args.img_size)).astype(np.float32))\
        .permute(2, 0, 1).unsqueeze(0)
    if torch.cuda.is_available():
        source = source.cuda()
        driving = driving.cuda()

    kp_canonical = kp_detector(source)
    he_source = he_estimator(source)
    he_driving = he_estimator(driving)

    get_head_pose(source, face_align, hopenet, he_source)
    get_head_pose(driving, face_align, hopenet, he_driving)

    kp_source = get_keypoint(he_source, kp_canonical)
    kp_driving = get_keypoint(he_driving, kp_canonical)

    appearance_feature = af_extractor(source)
    generated = generator(appearance_feature, kp_source, kp_driving)
    generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

    image = visualize(source, driving, generated)
    imageio.imsave('sample/out.png', image)
