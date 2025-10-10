# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import gc
import math
import pkg_resources
from functools import partial
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.transforms import Compose
from mmengine.config import Config, DictAction
from mmengine.dataset import default_collate
from mmengine.utils import to_2tuple
from mmengine.utils.dl_utils import is_norm

from mmpretrain import digit_version
from mmpretrain.apis import get_model
from mmpretrain.registry import TRANSFORMS

try:
    import pytorch_grad_cam as cam
    from pytorch_grad_cam.activations_and_gradients import \
        ActivationsAndGradients
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError('Please run `pip install "grad-cam>=1.3.6"` to install '
                      '3rd party package pytorch_grad_cam.')

# Alias name
METHOD_MAP = {
    'gradcam++': cam.GradCAMPlusPlus,
}
METHOD_MAP.update({
    cam_class.__name__.lower(): cam_class
    for cam_class in cam.base_cam.BaseCAM.__subclasses__()
})


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('img', nargs='?', default=None, help='Image file (for single image mode)')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--batch-file',
        type=str,
        help='Text file containing image paths and labels for batch processing '
        '(format: "image_path label" per line)')
    parser.add_argument(
        '--out-dir',
        type=Path,
        help='Output directory for batch processing (CAM images will be saved here)')
    parser.add_argument(
        '--clear-cache-interval',
        type=int,
        default=10,
        help='Clear GPU cache every N images during batch processing (default: 10)')
    parser.add_argument(
        '--target-layers',
        default=[],
        nargs='+',
        type=str,
        help='The target layers to get CAM, if not set, the tool will '
        'specify the norm layer in the last block. Backbones '
        'implemented by users are recommended to manually specify'
        ' target layers in commmad statement.')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--method',
        default='GradCAM',
        help='Type of method to use, supports '
        f'{", ".join(list(METHOD_MAP.keys()))}.')
    parser.add_argument(
        '--target-category',
        default=[],
        nargs='+',
        type=int,
        help='The target category to get CAM, default to use result '
        'get from given model.')
    parser.add_argument(
        '--eigen-smooth',
        default=False,
        action='store_true',
        help='Reduce noise by taking the first principle componenet of '
        '``cam_weights*activations``')
    parser.add_argument(
        '--aug-smooth',
        default=False,
        action='store_true',
        help='Wether to use test time augmentation, default not to use')
    parser.add_argument(
        '--save-path',
        type=Path,
        help='The path to save visualize cam image, default not to save.')
    parser.add_argument('--device', default='cpu', help='Device to use cpu')
    parser.add_argument(
        '--vit-like',
        action='store_true',
        help='Whether the network is a ViT-like network.')
    parser.add_argument(
        '--num-extra-tokens',
        type=int,
        help='The number of extra tokens in ViT-like backbones. Defaults to'
        ' use num_extra_tokens of the backbone.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    if args.method.lower() not in METHOD_MAP.keys():
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(list(METHOD_MAP.keys()))}.')
    
    # Validate arguments
    if args.batch_file and not args.out_dir:
        raise ValueError('--out-dir is required when using --batch-file')
    if not args.batch_file and not args.img:
        raise ValueError('Either provide an image file or use --batch-file for batch processing')

    return args


def reshape_transform(tensor, model, args):
    """Build reshape_transform for `cam.activations_and_grads`, which is
    necessary for ViT-like networks."""
    # ViT_based_Transformers have an additional clstoken in features
    if tensor.ndim == 4:
        # For (B, C, H, W)
        return tensor
    elif tensor.ndim == 3:
        if not args.vit_like:
            raise ValueError(f"The tensor shape is {tensor.shape}, if it's a "
                             'vit-like backbone, please specify `--vit-like`.')
        # For (B, L, C)
        num_extra_tokens = args.num_extra_tokens or getattr(
            model.backbone, 'num_extra_tokens', 1)

        tensor = tensor[:, num_extra_tokens:, :]
        # get heat_map_height and heat_map_width, preset input is a square
        heat_map_area = tensor.size()[1]
        height, width = to_2tuple(int(math.sqrt(heat_map_area)))
        assert height * height == heat_map_area, \
            (f"The input feature's length ({heat_map_area+num_extra_tokens}) "
             f'minus num-extra-tokens ({num_extra_tokens}) is {heat_map_area},'
             ' which is not a perfect square number. Please check if you used '
             'a wrong num-extra-tokens.')
        # (B, L, C) -> (B, H, W, C)
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        # (B, H, W, C) -> (B, C, H, W)
        result = result.permute(0, 3, 1, 2)
        return result
    else:
        raise ValueError(f'Unsupported tensor shape {tensor.shape}.')


def init_cam(method, model, target_layers, use_cuda, reshape_transform):
    """Construct the CAM object once, In order to be compatible with
    mmpretrain, here we modify the ActivationsAndGradients object."""
    GradCAM_Class = METHOD_MAP[method.lower()]
    #cam = GradCAM_Class(
        #model=model, target_layers=target_layers, use_cuda=use_cuda)
    cam = GradCAM_Class(
        model=model, target_layers=target_layers)
    # Release the original hooks in ActivationsAndGradients to use
    # ActivationsAndGradients.
    cam.activations_and_grads.release()
    cam.activations_and_grads = ActivationsAndGradients(
        cam.model, cam.target_layers, reshape_transform)

    return cam


def get_layer(layer_str, model):
    """get model layer from given str."""
    for name, layer in model.named_modules():
        if name == layer_str:
            return layer
    raise AttributeError(
        f'Cannot get the layer "{layer_str}". Please choose from: \n' +
        '\n'.join(name for name, _ in model.named_modules()))


def show_cam_grad(grayscale_cam, src_img, title, out_path=None):
    """fuse src_img and grayscale_cam and show or save."""
    grayscale_cam = grayscale_cam[0, :]
    src_img = np.float32(src_img) / 255
    visualization_img = show_cam_on_image(
        src_img, grayscale_cam, use_rgb=False)

    if out_path:
        # 1. overlay 이미지 저장
        mmcv.imwrite(visualization_img, str(out_path))

        # 2. 순수 히트맵도 함께 저장
        # heatmap_path = str(out_path).replace('.jpg', '_heatmap.npy')
        # np.save(heatmap_path, grayscale_cam)
    else:
        mmcv.imshow(visualization_img, win_name=title)


def get_default_target_layers(model, args):
    """get default target layers from given model, here choose nrom type layer
    as default target layer."""
    norm_layers = [
        (name, layer)
        for name, layer in model.backbone.named_modules(prefix='backbone')
        if is_norm(layer)
    ]
    if args.vit_like:
        # For ViT models, the final classification is done on the class token.
        # And the patch tokens and class tokens won't interact each other after
        # the final attention layer. Therefore, we need to choose the norm
        # layer before the last attention layer.
        num_extra_tokens = args.num_extra_tokens or getattr(
            model.backbone, 'num_extra_tokens', 1)

        # models like swin have no attr 'out_type', set out_type to avg_featmap
        out_type = getattr(model.backbone, 'out_type', 'avg_featmap')
        if out_type == 'cls_token' or num_extra_tokens > 0:
            # Assume the backbone feature is class token.
            name, layer = norm_layers[-3]
            print('Automatically choose the last norm layer before the '
                  f'final attention block "{name}" as the target layer.')
            return [layer]

    # For CNN models, use the last norm layer as the target-layer
    name, layer = norm_layers[-1]
    print('Automatically choose the last norm layer '
          f'"{name}" as the target layer.')
    return [layer]


def clear_memory():
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_single_image(img_path, label, model, transforms, cam_calculator, 
                         args, grad_cam_v):
    """Process a single image and generate CAM visualization."""
    # Apply transform and prepare data
    data = transforms({'img_path': img_path})
    src_img = copy.deepcopy(data['inputs']).numpy().transpose(1, 2, 0)
    data = model.data_preprocessor(default_collate([data]), False)
    
    # Prepare target category
    targets = None
    if label is not None:
        if digit_version(grad_cam_v) >= digit_version('1.3.7'):
            from pytorch_grad_cam.utils.model_targets import \
                ClassifierOutputTarget
            targets = [ClassifierOutputTarget(label)]
        else:
            targets = [label]
    
    # Calculate CAM
    grayscale_cam = cam_calculator(
        data['inputs'],
        targets,
        eigen_smooth=args.eigen_smooth,
        aug_smooth=args.aug_smooth)
    
    return grayscale_cam, src_img


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build the model from a config file and a checkpoint file
    model: nn.Module = get_model(cfg, args.checkpoint, device=args.device)
    if args.preview_model:
        print(model)
        print('\n Please remove `--preview-model` to get the CAM.')
        return

    # apply transform
    transforms = Compose(
        [TRANSFORMS.build(t) for t in cfg.test_dataloader.dataset.pipeline])

    # build target layers
    if args.target_layers:
        target_layers = [
            get_layer(layer, model) for layer in args.target_layers
        ]
    else:
        target_layers = get_default_target_layers(model, args)

    # init a cam grad calculator
    use_cuda = ('cuda' in args.device)
    cam_calculator = init_cam(args.method, model, target_layers, use_cuda,
                              partial(reshape_transform, model=model, args=args))

    # Get grad_cam version for compatibility
    grad_cam_v = pkg_resources.get_distribution('grad_cam').version

    try:
        # Batch processing mode
        if args.batch_file:
            print(f"Batch processing mode: {args.batch_file}")
            args.out_dir.mkdir(parents=True, exist_ok=True)
            
            with open(args.batch_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            total = len(lines)
            print(f"Processing {total} images...")
            
            for idx, line in enumerate(lines, 1):
                parts = line.split()
                img_path = parts[0]
                label = int(parts[1]) if len(parts) > 1 else None
                
                img = Path(img_path)
                save_path = args.out_dir / f"{img.stem}_cam.jpg"
                
                print(f"[{idx}/{total}] Processing: {img.name}")
                
                # Process image
                grayscale_cam, src_img = process_single_image(
                    img_path, label, model, transforms, cam_calculator,
                    args, grad_cam_v)
                
                # Save result
                show_cam_grad(grayscale_cam, src_img, 
                             title=args.method, out_path=save_path)
                
                # Clear memory periodically
                if idx % args.clear_cache_interval == 0:
                    print(f"  -> Clearing GPU cache (every {args.clear_cache_interval} images)")
                    clear_memory()
            
            print(f"Batch processing completed! Results saved to: {args.out_dir}")
        
        # Single image mode
        else:
            label = args.target_category[0] if args.target_category else None
            
            grayscale_cam, src_img = process_single_image(
                args.img, label, model, transforms, cam_calculator,
                args, grad_cam_v)
            
            show_cam_grad(grayscale_cam, src_img, 
                         title=args.method, out_path=args.save_path)
    
    finally:
        # Clean up resources
        print("Cleaning up resources...")
        del model
        del cam_calculator
        clear_memory()


if __name__ == '__main__':
    main()
