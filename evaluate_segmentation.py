import colorsys
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from skimage.measure import find_contours
from torch.nn.functional import interpolate
from tqdm import tqdm

from utils import get_voc_dataset, get_model, parse_args


def get_attention_masks(args, image, model, device):
    """Original function for binary masks"""
    # make the image divisible by the patch size
    w, h = image.shape[2] - image.shape[2] % args.patch_size, image.shape[3] - \
        image.shape[3] % args.patch_size
    image = image[:, :w, :h]
    w_featmap = image.shape[-2] // args.patch_size
    h_featmap = image.shape[-1] // args.patch_size

    outputs = model(image.to(device), output_attentions=True)
    attentions = outputs.attentions[-1]  # or specific layer

    nh = attentions.shape[1]

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cum_val = torch.cumsum(val, dim=1)
    th_attn = cum_val > (1 - args.threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]

    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = interpolate(th_attn.unsqueeze(
        0), scale_factor=args.patch_size, mode="nearest")[0]

    return th_attn


def get_attention_heatmaps(args, image, model, device, layer_idx=-1):
    """New function for continuous attention heatmaps"""
    # make the image divisible by the patch size
    w, h = image.shape[2] - image.shape[2] % args.patch_size, image.shape[3] - \
        image.shape[3] % args.patch_size
    image = image[:, :w, :h]
    w_featmap = image.shape[-2] // args.patch_size
    h_featmap = image.shape[-1] // args.patch_size

    outputs = model(image.to(device), output_attentions=True)
    attentions = outputs.attentions[layer_idx]  # use specified layer

    nh = attentions.shape[1]

    # Get raw attention values from CLS token to all patches
    raw_attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # Normalize each head to [0, 1] for better visualization
    normalized_attentions = torch.zeros_like(raw_attentions)
    for head in range(nh):
        att_head = raw_attentions[head]
        # Min-max normalization
        att_min, att_max = att_head.min(), att_head.max()
        if att_max > att_min:  # Avoid division by zero
            normalized_attentions[head] = (
                att_head - att_min) / (att_max - att_min)
        else:
            normalized_attentions[head] = att_head

    # Reshape to spatial dimensions
    heatmaps = normalized_attentions.reshape(nh, w_featmap, h_featmap).float()

    # Upsample to original image size using bilinear for smoother heatmaps
    heatmaps = interpolate(heatmaps.unsqueeze(
        0), scale_factor=args.patch_size, mode="bilinear", align_corners=False)[0]

    return heatmaps, image


def get_per_sample_jaccard(pred, target):
    """Original Jaccard computation for binary masks"""
    jac = 0
    object_count = 0
    for mask_idx in torch.unique(target):
        if mask_idx in [0, 255]:  # ignore index
            continue
        cur_mask = target == mask_idx
        intersection = (cur_mask * pred) * \
            (cur_mask != 255)  # handle void labels
        intersection = torch.sum(
            intersection, dim=[1, 2])  # handle void labels
        union = ((cur_mask + pred) > 0) * (cur_mask != 255)
        union = torch.sum(union, dim=[1, 2])
        jac_all = intersection / union
        jac += jac_all.max().item()
        object_count += 1
    return jac / object_count


def run_eval(args, data_loader, model, device):
    """Original evaluation function using binary masks"""
    model.to(device)
    model.eval()
    total_jac = 0
    image_count = 0

    for idx, (sample, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        sample, target = sample.to(device), target.to(device)
        attention_mask = get_attention_masks(args, sample, model, device)
        jac_val = get_per_sample_jaccard(attention_mask, target)
        total_jac += jac_val
        image_count += 1
    return total_jac / image_count


def display_attention_heatmap(image, attention, fname="test", figsize=(10, 5), colormap='jet', alpha=0.6):
    """Display attention as heatmap overlay"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Convert image to numpy
    img_np = image.permute(1, 2, 0).cpu().numpy()
    att_np = attention.detach().cpu().numpy()

    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Image with attention heatmap overlay
    axes[1].imshow(img_np)

    # Create heatmap overlay
    heatmap = cm.get_cmap(colormap)(att_np)
    axes[1].imshow(heatmap, alpha=alpha)
    axes[1].set_title('Attention Heatmap')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    """Original function for binary mask visualization"""
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.cpu().numpy()

    plt.ioff()
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]

    def random_colors(N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        return colors

    colors = random_colors(N)

    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = (image * 255).astype(np.uint32).copy()

    for i in range(N):
        color = colors[i]
        _mask = mask[i]

        # Mask
        masked_image = apply_mask_last(masked_image, _mask, color, alpha)

        # Mask Polygon
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    plt.close(fig)


def apply_mask_last(image, mask, color=(0.0, 0.0, 1.0), alpha=0.5):
    """Helper function for binary mask overlay"""
    for c in range(3):
        image[:, :, c] = image[:, :, c] * \
            (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def generate_images_per_model(args, model, device, use_heatmaps=True, layer_idx=-1):
    """Generate visualizations - can use either binary masks or heatmaps"""
    model.to(device)
    model.eval()

    samples = []
    for im_name in tqdm(os.listdir(args.test_dir)):
        im_path = f"{args.test_dir}/{im_name}"

        # Calculate size divisible by patch_size
        target_size = 512 - (512 % args.patch_size)

        img = Image.open(f"{im_path}").resize((target_size, target_size))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = torchvision.transforms.functional.to_tensor(img)
        samples.append(img)

    samples = torch.stack(samples, 0).to(device)

    if use_heatmaps:
        # Generate attention heatmaps
        attention_maps = []
        processed_images = []
        for sample in samples:
            heatmaps, processed_img = get_attention_heatmaps(
                args, sample.unsqueeze(0), model, device, layer_idx)
            attention_maps.append(heatmaps)
            processed_images.append(processed_img.squeeze(0))

        # Save heatmap visualizations
        os.makedirs(f"{args.save_path}", exist_ok=True)
        os.makedirs(
            f"{args.save_path}/{args.model_name}_heatmaps_layer{layer_idx}", exist_ok=True)

        for idx, (sample, heatmaps) in enumerate(zip(processed_images, attention_maps)):
            for head_idx, heatmap in enumerate(heatmaps):
                f_name = f"{args.save_path}/{args.model_name}_heatmaps_layer{layer_idx}/im_{idx:03d}_head_{head_idx}.png"
                display_attention_heatmap(sample, heatmap, fname=f_name)

    else:
        # Generate binary attention masks (original functionality)
        attention_masks = []
        for sample in samples:
            attention_masks.append(get_attention_masks(
                args, sample.unsqueeze(0), model, device))

        os.makedirs(f"{args.save_path}", exist_ok=True)
        os.makedirs(
            f"{args.save_path}/{args.model_name}_{args.threshold}", exist_ok=True)

        for idx, (sample, mask) in enumerate(zip(samples, attention_masks)):
            for head_idx, mask_h in enumerate(mask):
                f_name = f"{args.save_path}/{args.model_name}_{args.threshold}/im_{idx:03d}_{head_idx}.png"
                display_instances(sample, mask_h, fname=f_name)


def generate_average_attention_heatmap(args, model, device, layer_idx=-1):
    """Generate average attention across all heads for cleaner visualization"""
    model.to(device)
    model.eval()

    samples = []
    im_names = []
    for im_name in tqdm(os.listdir(args.test_dir)):
        im_path = f"{args.test_dir}/{im_name}"
        target_size = 512 - (512 % args.patch_size)
        img = Image.open(f"{im_path}").resize((target_size, target_size))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = torchvision.transforms.functional.to_tensor(img)
        samples.append(img)
        im_names.append(im_name.split('.')[0])

    samples = torch.stack(samples, 0).to(device)

    os.makedirs(f"{args.save_path}", exist_ok=True)
    os.makedirs(
        f"{args.save_path}/{args.model_name}_{args.pretrained_weights}_avg_attention_layer{layer_idx}", exist_ok=True)

    for sample, fn in zip(samples, im_names):
        heatmaps, processed_img = get_attention_heatmaps(
            args, sample.unsqueeze(0), model, device, layer_idx)

        # Average across all heads
        avg_attention = torch.mean(heatmaps, dim=0)

        f_name = f"{args.save_path}/{args.model_name}_{args.pretrained_weights}_avg_attention_layer{layer_idx}/{fn}_avg.png"
        display_attention_heatmap(
            processed_img.squeeze(0), avg_attention, fname=f_name)


if __name__ == '__main__':
    opt = parse_args()
    dino_model, mean, std = get_model(opt)
    device = 'cuda'

    if opt.generate_images:
        # Choose what type of visualization to generate

        # Option 1: Generate heatmaps (like in the paper)
        # generate_images_per_model( opt, dino_model, device, use_heatmaps=True, layer_idx=-1)

        # Option 2: Generate average attention heatmaps (cleaner visualization)
        generate_average_attention_heatmap(
            opt, dino_model, device, layer_idx=-1)

        # Option 3: Generate binary masks (original functionality)
        # generate_images_per_model(opt, dino_model, device, use_heatmaps=False)

    else:
        # Original evaluation using binary masks
        test_dataset, test_data_loader = get_voc_dataset()
        model_accuracy = run_eval(opt, test_data_loader, dino_model, device)
        print(f"Jaccard index for {opt.model_name}: {model_accuracy}")
