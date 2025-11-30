import shutil
import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional
import cv2
import sys
import argparse
import supervision as sv
from pathlib import Path
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from segment_anything import sam_model_registry, SamPredictor

def show_masks(image, masks, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([39, 144, 255, 153])
    for mask in masks:
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1,1,-1)
        mask_image = mask_image.astype(np.uint8)

        cv2.addWeighted(mask_image[:,:,:3], 0.6, image, 1, 0, image)

def show_points(image, coords, labels, marker_size=20):
    coords = np.array(coords)
    for coord, label in zip(coords, labels):
        color = (0,255,0) if label == 1 else (0, 0, 255)
        cv2.drawMarker(image, tuple(coord.astype(int)), color, cv2.MARKER_STAR, marker_size, 2)

def prepare_content_frames__(video_path, content_dir='content_video_for_SAM2'):
    content_path = Path(content_dir)
    # Remove directory if it exists, then create it
    shutil.rmtree(content_path, ignore_errors=True)
    content_path.mkdir(exist_ok=True)

    # Generate and save frames
    frames_generator = sv.get_video_frames_generator(video_path)
    with sv.ImageSink(target_dir_path=content_path, image_name_pattern="{:05d}.JPEG") as sink:
        for frame in frames_generator:
            sink.save_image(frame)

    return content_path

def prepare_content_frames(video_path, content_dir='content_video_for_SAM2'):
    content_path = Path(content_dir)
    # Remove directory if it exists, then create it
    shutil.rmtree(content_path, ignore_errors=True)
    content_path.mkdir(exist_ok=True)

    # Generate and save frames
    frames_generator = sv.get_video_frames_generator(video_path)
    with sv.ImageSink(target_dir_path=content_path, image_name_pattern="{:05d}.JPEG") as sink:
        for frame in frames_generator:
            sink.save_image(frame)

    return content_path

def img_to_tensor(img):
    return (torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.).unsqueeze(0)

def tensor_to_img(img):
    if img.dtype == torch.bfloat16:
        img = img.float()
    return (img[0].data.cpu().numpy().transpose((1, 2, 0)).clip(0, 1) * 255 + 0.5).astype(np.uint8)

def resize_video(cap, long_side=512, keep_ratio=True):
    # Get original video dimensions
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate new dimensions
    if keep_ratio:
        if original_h < original_w:
            new_h = int(long_side * original_h / original_w)
            new_w = int(long_side)
        else:
            new_w = int(long_side * original_w / original_h)
            new_h = int(long_side)
    else:
        new_w, new_h = long_side, long_side

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./output/resized_video.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (new_w, new_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        resized_frame = cv2.resize(frame, (new_w, new_h))

        # Write the resized frame
        out.write(resized_frame)

    # Release resources
    cap.release()
    out.release()

    return './output/resized_video.mp4', (original_w, original_h)

def play_video_until_paused(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None, None

    current_frame = 0
    paused_frame_index = None
    paused_frame_image = None

    while True:
        ret, frame = cap.read()
        if not ret:
            current_frame = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.imshow('Video', frame)
        key = cv2.waitKey(25) & 0xFF

        if key == ord('p'):
            paused_frame_index = current_frame
            paused_frame_image = frame.copy()  # Make a copy of the current frame
            break
        elif key == ord('q'):
            break

        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()
    return paused_frame_index, paused_frame_image

def resize(img, long_side=512, keep_ratio=True):
    if keep_ratio:
        h, w = img.shape[:2]
        if h < w:
            new_h = int(long_side * h / w)
            new_w = int(long_side)
        else:
            new_w = int(long_side * w / h)
            new_h = int(long_side)
        return cv2.resize(img, (new_w, new_h))
    else:
        return cv2.resize(img, (long_side, long_side))

def padding(img, factor=32):
    h, w = img.shape[:2]
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    new_img = np.zeros((h + pad_h, w + pad_w, img.shape[2]), dtype=img.dtype)
    new_img[:h, :w, :] = img
    return new_img

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def get_key(feats, last_layer_idx):
    results = []
    _, _, h, w = feats[last_layer_idx].shape
    for i in range(last_layer_idx):
        results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
    results.append(mean_variance_norm(feats[last_layer_idx]))
    return torch.cat(results, dim=1)

class AdaAttN(nn.Module):

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None, content_masks=None, style_masks=None):
        if content_masks is None:
            content_masks = []
        if style_masks is None:
            style_masks = []
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        style_masks = [nn.functional.interpolate(mask, size=(h_g, w_g), mode='nearest').view(
            b, mask.size(1), h_g * w_g).contiguous() for mask in style_masks]
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_masks = [mask[:, :, index] for mask in style_masks]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        content_masks = [nn.functional.interpolate(mask, size=(h, w), mode='nearest').view(
            b, mask.size(1), w * h).permute(0, 2, 1).contiguous() for mask in content_masks]
        S = torch.bmm(F, G)
        for content_mask, style_mask in zip(content_masks, style_masks):
            style_mask = 1. - style_mask
            attn_mask = torch.bmm(content_mask, style_mask)
            S = S.masked_fill(attn_mask.bool(), -1e15)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean

class Transformer(nn.Module):

    def __init__(self, in_planes, key_planes=None):
        super(Transformer, self).__init__()
        self.ada_attn_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
        self.ada_attn_5_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes + 512)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1, content4_1_key, style4_1_key,
                content5_1_key, style5_1_key, seed=None, content_masks=None, style_masks=None):
        return self.merge_conv(self.merge_conv_pad(
            self.ada_attn_4_1(
                content4_1, style4_1, content4_1_key, style4_1_key, seed, content_masks, style_masks) +
            self.upsample5_1(self.ada_attn_5_1(
                content5_1, style5_1, content5_1_key, style5_1_key, seed, content_masks, style_masks))))

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_layer_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.decoder_layer_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256 + 256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self, cs, c_adain_3_feat):
        cs = self.decoder_layer_1(cs)
        cs = self.decoder_layer_2(torch.cat((cs, c_adain_3_feat), dim=1))
        return cs

def setup_args(parser):
    parser.add_argument(
        "--content_path", type=str, required=True,
        help="Path to a single content img",
    )
    parser.add_argument(
        "--style_path", type=str, required=True,
        help="Path to a single style img",
    )
    parser.add_argument(
        "--output_dir", type=str, default='output/',
        help="Output path",
    )
    parser.add_argument(
        "--resize", action='store_true',
        help="Whether resize images to the 512 scale, which is the training resolution "
             "of the model and may yield better performance"
    )
    parser.add_argument(
        "--keep_ratio", action='store_true',
        help="Whether keep the aspect ratio of original images while resizing"
    )

def main(args):
    """ Argument """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(args)

    if args.content_path.endswith('.mp4'):
        handle_video(args)
    else:
        handle_image(args)


def handle_video(args):

    print("Starting video style transfer...")

    content_name = os.path.basename(args.content_path)
    style_name = os.path.basename(args.style_path)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cap = cv2.VideoCapture(args.content_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if args.resize:
        resized_video_path, original_size = resize_video(cap, 512, args.keep_ratio)
        cap = cv2.VideoCapture(resized_video_path)
        video_path = resized_video_path
    else:
        original_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_path = args.content_path

    style_im = cv2.imread(args.style_path)
    if args.resize:
        style_im = resize(style_im, 512, args.keep_ratio)

    # Get video dimensions
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print("Error reading video")
        return

    h, w = first_frame.shape[:2]
    h_s, w_s = style_im.shape[:2]

    # Prepare frames for SAM2
    content_frames_dir = prepare_content_frames(video_path)
    print(f'Frames extracted to: {content_frames_dir}')

    """ Building Models """
    transformer_path = 'ckpt/latest_net_transformer.pth'
    decoder_path = 'ckpt/latest_net_decoder.pth'
    ada_attn_3_path = 'ckpt/latest_net_adaattn_3.pth'
    vgg_path = 'ckpt/vgg_normalised.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build encoder
    image_encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU()  # relu5-4
    )

    image_encoder.load_state_dict(torch.load(vgg_path))
    enc_layers = list(image_encoder.children())
    enc_1 = nn.Sequential(*enc_layers[:4]).to(device)
    enc_2 = nn.Sequential(*enc_layers[4:11]).to(device)
    enc_3 = nn.Sequential(*enc_layers[11:18]).to(device)
    enc_4 = nn.Sequential(*enc_layers[18:31]).to(device)
    enc_5 = nn.Sequential(*enc_layers[31:44]).to(device)
    image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]

    for layer in image_encoder_layers:
        layer.eval()
        for p in layer.parameters():
            p.requires_grad = False

    # Load style transfer models
    transformer = Transformer(in_planes=512, key_planes=512 + 256 + 128 + 64).to(device)
    decoder = Decoder().to(device)
    ada_attn_3 = AdaAttN(in_planes=256, key_planes=256 + 128 + 64, max_sample=64 * 64).to(device)

    transformer.load_state_dict(torch.load(transformer_path))
    decoder.load_state_dict(torch.load(decoder_path))
    ada_attn_3.load_state_dict(torch.load(ada_attn_3_path))

    transformer.eval()
    decoder.eval()
    ada_attn_3.eval()

    for p in transformer.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False
    for p in ada_attn_3.parameters():
        p.requires_grad = False

    print("\n=== Model Weight Diagnostics ===")
    transformer_params = list(transformer.parameters())
    decoder_params = list(decoder.parameters())
    ada_attn_3_params = list(ada_attn_3.parameters())
    print(f"Transformer first layer weight sample: {transformer_params[0].flatten()[:5]}")
    print(f"Decoder first layer weight sample: {decoder_params[0].flatten()[:5]}")
    print(f"Ada_attn_3 first layer weight sample: {ada_attn_3_params[0].flatten()[:5]}")
    print(f"training mode: {transformer.training}, {decoder.training}, {ada_attn_3.training}")
    print(f"Transformer has NaN: {any(torch.isnan(p).any() for p in transformer_params)}")
    print(f"Decoder has NaN: {any(torch.isnan(p).any() for p in decoder_params)}")

    def encode_with_intermediate(img):
        results = [img]
        for i in range(len(image_encoder_layers)):
            func = image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    """ SAM2 Setup """
    CHECKPOINT = './checkpoints/sam2.1_hiera_small.pt'
    CONFIG = 'configs/sam2.1/sam2.1_hiera_s.yaml'
    sam2_c = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=device)
    sam2_s = SAM2ImagePredictor(build_sam2(CONFIG, CHECKPOINT, device=device))

    frame_paths = sorted([f for f in content_frames_dir.iterdir() if f.suffix.upper() == '.JPEG'])

    # Create reversed frames directory
    reversed_dir = content_frames_dir.parent / 'reversed_frames'
    shutil.rmtree(reversed_dir, ignore_errors=True)
    reversed_dir.mkdir()
    for new_idx, frame_path in enumerate(reversed(frame_paths)):
        dst = reversed_dir / f'{new_idx:05d}.JPEG'
        shutil.copy(frame_path, dst)

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        sam2_s.set_image(style_im)
        sam2_c.eval()
        inference_state = sam2_c.init_state(video_path=str(content_frames_dir))
        back_state = sam2_c.init_state(video_path=str(reversed_dir))
    print('SAM2 loaded successfully')

    def mouse_handle_sam(event, x, y, _1, _2):
        nonlocal predictor, frame_index, vis, vis_, is_working, is_dragging, cur_points_labels, \
            cur_box, mask, window_name, color, is_valid, content_obj_id, inference_state, back_state

        if (not is_working) and (event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN or (
                is_dragging and event == cv2.EVENT_MBUTTONUP)):
            is_working = True
            if event == cv2.EVENT_LBUTTONDOWN:
                cur_points_labels[0].append([x, y])
                cur_points_labels[1].append(1)
                cv2.circle(vis, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
            elif event == cv2.EVENT_RBUTTONDOWN:
                cur_points_labels[0].append([x, y])
                cur_points_labels[1].append(0)
                cv2.circle(vis, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            elif event == cv2.EVENT_MBUTTONUP:
                cur_box += [x, y]
                if cur_box[0] > cur_box[2]:
                    cur_box[2], cur_box[0] = cur_box[0], x
                if cur_box[1] > cur_box[3]:
                    cur_box[3], cur_box[1] = cur_box[1], y
                is_dragging = False

            if cur_points_labels[0]:
                # Add to forward state
                _, object_ids, mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_index,
                    obj_id=content_obj_id,
                    points=np.array(cur_points_labels[0]),
                    labels=np.array(cur_points_labels[1]),
                    clear_old_points=False
                )

                # Map frame_index to reversed frame index for back_state
                reversed_frame_idx = len(frame_paths) - 1 - frame_index
                predictor.add_new_points_or_box(
                    inference_state=back_state,
                    frame_idx=reversed_frame_idx,
                    obj_id=content_obj_id,
                    points=np.array(cur_points_labels[0]),
                    labels=np.array(cur_points_labels[1]),
                    clear_old_points=False
                )

                print(f"Added object IDs: {object_ids}")
                content_obj_id += 1
                if not is_valid:
                    is_valid = True
                final_mask = (mask_logits > 0.0).cpu().numpy()
                if len(final_mask.shape) > 2:
                    final_mask = np.any(final_mask, axis=0)

                vis_ = vis.copy()
                show_masks(vis_, [final_mask])
                show_points(vis_, cur_points_labels[0], cur_points_labels[1])
                cv2.imshow(window_name, vis_)
                is_working = False

        elif not is_dragging and event == cv2.EVENT_MBUTTONDOWN:
            is_dragging = True
            cur_box = [x, y]
        elif is_dragging and event == cv2.EVENT_MOUSEMOVE:
            vis__ = vis_.copy()
            cv2.rectangle(vis__, (cur_box[0], cur_box[1]), (x, y), (255, 0, 0))
            cv2.imshow(window_name, vis__)

    def mouse_handle_circle(event, x, y, _1, _2):
        nonlocal is_dragging, point_a, x_max, x_min, y_max, y_min, vis_, window_name, mask_style, is_valid_style
        if is_dragging and event == cv2.EVENT_LBUTTONUP:
            is_dragging = False
            cv2.line(vis_, point_a, (x, y), (255, 255, 255), 2)
            cv2.line(mask_style, point_a, (x, y), (255, 255, 255), 2)
            point_a = (x, y)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
            cv2.imshow(window_name, vis_)
        elif is_dragging and event == cv2.EVENT_MOUSEMOVE:
            cv2.line(vis_, point_a, (x, y), (255, 255, 255), 2)
            cv2.line(mask_style, point_a, (x, y), (255, 255, 255), 2)
            point_a = (x, y)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
            cv2.imshow(window_name, vis_)
        elif not is_dragging and event == cv2.EVENT_LBUTTONDOWN:
            is_dragging = True
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
            point_a = (x, y)
            is_valid_style = True

    def create_temp_video_from_frames(frame_dir, output_path, fps, frame_size):
        """Helper to create a video from frame directory"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        frame_paths = sorted([f for f in frame_dir.iterdir() if f.suffix.upper() == '.JPEG'])
        for fp in frame_paths:
            frame = cv2.imread(str(fp))
            if frame is not None:
                writer.write(frame)
        writer.release()
        return output_path

    # Store all iterations of selections
    all_content_masks = []  # List of list of masks per iteration
    all_style_masks = []  # List of list of masks per iteration

    iteration = 0
    continue_selecting = True
    current_video_path = video_path  # Track which video to use for selection

    while continue_selecting:
        print(f"\n=== Iteration {iteration + 1} ===")

        # Initialize variables for user interaction
        content_obj_id = iteration  # Use iteration number as obj_id

        # FIXED: Use the current video (original or previously stylized)
        print(f'Press "p" to pause the video and select objects...')
        frame_index, frame_image = play_video_until_paused(current_video_path)

        if frame_index is None:
            print("No frame selected")
            break

        # FIXED: Reset SAM2 states for each iteration to avoid dtype issues
        if iteration > 0:
            # Re-initialize SAM2 states to clear previous memory
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                inference_state = sam2_c.init_state(video_path=str(content_frames_dir))
                back_state = sam2_c.init_state(video_path=str(reversed_dir))

        # Content selection with SAM2
        content_vis = frame_image.copy()
        cv2.namedWindow('Content Image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Content Image', content_vis)
        cv2.waitKey(1000)

        is_working = False
        is_dragging = False
        is_valid = False
        cur_points_labels = ([], [])
        cur_box = []
        vis = frame_image
        vis_ = vis.copy()
        predictor = sam2_c
        color = (np.random.random(3) * 255).reshape(1, 1, 3)
        mask = np.zeros((h, w, 3)).astype(np.uint8)
        window_name = 'Content Image'

        print('\\t\\tLeft Click on the Content Image to Set a Foreground Point;')
        print('\\t\\tRight Click on the Content Image to Set a Background Point;')
        print('\\t\\tPress Any Key to Finish Your Current Selection')
        cv2.setMouseCallback('Content Image', mouse_handle_sam)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if not is_valid:
            print("No valid content selection made. Stopping iterations.")
            break

        # Style selection with hand-drawn contour (unchanged)
        style_vis = style_im.copy()
        cv2.namedWindow('Style Image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Style Image', style_vis)
        cv2.waitKey(1000)

        is_dragging = False
        is_valid_style = False
        mask_style = np.zeros((h_s, w_s, 3)).astype(np.uint8)
        vis_ = style_vis.copy()
        x_max, x_min, y_max, y_min = 0, 1e10, 0, 1e10
        point_a = (0, 0)
        window_name = 'Style Image'

        print('\\t\\tDrag Your Mouse to Draw a Contour on Style Image;')
        print('\\t\\tPress Any Key to Finish Your Drawing')
        cv2.setMouseCallback('Style Image', mouse_handle_circle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        current_style_masks = []
        if is_valid_style and x_max > x_min and y_max > y_min:
            attempt = 0
            while attempt < 500:
                temp_mask = mask_style.copy()
                seed_x = np.random.randint(int(x_min) + 1, int(x_max))
                seed_y = np.random.randint(int(y_min) + 1, int(y_max))
                cv2.floodFill(temp_mask, np.zeros((h_s + 2, w_s + 2)).astype(np.uint8), (seed_x, seed_y),
                              (255, 255, 255), (50, 50, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
                temp_mask_sum = temp_mask.sum(axis=-1, keepdims=True).astype(bool)
                if temp_mask_sum.sum() <= h_s * w_s / 2 and temp_mask_sum.sum() > 0:
                    mask_style = temp_mask
                    current_style_masks.append(mask_style)
                    break
                attempt += 1

        all_style_masks.append(current_style_masks)

        # Propagate masks through video
        print("Propagating masks through video...")

        # FIXED: Clear CUDA cache before propagation to avoid memory issues
        torch.cuda.empty_cache()

        forward_results = [(idx, logits) for idx, _, logits in sam2_c.propagate_in_video(inference_state)]
        backward_results_raw = [(idx, logits) for idx, _, logits in sam2_c.propagate_in_video(back_state)]
        backward_results = [(len(frame_paths) - 1 - idx, logits) for idx, logits in backward_results_raw]

        all_propagated = forward_results + backward_results
        all_propagated = list({idx: logits for idx, logits in all_propagated}.items())
        all_propagated.sort(key=lambda x: x[0])

        # Store content masks for this iteration
        iteration_content_masks = {}
        for frame_idx, mask_logits in all_propagated:
            if frame_idx >= len(frame_paths):
                continue
            masks = (mask_logits > 0.0).cpu().numpy()
            if len(masks.shape) == 4:
                combined_mask = np.any(masks.reshape(-1, masks.shape[-2], masks.shape[-1]), axis=0)
            elif len(masks.shape) == 3:
                combined_mask = np.any(masks, axis=0)
            else:
                combined_mask = masks
            content_mask = combined_mask.astype(np.uint8) * 255
            content_mask = np.stack([content_mask] * 3, axis=2)
            iteration_content_masks[frame_idx] = content_mask

        all_content_masks.append(iteration_content_masks)

        # Generate file names for this iteration
        if iteration == 0:
            base_name = f"{content_name[:content_name.rfind('.')]}_{style_name[:style_name.rfind('.')]}"
        else:
            base_name = previous_base_name + "_masked"

        output_masked = os.path.join(args.output_dir, base_name + "_masked.mp4")
        output_nomask = os.path.join(args.output_dir, base_name + "_nomask.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer_masked = cv2.VideoWriter(output_masked, fourcc, fps, (w, h))
        video_writer_nomask = cv2.VideoWriter(output_nomask, fourcc, fps, (w, h))

        print(f"Processing video frames for iteration {iteration + 1}...")

        # FIXED: Read from correct source
        if iteration > 0:
            source_cap = cv2.VideoCapture(previous_masked_path)
        else:
            source_cap = None

        frame_count = 0
        for frame_idx in range(len(frame_paths)):
            # Load source frame
            if source_cap is not None:
                ret, current_frame = source_cap.read()
                if not ret:
                    print(f"Warning: Could not read frame {frame_idx} from previous video")
                    break
            else:
                frame_path = frame_paths[frame_idx]
                current_frame = cv2.imread(str(frame_path))

            if current_frame is None:
                continue

            print(f"Processing frame {frame_idx + 1}/{len(frame_paths)}")

            if frame_idx in iteration_content_masks:
                all_mask_c_frame = [iteration_content_masks[frame_idx]]
            else:
                all_mask_c_frame = [np.zeros((h, w, 3), dtype=np.uint8)]

            current_h, current_w = current_frame.shape[:2]

            # Style transfer
            with torch.no_grad():
                style = img_to_tensor(cv2.cvtColor(padding(style_im, 32), cv2.COLOR_BGR2RGB)).to(device)
                content = img_to_tensor(cv2.cvtColor(padding(current_frame, 32), cv2.COLOR_BGR2RGB)).to(device)

                c_masks = [torch.from_numpy(padding(m, 32)).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) / 255.0
                           for m in all_mask_c_frame]
                s_masks = [torch.from_numpy(padding(m, 32)).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) / 255.0
                           for m in current_style_masks] if current_style_masks else []

                c_feats = encode_with_intermediate(content)
                s_feats = encode_with_intermediate(style)

                # WITH masks
                c_adain_feat_3_masked = ada_attn_3(c_feats[2], s_feats[2], get_key(c_feats, 2), get_key(s_feats, 2),
                                                   None,
                                                   c_masks, s_masks)
                cs_masked = transformer(c_feats[3], s_feats[3], c_feats[4], s_feats[4], get_key(c_feats, 3),
                                        get_key(s_feats, 3),
                                        get_key(c_feats, 4), get_key(s_feats, 4), None, c_masks, s_masks)
                cs_masked = decoder(cs_masked, c_adain_feat_3_masked)

                CONTENT_PRESERVATION = 0.3
                cs_masked = CONTENT_PRESERVATION * content + (1 - CONTENT_PRESERVATION) * cs_masked

                result_masked = tensor_to_img(cs_masked[:, :, :current_h, :current_w])
                result_masked = cv2.cvtColor(result_masked, cv2.COLOR_RGB2BGR)
                video_writer_masked.write(result_masked)

                # WITHOUT masks (keep original frame)
                c_adain_feat_3_nomask = ada_attn_3(c_feats[2], s_feats[2], get_key(c_feats, 2), get_key(s_feats, 2),
                                                   None,
                                                   [], [])
                cs_nomask = transformer(c_feats[3], s_feats[3], c_feats[4], s_feats[4], get_key(c_feats, 3),
                                        get_key(s_feats, 3),
                                        get_key(c_feats, 4), get_key(s_feats, 4), None, [], [])
                cs_nomask = decoder(cs_nomask, c_adain_feat_3_nomask)

                cs_nomask = CONTENT_PRESERVATION * content + (1 - CONTENT_PRESERVATION) * cs_nomask

                result_nomask = tensor_to_img(cs_nomask[:, :, :current_h, :current_w])
                result_nomask = cv2.cvtColor(result_nomask, cv2.COLOR_RGB2BGR)
                video_writer_nomask.write(result_nomask)

                frame_count += 1

        if source_cap is not None:
            source_cap.release()
        video_writer_masked.release()
        video_writer_nomask.release()

        print(f"\\nIteration {iteration + 1} completed:")
        print(f"  Masked: {output_masked}")
        print(f"  No-mask: {output_nomask}")
        print(f"  Processed {frame_count} frames")

        # FIXED: Update current_video_path to use the newly created masked video
        previous_base_name = base_name
        previous_masked_path = output_masked
        current_video_path = output_masked  # Use this for next iteration's pause selection

        # Ask user if they want to continue
        print("\\nDo you want to add another selection? (y/n): ")
        user_input = input().strip().lower()
        if user_input != 'y':
            continue_selecting = False

        iteration += 1

    print(f"\\nAll {iteration + 1} iteration(s) completed!")


def handle_image(args):
    content_name = os.path.basename(args.content_path)
    style_name = os.path.basename(args.style_path)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, content_name[:content_name.rfind('.')] +
                               '_' + style_name[:style_name.rfind('.')] + '.jpg')

    content_im = cv2.imread(args.content_path)
    style_im = cv2.imread(args.style_path)
    original_h, original_w = content_im.shape[:2]
    if args.resize:
        content_im = resize(content_im, 512, args.keep_ratio)
        style_im = resize(style_im, 512, args.keep_ratio)
    h, w = content_im.shape[:2]
    h_s, w_s = style_im.shape[:2]

    """ Building Models """
    transformer_path = 'ckpt/latest_net_transformer.pth'
    decoder_path = 'ckpt/latest_net_decoder.pth'
    ada_attn_3_path = 'ckpt/latest_net_adaattn_3.pth'
    vgg_path = 'ckpt/vgg_normalised.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU()  # relu5-4
    )
    image_encoder.load_state_dict(torch.load(vgg_path))
    enc_layers = list(image_encoder.children())
    enc_1 = nn.Sequential(*enc_layers[:4]).to(device)
    enc_2 = nn.Sequential(*enc_layers[4:11]).to(device)
    enc_3 = nn.Sequential(*enc_layers[11:18]).to(device)
    enc_4 = nn.Sequential(*enc_layers[18:31]).to(device)
    enc_5 = nn.Sequential(*enc_layers[31:44]).to(device)
    image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]
    for layer in image_encoder_layers:
        layer.eval()
        for p in layer.parameters():
            p.requires_grad = False
    transformer = Transformer(in_planes=512, key_planes=512 + 256 + 128 + 64).to(device)
    decoder = Decoder().to(device)
    ada_attn_3 = AdaAttN(in_planes=256, key_planes=256 + 128 + 64, max_sample=64 * 64).to(device)
    transformer.load_state_dict(torch.load(transformer_path))
    decoder.load_state_dict(torch.load(decoder_path))
    ada_attn_3.load_state_dict(torch.load(ada_attn_3_path))
    transformer.eval()
    decoder.eval()
    ada_attn_3.eval()
    for p in transformer.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False
    for p in ada_attn_3.parameters():
        p.requires_grad = False

    def encode_with_intermediate(img):
        results = [img]
        for i in range(len(image_encoder_layers)):
            func = image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    def style_transfer():
        with torch.no_grad():
            style = img_to_tensor(cv2.cvtColor(padding(style_im, 32), cv2.COLOR_BGR2RGB)).to(device)
            content = img_to_tensor(cv2.cvtColor(padding(content_im, 32), cv2.COLOR_BGR2RGB)).to(device)
            c_masks = [torch.from_numpy(padding(m, 32)).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
                       for m in all_mask_c]
            s_masks = [torch.from_numpy(padding(m, 32)).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
                       for m in all_mask_s]
            c_feats = encode_with_intermediate(content)
            s_feats = encode_with_intermediate(style)
            c_adain_feat_3 = ada_attn_3(c_feats[2], s_feats[2], get_key(c_feats, 2), get_key(s_feats, 2), None,
                                        c_masks, s_masks)
            cs = transformer(c_feats[3], s_feats[3], c_feats[4], s_feats[4], get_key(c_feats, 3), get_key(s_feats, 3),
                             get_key(c_feats, 4), get_key(s_feats, 4), None, c_masks, s_masks)
            cs = decoder(cs, c_adain_feat_3)
            cs = tensor_to_img(cs[:, :, :h, :w])
            cs = cv2.cvtColor(cs, cv2.COLOR_RGB2BGR)
            return cs

    """ Interaction """
    sam_checkpoint = "segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor_c = SamPredictor(sam)
    predictor_s = SamPredictor(sam)
    predictor_c.set_image(cv2.cvtColor(content_im, cv2.COLOR_BGR2RGB))
    predictor_s.set_image(cv2.cvtColor(style_im, cv2.COLOR_BGR2RGB))
    all_vis_c = [content_im.copy()]
    all_vis_s = [style_im.copy()]
    all_mask_c = []
    all_mask_s = []

    content_vis = content_im.copy()
    style_vis = style_im.copy()

    cv2.namedWindow('Content Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Content Image', content_vis)
    cv2.waitKey(1000)
    cv2.namedWindow('Style Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Style Image', style_vis)
    cv2.waitKey(1000)
    result = style_transfer()
    cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Result", result)
    cv2.waitKey(1000)

    def mouse_handle_sam(event, x, y, _1, _2):
        nonlocal predictor, vis, vis_, is_working, is_dragging, cur_points_labels, \
            cur_box, mask, window_name, color, is_valid

        if (not is_working) and (event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN or (
                is_dragging and event == cv2.EVENT_MBUTTONUP)):
            is_working = True
            if event == cv2.EVENT_LBUTTONDOWN:
                cur_points_labels[0].append([x, y])
                cur_points_labels[1].append(1)
                cv2.circle(vis, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
                cv2.imshow(window_name, vis)
            elif event == cv2.EVENT_RBUTTONDOWN:
                cur_points_labels[0].append([x, y])
                cur_points_labels[1].append(0)
                cv2.circle(vis, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.imshow(window_name, vis)
            elif event == cv2.EVENT_MBUTTONUP:
                cur_box += [x, y]
                if cur_box[0] > cur_box[2]:
                    cur_box[2] = cur_box[0]
                    cur_box[0] = x
                if cur_box[1] > cur_box[3]:
                    cur_box[3] = cur_box[1]
                    cur_box[1] = y
                is_dragging = False
                cv2.imshow(window_name, vis)
            mask, _, _ = predictor.predict(
                point_coords=np.array(cur_points_labels[0]) if len(cur_points_labels[0]) > 0 else None,
                point_labels=np.array(cur_points_labels[1]) if len(cur_points_labels[1]) > 0 else None,
                box=np.array(cur_box) if len(cur_box) == 4 else None,
                multimask_output=False
            )
            if not is_valid:
                is_valid = True

            mask = mask.reshape((mask.shape[1], mask.shape[2], 1))
            vis_ = (mask.astype(np.float64) * (color * 0.6 + vis.astype(np.float64) * 0.4) +
                    (1 - mask.astype(np.float64)) * vis.astype(np.float64)).astype(np.uint8)
            cv2.imshow(window_name, vis_)
            is_working = False
        elif not is_dragging and event == cv2.EVENT_MBUTTONDOWN:
            is_dragging = True
            cur_box = [x, y]
        elif is_dragging and event == cv2.EVENT_MOUSEMOVE:
            vis__ = vis_.copy()
            cv2.rectangle(vis__, (cur_box[0], cur_box[1]), (x, y), (255, 0, 0))
            cv2.imshow(window_name, vis__)

    def mouse_handle_circle(event, x, y, _1, _2):
        nonlocal is_dragging, point_a, x_max, x_min, y_max, y_min, vis_, window_name, mask, is_valid
        if is_dragging and event == cv2.EVENT_LBUTTONUP:
            is_dragging = False
            cv2.line(vis_, point_a, (x, y), (255, 255, 255))
            cv2.line(mask, point_a, (x, y), (255, 255, 255))
            point_a = (x, y)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
            cv2.imshow(window_name, vis_)
        elif is_dragging and event == cv2.EVENT_MOUSEMOVE:
            cv2.line(vis_, point_a, (x, y), (255, 255, 255))
            cv2.line(mask, point_a, (x, y), (255, 255, 255))
            point_a = (x, y)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
            cv2.imshow(window_name, vis_)
        elif not is_dragging and event == cv2.EVENT_LBUTTONDOWN:
            is_dragging = True
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
            point_a = (x, y)

    while True:
        print('Please Choose an Option for Content Image:')
        print('\t1: Select an Area by SAM')
        print('\t2: Specify an Area by a Contour')
        print('\t3: Undo Previous Content & Style Selection')
        print('\tOther: Finish!')
        option = input()
        if option == '1' or option == '2':
            is_working = False
            is_dragging = False
            is_valid = False
            cur_points_labels = ([], [])
            cur_box = []
            vis = content_vis
            vis_ = vis.copy()
            predictor = predictor_c
            color = (np.random.random(3) * 255).reshape(1, 1, 3)
            mask = np.zeros((h, w, 3)).astype(np.uint8)
            window_name = 'Content Image'

            if option == '1':
                print('\t\tLeft Click on the Content Image to Set a Foreground Point;')
                print('\t\tRight Click on the Content Image to Set a Background Point;')
                print('\t\tMiddle Click on the Content Image and Drag Your Mouse to Specify a Bounding Box;')
                print('\t\tPress Any Key to Finish Your Current Selection')
                cv2.setMouseCallback('Content Image', mouse_handle_sam)
                cv2.waitKey(0)
            else:
                print('\t\tDrag Your Mouse to Draw a Contour;')
                print('\t\tPress Any Key to Finish Your Current Drawing')
                x_max = 0
                x_min = 1e10
                y_max = 0
                y_min = 1e10
                point_a = (0, 0)
                cv2.setMouseCallback('Content Image', mouse_handle_circle)
                cv2.waitKey(0)
                attempt = 0
                while True:
                    temp_mask = mask.copy()
                    seed_x = np.random.randint(x_min + 1, x_max)
                    seed_y = np.random.randint(y_min + 1, y_max)
                    cv2.floodFill(temp_mask, np.zeros((h + 2, w + 2)).astype(np.uint8), (seed_x, seed_y),
                                  (255, 255, 255), (50, 50, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
                    temp_mask = temp_mask.sum(axis=-1, keepdims=True).astype(np.bool)
                    if temp_mask.sum() <= h * w / 2:
                        is_valid = True
                        break
                    attempt += 1
                    if attempt > 500:
                        is_valid = False
                if is_valid:
                    mask = temp_mask
                    vis_ = (mask.astype(np.float64) * color +
                            (1 - mask.astype(np.float64)) * vis.astype(np.float64)).astype(np.uint8)
                    cv2.imshow('Content Image', vis_)
                    cv2.waitKey(1000)
            if not is_valid:
                print('\t\tInvalid Selection! Please Re-try:')
            else:
                all_vis_c.append(vis_.copy())
                all_mask_c.append(mask)
                content_vis = vis_

                print('Please Choose an Option for Style Image:')
                while True:
                    print('\t1: Select an Area by SAM')
                    print('\t2: Specify an Area by a Contour')
                    print('\tOther: Undo Previous Content Selection')
                    option_s = input()

                    is_working = False
                    is_dragging = False
                    is_valid = False
                    cur_points_labels = ([], [])
                    cur_box = []
                    vis = style_vis
                    vis_ = vis.copy()
                    predictor = predictor_s
                    mask = np.zeros((h_s, w_s, 3)).astype(np.uint8)
                    window_name = 'Style Image'

                    if option_s == '1':
                        print('\t\tLeft Click on the Style Image to Set a Foreground Point;')
                        print('\t\tRight Click on the Style Image to Set a Background Point;')
                        print('\t\tMiddle Click on the Style Image and Drag Your Mouse to Specify a Bounding Box;')
                        print('\t\tPress Any Key to Finish Your Current Selection')
                        cv2.setMouseCallback('Style Image', mouse_handle_sam, param='style')
                        cv2.waitKey(0)
                    elif option_s == '2':
                        print('\t\tDrag Your Mouse to Draw a Contour;')
                        print('\t\tPress Any Key to Finish Your Current Drawing')
                        x_max = 0
                        x_min = 1e10
                        y_max = 0
                        y_min = 1e10
                        point_a = (0, 0)
                        cv2.setMouseCallback('Style Image', mouse_handle_circle)
                        cv2.waitKey(0)
                        attempt = 0
                        while True:
                            temp_mask = mask.copy()
                            seed_x = np.random.randint(x_min + 1, x_max)
                            seed_y = np.random.randint(y_min + 1, y_max)
                            cv2.floodFill(temp_mask, np.zeros((h_s + 2, w_s + 2)).astype(np.uint8), (seed_x, seed_y),
                                          (255, 255, 255), (50, 50, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
                            temp_mask = temp_mask.sum(axis=-1, keepdims=True).astype(np.bool)
                            if temp_mask.sum() <= h_s * w_s / 2:
                                is_valid = True
                                break
                            attempt += 1
                            if attempt > 500:
                                is_valid = False
                        if is_valid:
                            mask = temp_mask
                            vis_ = (mask.astype(np.float64) * color +
                                    (1 - mask.astype(np.float64)) * vis.astype(np.float64)).astype(np.uint8)
                            cv2.imshow('Style Image', vis_)
                            cv2.waitKey(1000)
                    else:
                        all_vis_c.pop()
                        all_mask_c.pop()
                        content_vis = all_vis_c[-1]
                        cv2.imshow("Content Image", content_vis)
                        cv2.waitKey(1000)
                        break
                    if is_valid:
                        all_vis_s.append(vis_.copy())
                        all_mask_s.append(mask)
                        style_vis = vis_
                        result = style_transfer()
                        cv2.imshow("Result", result)
                        cv2.waitKey(1000)
                        break
                    else:
                        print('\t\tInvalid Selection! Please Re-try:')
        elif option == '3':
            if len(all_vis_c) == 0 or len(all_mask_c) == 0:
                print('\t\tNo Previous Selection! Please Re-enter:')
            else:
                all_vis_c.pop()
                all_mask_c.pop()
                all_vis_s.pop()
                all_mask_s.pop()
                content_vis = all_vis_c[-1]
                style_vis = all_vis_s[-1]
                cv2.imshow("Content Image", content_vis)
                cv2.waitKey(1000)
                cv2.imshow("Style Image", style_vis)
                cv2.waitKey(1000)
                result = style_transfer()
                cv2.imshow("Result", result)
                cv2.waitKey(1000)
        else:
            break

    if args.resize:
        result = cv2.resize(result, (original_w, original_h))
    cv2.destroyAllWindows()
    cv2.imwrite(output_path, result)

if __name__ == '__main__':
    main(sys.argv[1:])
