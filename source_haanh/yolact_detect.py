import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
import os
from pathlib import Path
import sys
import csv

# Patch để tránh lỗi pyximport khi import
import warnings
warnings.filterwarnings('ignore')

# Import từ YOLACT
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess
from data import cfg, set_cfg
from utils.functions import SavePath



def generate_masks(image_folder, model_path, output_folder, 
                   config='yolact_base_config', 
                   score_threshold=0.15,
                   top_k=100,
                   cuda=True):
    """
    Tạo mask cho tất cả ảnh trong folder
    
    Args:
        image_folder: Đường dẫn đến folder chứa ảnh RGB
        model_path: Đường dẫn đến model weights (.pth)
        output_folder: Đường dẫn để lưu masks
        config: Tên config của model
        score_threshold: Ngưỡng confidence tối thiểu
        top_k: Số lượng detections tối đa
        cuda: Sử dụng GPU hay không
    """
    
    # Tạo output folder nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Thiết lập config
    set_cfg(config)
    
    # Set thêm các config cần thiết
    cfg.mask_proto_debug = False
    cfg.eval_mask_branch = True  # Để tạo masks
    
    # Thiết lập device
    if cuda:
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    
    # Load model
    print('Loading model...', end='')
    net = Yolact()
    net.load_weights(model_path)
    net.eval()
    
    # Bắt buộc dùng Fast NMS để tránh lỗi pyximport
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    
    print(' Done.')
    
    if cuda:
        net = net.cuda()
    
    # Tìm tất cả ảnh trong folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(image_folder).glob(ext))
        image_paths.extend(Path(image_folder).glob(ext.upper()))
    
    print(f'\nFound {len(image_paths)} images')

    mask_output_dir = os.path.join(output_folder, "masks")
    os.makedirs(mask_output_dir, exist_ok=True)
    
    # Process từng ảnh
    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            img_path = str(img_path)
            img_name = os.path.basename(img_path)
            name_without_ext = '.'.join(img_name.split('.')[:-1])
            
            print(f'Processing [{idx+1}/{len(image_paths)}]: {img_name}')
            
            # Đọc ảnh
            frame = cv2.imread(img_path)
            if frame is None:
                print(f'  Error: Cannot read image {img_path}')
                continue
            
            # Chuyển sang tensor
            frame_tensor = torch.from_numpy(frame).cuda().float() if cuda else torch.from_numpy(frame).float()
            batch = FastBaseTransform()(frame_tensor.unsqueeze(0))
            
            # Inference
            preds = net(batch)
            
            # Postprocess để lấy masks
            h, w = frame.shape[:2]
            classes, scores, boxes, masks = postprocess(
                preds, w, h, 
                crop_masks=True,
                score_threshold=score_threshold
            )
            
            # Lọc theo top_k và score_threshold
            if classes.size(0) > 0:
                idx_keep = scores.argsort(0, descending=True)[:top_k]
                classes = classes[idx_keep]
                scores = scores[idx_keep]
                boxes = boxes[idx_keep]
                masks = masks[idx_keep]

                for i, (c, s, b, m) in enumerate(zip(classes, scores, boxes, masks)):
                    mask_filename = f"{os.path.splitext(img_name)[0]}_{i}_class{int(c)}.npy"
                    mask_path = os.path.join(mask_output_dir, mask_filename)
                    np.save(mask_path, m.cpu().numpy())


                
                # Chuyển masks về numpy
                masks_np = masks.cpu().numpy()
                
                # Lưu từng mask
                # for i in range(masks_np.shape[0]):
                #     if scores[i] >= score_threshold:
                #         mask = (masks_np[i] * 255).astype(np.uint8)
                #         mask_filename = f'{name_without_ext}_mask_{i}_class_{classes[i].item()}_score_{scores[i].item():.2f}.png'
                #         mask_path = os.path.join(output_folder, mask_filename)
                #         cv2.imwrite(mask_path, mask)
                
                # Lưu combined mask (tất cả masks trên 1 ảnh)
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                for i in range(masks_np.shape[0]):
                    if scores[i] >= score_threshold:
                        combined_mask = np.maximum(combined_mask, (masks_np[i] > 0.5).astype(np.uint8) * 255)

                
                combined_filename = f'{name_without_ext}_combined_mask.png'
                combined_path = os.path.join(output_folder, combined_filename)
                cv2.imwrite(combined_path, combined_mask)
                
                print(f'  Saved {masks_np.shape[0]} masks')
            else:
                print(f'  No detections found')
    
    print('\nDone! All masks saved to:', output_folder)


if __name__ == '__main__':
    # ===== CẤU HÌNH Ở ĐÂY =====
    IMAGE_FOLDER = 'labelcoco/JPEGImages'           # Folder chứa ảnh RGB
    MODEL_PATH = 'yolact_base_75_4000.pth'          # Đường dẫn model weights
    OUTPUT_FOLDER = 'yolact_result'         # Folder lưu masks
    CONFIG = 'yolact_base_config'                   # Config của model
    SCORE_THRESHOLD = 0.15                          # Ngưỡng confidence
    TOP_K = 100                                     # Số detections tối đa
    USE_CUDA = True                                 # Dùng GPU
    # ==========================
    
    generate_masks(
        image_folder=IMAGE_FOLDER,
        model_path=MODEL_PATH,
        output_folder=OUTPUT_FOLDER,
        config=CONFIG,
        score_threshold=SCORE_THRESHOLD,
        top_k=TOP_K,
        cuda=USE_CUDA
    )

