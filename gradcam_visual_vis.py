import os
import random
import time
import argparse
import numpy as np
from models.gradcam_test import YOLOV5GradCAM
# from models.yolo_v5_object_detector import YOLOV5TorchObjectDetector
from models.yolo_v5_object_detector_test import YOLOV5TorchObjectDetector
import cv2
from deep_utils import Box, split_extension
# 数据集类别名
# names = ['person']
names = ['person', 'car', 'bicycle']

# target = ['model_30_cv1_act', 'model_30_cv2_act', 'model_30_cv3_act', \
#           'model_33_cv1_act', 'model_33_cv2_act', 'model_33_cv3_act', \
#           'model_36_cv1_act', 'model_36_cv2_act', 'model_36_cv3_act']
# target = ['model_30_cv3_act']
target = ['model_30_cv3_act', 'model_33_cv3_act', 'model_36_cv3_act']

# Arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--model-path', type=str, default="/home/shen/Chenyf/exp_save/multispectral-object-detection/5m_NiNfusion/weights/best.pt", help='Path to the model')
# parser.add_argument('--model-path', type=str, default="ICAFusion_kaist.pt", help='Path to the model')
# parser.add_argument('--model-path', type=str, default="ICAFusion_FLIR.pt", help='Path to the model')
parser.add_argument('--model-path', type=str, default="MOD_FLIR.pt", help='Path to the model')

# parser.add_argument('--source1', type=str, default='/home/shen/Chenyf/kaist/visible/test', help='source')  # file/folder, 0 for webcam
# parser.add_argument('--source2', type=str, default='/home/shen/Chenyf/kaist/infrared/test', help='source')  # file/folder, 0 for webcam

parser.add_argument('--source1', type=str, default='F:/code/dataset/kaist/visible', help='source')  # file/folder, 0 for webcam
parser.add_argument('--source2', type=str, default='F:/code/dataset/kaist/infrared', help='source')  # file/folder, 0 for webcam

# parser.add_argument('--output-dir', type=str, default='/home/shen/Chenyf/kaist/Grad_CAM_visual/outputs_nin_head', help='output dir')
parser.add_argument('--output-dir', type=str, default='F:/code/dataset/kaist', help='output dir')
parser.add_argument('--img-size', type=int, default=640, help="input image size")
parser.add_argument('--target-layer', type=str, default=target,
                    help='The layer hierarchical address to which gradcam will applied,'
                         ' the names should be separated by underline')
parser.add_argument('--method', type=str, default='gradcam', help='gradcam or gradcampp')
parser.add_argument('--model-name', type=str, default='ours', help='ours or baseline')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
# parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
parser.add_argument('--names', type=str, default=names,
                    help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')
#'person, car, bicycle'
args = parser.parse_args()


def get_res_img2(heat, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    n_heatmat = (heatmap / 255).astype(np.float32)
    heat.append(n_heatmat)
    return res_img, heat


def get_res_img(bbox, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    #n_heatmat = (Box.fill_outer_box(heatmap, bbox, value=0) / 255).astype(np.float32)
    n_heatmat = (heatmap / 255).astype(np.float32)
    res_img = cv2.addWeighted(res_img, 0.7, n_heatmat, 0.3, 0)
    res_img = (res_img / res_img.max())
    return res_img, n_heatmat

def get_res_img_test(bbox, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
        np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    n_heatmat = (heatmap / 255).astype(np.float32)
    res_img = res_img / 255
    res_img = cv2.add(res_img, n_heatmat)
    res_img = (res_img / res_img.max())
    return res_img, n_heatmat

def put_text_box(bbox, cls_name, res_img):
    x1, y1, x2, y2 = bbox
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    #cv2.imwrite('temp.jpg', (res_img * 255).astype(np.uint8))
    #res_img = cv2.imread('temp.jpg')
    res_img = Box.put_box(res_img, bbox)
    #res_img = Box.put_text(res_img, cls_name, (x1 - 3, y1))
    # res_img = Box.put_text(res_img, str(round((float(cls_name)+0.40), 2)), (x1-3, y1))
    return res_img

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    cv2.imwrite('temp.jpg', (img * 255).astype(np.uint8))
    img = cv2.imread('temp.jpg')

    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
 
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        outside = c1[1] - t_size[1] - 3 >= 0  # label fits outside box up
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3 if outside else c1[1] + t_size[1] + 3
        outsize_right = c2[0] - img.shape[:2][1] > 0  # label fits outside box right
        c1 = c1[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c1[0], c1[1]
        c2 = c2[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c2[0], c2[1]
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2 if outside else c2[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                    lineType=cv2.LINE_AA)
    return img

def concat_images(images):
    w, h = images[0].shape[:2]
    width = w
    height = h * len(images)
    base_img = np.zeros((width, height, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        base_img[:, h * i:h * (i + 1), ...] = img
    return base_img

def main(img_vis_path, img_ir_path):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    device = args.device
    input_size = (args.img_size, args.img_size)
    img_vis, img_ir = cv2.imread(img_vis_path), cv2.imread(img_ir_path)

    # import pdb;
    # pdb.set_trace()

    print('[INFO] Loading the model')
    # 实例化YOLOv5模型，得到检测结果
    model = YOLOV5TorchObjectDetector(args.model_path, device, img_size=input_size,
                                      names=names, confidence=0.3)
    # preprocess the images
    # img[..., ::-1]: BGR --> RGB
    torch_img_vis, torch_img_ir = model.preprocessing(img_vis[..., ::-1], img_ir[..., ::-1])
    tic = time.time()

    if args.method == 'gradcam':
        for target_layer in args.target_layer:
            saliency_method = YOLOV5GradCAM(model=model, layer_name=target_layer, img_size=input_size)
            tic = time.time()
            masks, logits, [boxes, _, class_names, conf] = saliency_method(torch_img_vis, torch_img_ir)
            result = torch_img_vis.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
            result = result[..., ::-1]  # convert to bgr
            
            # 保存设置
            image_name = os.path.basename(img_vis_path)  # 获取图片名
            # save_path = f'{args.output_dir}/{image_name[:-4]}/{args.method}'
            save_path = f'{args.output_dir}/{image_name[:-4]}/{args.model_name}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print(f'[INFO] Saving the final image at {save_path}')
            for i, mask in enumerate(masks):
                # 遍历每张图片中的每个目标
                res_img = result.copy()
                # 获取目标的位置和类别信息
                bbox, cls_name = boxes[0][i], class_names[0][i]
                # 类别+置信分数
                label = f'{cls_name} {conf[0][i]}'

                # import pdb;
                # pdb.set_trace()

                # 获取目标的热力图
                res_img, heat_map = get_res_img_test(bbox, mask, res_img)
                res_img = plot_one_box(bbox, res_img, label=label, color=colors[int(names.index(cls_name))],
                                    line_thickness=3)
            
                # res_img = cv2.resize(res_img, dsize=(img_vis.shape[:-1][::-1]))
                output_path = f'{save_path}/{target_layer}_{i}.jpg'
                cv2.imwrite(output_path, res_img)
                print(f'{target_layer}_{i}.jpg done!!')
    print(f'Total time : {round(time.time() - tic, 4)} s')

# def main(img_vis_path, img_ir_path):
#     device = args.device
#     input_size = (args.img_size, args.img_size)
#     img_vis, img_ir = cv2.imread(img_vis_path), cv2.imread(img_ir_path)
#     print('[INFO] Loading the model')
#     # load model
#     model = YOLOV5TorchObjectDetector(args.model_path, device, img_size=input_size,
#                                       names=None if args.names is None else args.names.strip().split(","), confidence=0.3)
#     # preprocess the images
#     torch_img_vis, torch_img_ir = model.preprocessing(img_vis[..., ::-1], img_ir[..., ::-1])
#     result = torch_img_vis.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
#     result = result[..., ::-1]  # convert to bgr
#     images = []

#     if args.method == 'gradcam':
#         for layer in args.target_layer:
#             saliency_method = YOLOV5GradCAM(model=model, layer_name=layer, img_size=input_size)
#             tic = time.time()
#             masks, logits, [boxes, _, class_names, confs] = saliency_method(torch_img_vis, torch_img_ir)
#             print("total time:", round(time.time() - tic, 4))
#             res_img = result.copy()
#             res_img = res_img / 255
#             heat = []
#             # 遍历每张图片中的每个目标
#             for i, mask in enumerate(masks):
#                 bbox = boxes[0][i]
#                 mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
#                 heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
#                 n_heatmat = (heatmap / 255).astype(np.float32)
#                 heat.append(n_heatmat)
#                 #res_img, heat_map = get_res_img(bbox, mask, res_img)
#                 #res_img = put_text_box(bbox, cls_name, res_img)  # plot the bboxes
#                 #images.append(res_img)

#             if(len(heat) != 0):
#                 heat_all = heat[0]
#                 for h in heat[1:]:
#                     heat_all += h
#                 heat_avg = heat_all / len(heat)
#                 res_img = cv2.addWeighted(res_img, 0.3, heat_avg, 0.7, 0)
#             res_img = (res_img / res_img.max())
#             cv2.imwrite('temp.jpg', (res_img * 255).astype(np.uint8))
#             heat_map = cv2.imread('temp.jpg')
#             # for i, mask in enumerate(masks):
#             #     bbox, cls_name, conf = boxes[0][i], class_names[0][i], str(confs[0][i])
#             #     heat_map = put_text_box(bbox, conf, heat_avg)  # plot the bboxes
#             final_image = heat_map
#             images.append(final_image)
#             # save the images
#             suffix = '-res-' + layer
#             img_name = split_extension(os.path.split(img_vis_path)[-1], suffix=suffix)
#             output_path = f'{args.output_dir}/{img_name}'
#             os.makedirs(args.output_dir, exist_ok=True)
#             print(f'[INFO] Saving the final image at {output_path}')
#             cv2.imwrite(output_path, final_image)

#         img_name = split_extension(os.path.split(img_vis_path)[-1], suffix='_avg')
#         output_path = f'{args.output_dir}/{img_name}'
#         img_all = images[0].astype(np.uint16)
#         for img in images[1:]:
#             img_all += img
#         img_avg = img_all / len(images)
#         cv2.imwrite(output_path, img_avg.astype(np.uint8))


if __name__ == '__main__':
    if os.path.isdir(args.source1):
        # print('1')
        img_vis_list = os.listdir(args.source1)
        img_vis_list.sort()
        # for item in img_vis_list[1127:]:
        for item in img_vis_list:
            img_vis_path = os.path.join(args.source1 ,item)
            # if args.source1 == '/home/shen/Chenyf/FLIR-align-3class/visible/test':
            if args.source1 == 'F:/code/dataset/kaist/visible':
                new_item = item[:-4] + '.jpg'
                img_ir_path = os.path.join(args.source2, new_item)
            else:
                img_ir_path = os.path.join(args.source2, item)

            main(img_vis_path, img_ir_path)
            print(item)
    else:
        main(img_vis_path, img_ir_path)