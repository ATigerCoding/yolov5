import argparse
import os
import platform
import sys

from pathlib import Path
import shutil
import torch
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from PIL import ImageEnhance
import cv2
import numpy as np
import os
import time
from PIL import ImageStat
from PIL import Image
from flask import Flask, request
import requests

# 创建一个 Flask 应用
app = Flask(__name__)



def get_data(path1, path2):
    """
        遍历文件夹下所有文件,并将每个文件的数据以datafrma形式加载合并
    :param path: 测试集真实标签的数据路径
    :return: 测试集预测标签的路径
    """
    data = []
    for file in os.listdir(path1):
        if file.endswith(".txt"):
            file_path = os.path.join(path1, file)
            df = pd.read_csv(file_path, header=None, sep=' ', names=['label', 'x', 'y', 'w', 'h'])
            # 取出label值为 1 3 4 的数据合并
            df = df[(df['label'] == 1) | (df['label'] == 3) | (df['label'] == 5)]
            df['file_name'] = file.split('.')[0]
            data.append(df)
    df_real = pd.concat(data, ignore_index=True)
    df_real = df_real[['file_name', 'label']]
    df_real.groupby('file_name').agg({'label': lambda x: str(set(x))}).reset_index()

    df_pred = pd.read_csv(path2,sep=',')
    df_pred = df_pred[['picture_name', 'type_']]
    df_pred['picture_name'] = df_pred['picture_name'].str.replace('.txt', '')
    df_pred.rename(columns={'picture_name': 'file_name'}, inplace=True)
    df_pred['type_'] = df_pred['type_'].map({'detect_xk': 1, 'detect_fhxk': 3, 'detect_dj': 5})
    df_pred.groupby('file_name').agg({'type_': lambda x: str(set(x))}).reset_index()

    # 对两份数据进行join
    df_merge = pd.merge(df_real, df_pred, on='file_name', how='left')
    df_merge.to_csv(os.path.join(os.path.dirname(path2), 'test_merge.csv'), index=False)
def strategy1(param):
    """
        对图像框进行筛选
    :param param:
    :return:
    """

    if (min(abs(param['distance_top'] - param['distance_bottom']) / abs(
            param['distance_left'] - param['distance_right']),
            abs(param['distance_left'] - param['distance_right']) / abs(
                param['distance_top'] - param['distance_bottom'])) > 0.5) and (param['distance_right'] > 220) and (
            abs(param['im_w'] - param['distance_right']) > 220) and (
            param['distance_bottom'] - param['distance_top'] > 200):
        return True
    else:
        return False


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        systemParam=None,
        path_ori=None,
        path_enhance=None,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories  save_dir  输出目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    try:
        for path, im, im0s, vid_cap, s in dataset:
            try:
                im_w, im_h = im0s.shape[1], im0s.shape[0]  # 增加原始图像的信息
                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Inference
                with dt[1]:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred = model(im, augment=augment, visualize=visualize)

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + (
                        '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                # line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                # 转换为绝对坐标
                                x_center, y_center, width, height = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(
                                    -1).tolist()
                                # 构建包含绝对坐标的行
                                # line = (names.get(cls.item()), x_center, y_center, width, height, conf, im_w, im_h) if save_conf else (
                                #     cls, x_center, y_center, width, height)
                                line = (
                                    names.get(cls.item()), x_center, y_center, width, height,
                                    "{:.4f}".format(conf.item() * 100), im_w, im_h) if save_conf else (
                                    cls, x_center, y_center, width, height)
                                with open(f'{txt_path}.txt', 'a') as f:
                                    f.write(('%s ' * len(line)).rstrip() % line + '\n')
                                # with open(f'{txt_path}.txt', 'a') as f:
                                #     formatted_line = ''
                                #     for item in line:
                                #         if isinstance(item, str):
                                #             formatted_line += f'{item} '  # 处理字符串值
                                #         elif True:
                                #             pass
                                #         else:
                                #             formatted_line += f'{item:.0f} '
                                #     f.write(formatted_line.rstrip() + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    # Stream results
                    im0 = annotator.result()
                    if view_img:
                        if platform.system() == 'Linux' and p not in windows:
                            windows.append(p)
                            cv2.namedWindow(str(p),
                                            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img and (1 in det[:, 5] or 3 in det[:, 5] or 5 in det[:, 5] or 7 in det[:, 5]):
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if vid_path[i] != save_path:  # new video
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    vid_writer[i].release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path = str(
                                    Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer[i].write(im0)

                # Print time (inference-only)
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
            except Exception:
                print("数据异常：", p.name)
                xx = os.path.join(save_dir, 'error')
                if os.path.exists(xx):
                    shutil.rmtree(xx)
                else:
                    os.mkdir(xx)
                with open(os.path.join(os.path.join(save_dir, 'error', 'error.txt')), "a") as file:
                    file.write(f'总图像数量: {p.name}\n')

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
        torch.cuda.empty_cache()
        # 进行文件的合并
        merge_file_result(os.path.join(save_dir, 'labels'), os.path.join(save_dir, 'labels2', 'result.txt'))
        import pandas as pd
        if os.path.exists(os.path.join(save_dir, 'labels2', 'result.txt')):
            df = pd.read_csv(os.path.join(save_dir, 'labels2', 'result.txt'), sep=' ', header=None)
            df.columns = ['picture_name', 'type_', 'x_center', 'y_center', 'width', 'height', 'conf', 'im_w', 'im_h']
            df1 = df[(df['type_'] == 'detect_xk') | (df['type_'] == 'detect_fhxk') | (df['type_'] == 'detect_dj')]
            df1 = df1[df1['conf'] > 80]
            list_err2 = [i.split('.')[0] + '.txt' for i in df1['picture_name'].unique()]  # 有故障的图像.txt

            df = df[df["picture_name"].isin(list_err2)]
            df = df[(df['type_'] == 'detect_xk') | (df['type_'] == 'detect_fhxk') | (df['type_'] == 'detect_dj')]
            df = df[df['conf'] > 80]

            if not os.path.exists:
                os.mkdir(os.path.join(save_dir, 'labels2'))

            df['distance_left'] = df['x_center'] - (df['width'] / 2)
            df['distance_right'] = df['x_center'] + (df['width'] / 2)
            df['distance_top'] = df['y_center'] - df['height'] / 2
            df['distance_bottom'] = df['y_center'] + (df['height'] / 2)

            df['distance_left'] = df['distance_left'].astype(int)
            df['distance_right'] = df['distance_right'].astype(int)
            df['distance_top'] = df['distance_top'].astype(int)
            df['distance_bottom'] = df['distance_bottom'].astype(int)

            df = df[
                ['picture_name', 'type_', 'distance_left', 'distance_right', 'distance_top', 'distance_bottom', 'conf',
                 'im_w', 'im_h']]
            if len(df) > 0:
                df['flag'] = df.apply(lambda x: strategy1(x), axis=1)
                df = df[df['flag'] == True]
                df = df.drop('flag', axis=1)

            ##TODO
            if len(df) > 0:
                df.to_csv(os.path.join(save_dir, 'labels2', 'result.csv'), index=False)
            else:
                imitate_data().to_csv(os.path.join(save_dir, 'labels2', 'result.csv'), index=False)

            list_err = [i.split('.')[0] + '.jpg' for i in df['picture_name'].unique()]
            # 将缺陷图像进行转移到指定的文件夹
            detect_dir = os.path.join(save_dir, 'result')
            if os.path.exists(detect_dir):
                shutil.rmtree(detect_dir)
            else:
                os.mkdir(detect_dir)

            for i in list_err:
                shutil.copy(os.path.join(os.getcwd(), save_dir, i), os.path.join(detect_dir, i))

        else:
            imitate_data().to_csv(os.path.join(save_dir, 'labels2', 'result.csv'), index=False)

        systemParam['fileImageDir'] = os.path.join(os.getcwd(), save_dir, 'labels2', 'result.csv')
        print('完成')

        systemParam['resourceFile'] = systemParam['fileImageDir']
        del systemParam['fileImageDir']

        systemParam['status'] = 0
        url_requests(systemParam)

    except Exception as e:
        print("整体异常 ", e)
        if os.path.exists(os.path.join(save_dir, 'labels2', 'result.csv')):
            pass
        else:
            if os.path.exists(os.path.join(save_dir, 'labels2')):
                pass
            else:
                os.mkdir(os.path.join(save_dir, 'labels2'))
            imitate_data().to_csv(os.path.join(save_dir, 'labels2', 'result.csv'), index=None)
        systemParam['fileImageDir'] = os.path.join(os.getcwd(), save_dir, 'labels2', 'result.csv')

        systemParam['resourceFile'] = systemParam['fileImageDir']
        del systemParam['fileImageDir']
        systemParam['status'] = 1
        url_requests(systemParam)


def merge_file_result(folder_path, merged_file_path):
    """

    :param folder_path: 数据原始的目录
    :param merged_file_path: 数据最终结果存储目录
    :return:
    """
    merged_file_path_tmp = os.path.split(merged_file_path)[0]

    # 如果结果目录存在先进行删除处理
    if os.path.exists(merged_file_path_tmp):
        shutil.rmtree(merged_file_path_tmp)
    os.makedirs(merged_file_path_tmp)

    # if not os.path.exists(merged_file_path_tmp):
    #     os.makedirs(merged_file_path_tmp)
    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()
            # 处理每行数据，并在前面添加文件名
            new_lines = []
            for line in lines:
                line = line.strip()  # 移除行尾换行符和空白字符
                if line:  # 跳过空行
                    new_line = f"{filename} {line}" + "\n"
                    new_lines.append(new_line)

            # 将处理后的数据写回到文件
            with open(merged_file_path, "a") as file:
                file.write("".join(new_lines))
    print("结果文件合并完成！")


def parse_opt():
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('--weights', nargs='+', type=str, default=r'C:\Users\ljy\Desktop\model\train-val-test\train\exp\weights\epoch150.pt', help='model path or triton URL')
    # parser.add_argument('--source', type=str, default=r'F:\PythonProject\Job\data_a_b_cut\Dataset-原始\val\images',
    #                     help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--source', type=str, default=r'F:\PythonProject\Job\Dataset1\projectA\images',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/mydata.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
    parser.add_argument(    '--save-conf', action='store_true', default=True, help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')

    parser.add_argument('--systemParam', type=str, help='系统参数')
    parser.add_argument('--path_ori', type=str, help='原始图像路径')
    parser.add_argument('--path_enhance', type=str, help='增强图像路径')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def noise(img_data):
    coutn = 30
    for k in range(0, coutn):
        xi = int(np.random.uniform(0, img_data.shape[1]))
        xj = int(np.random.uniform(0, img_data.shape[0]))
        if img_data.ndim == 2:
            img_data[xj, xi] = 0
        elif img_data.ndim == 3:
            param = 30
            r1 = np.random.random_sample()
            r2 = np.random.random_sample()
            z1 = param * np.cos(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))
            z2 = param * np.sin(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))

            img_data[xj, xi, 0] = img_data[xj, xi, 0] + z1
            img_data[xj, xi, 1] = img_data[xj, xi, 1] + z1 / 2 + z2 / 2
            img_data[xj, xi, 2] = img_data[xj, xi, 2] + z2

            if img_data[xj, xi, 0] < 0:
                img_data[xj, xi, 0] = 0
            elif img_data[xj, xi, 0] > 255:
                img_data[xj, xi, 0] = 255

            if img_data[xj, xi, 1] < 0:
                img_data[xj, xi, 1] = 0
            elif img_data[xj, xi, 1] > 255:
                img_data[xj, xi, 1] = 255

            if img_data[xj, xi, 2] < 0:
                img_data[xj, xi, 2] = 0
            elif img_data[xj, xi, 2] > 255:
                img_data[xj, xi, 2] = 255
    return img_data


def brightness2(im):
    ''' 均值方法 '''
    stat = ImageStat.Stat(im)
    return stat.mean[0]


def process_enhance(folder_path1):
    for root, dirs, files in os.walk(folder_path1):
        last_level = os.path.basename(root)
        pre_level = os.path.dirname(root)
        root_new = os.path.join(pre_level, last_level + "_enhance")
        # root_new = root.replace(r'京天威图像数据\解析', 'jtw2')
        if os.path.exists(root_new):
            shutil.rmtree(root_new)
        os.mkdir(root_new)
        num = 0
        for file in files:
            num = num + 1
            time1 = time.time()
            file_path = os.path.join(root, file)
            print('数量:', num)
            print('开始处理: ', file_path)
            im = Image.open(file_path)
            # PLT方法 计算灰度图像的平均亮度  效率最高
            plt_mean = brightness2(im)
            # 这里的阈值100为判断亮暗图片，可调整
            if plt_mean <= 100:
                # 创建亮度增强对象
                brightness = ImageEnhance.Brightness(im)
                # 手动调整亮度
                im_brightness = brightness.enhance(4)

                # im_brightness = ImageOps.autocontrast(im)
                enh_col = ImageEnhance.Color(im_brightness)
                image_colored = enh_col.enhance(4)
                contrast = ImageEnhance.Contrast(image_colored)
                im_contrast = contrast.enhance(2.0)
                image_cv = cv2.cvtColor(np.array(im_contrast), cv2.COLOR_RGB2BGR)

                image2 = cv2.resize(image_cv, None, fx=.4, fy=.4, interpolation=cv2.INTER_LINEAR)

                image2 = cv2.fastNlMeansDenoisingColored(image2, None, 5, 1, 7, 21)
                # image3 = cv2.fastNlMeansDenoisingColored(image2, None, 3, 1, 7, 21)
                image2 = noise(image2)

                image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))  # 创建 CLAHE 对象

                imgLocalEqu = clahe.apply(image)

                cv2.imwrite(os.path.join(root_new, file), imgLocalEqu)
            else:
                shutil.copy(file_path, os.path.join(root_new, file))
            print('处理完成: ', os.path.join(root_new, file))
            print('处理耗时:', time.time() - time1)
    return root_new


# main_process(r"E:\jtw2\test")


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


# @app.route('/projectA', methods=['post', 'get'])
def main_process():
    # path_ori = r'E:\images\llds\561c304140614919a41b866fa83fd9d9\18822c5085a824049d8300511802d493'   # exp5
    # path_ori = r'E:\DATA\demo3'   # exp6
    # path_ori = r'D:\下载\A验证样本'  # exp6
    # path_ori = r'C:\Users\jzkj\PycharmProjects\fire111\data\fire\images\test'  # exp7

    # args = request.get_json('data')
    #
    # print(args)
    # # # path_ori = r"data/fire/images/test2"  # 原始图片的路径
    # path_ori = args.get("fileImageDir")
    #
    enhance_flag = False  # 是否使用增强方式
    path_enhance = None
    # path_ori = r"C:\Users\jzkj\PycharmProjects\fire111\data\fire\images\test"
    # enhance_flag = False  # 是否使用增强方式
    # path_enhance = None
    # 图像增强

    # if enhance_flag:
    #     path_enhance = process_enhance(path_ori)
    # 获取参数 修改参数
    opt = parse_opt()

    # opt.path_ori = path_ori
    # opt.path_enhance = path_enhance
    # opt.systemParam = args

    # for root, dirs, files in os.walk(path_ori):
    #     for file in files:
    #         file_path = os.path.join(root, file)
    #         # print("文件路径:", file_path)
    #
    #         try:
    #
    #             with Image.open(file_path) as img:
    #                 if img is None:
    #                     print(f"Image Not Found {file_path}")
    #                     continue
    #         except Exception as e:
    #             print(f"数据异常 {file_path}: {e}")
    #             if os.path.exists(root + '_tmp'):
    #                 pass
    #             else:
    #                 os.mkdir(root + '_tmp')
    #             shutil.move(file_path, os.path.join(root + '_tmp', file))
    #             continue
    #
    # file_list = os.listdir(path_ori)
    #
    # if len(file_list) == 0:
    #     ##TODO
    #
    #     """
    #         status: 状态
    #             0:成功   结果文件存在
    #             1:图像有且未捕捉到 跳过   会存在空文件 结果文件存在
    #             2:预处理后 图像目录为空    结果文件不存在
    #
    #     """
    #     # url = 'http://192.168.22.160:8088/pushResult'
    #     # r = requests.post(url=url, json={"status": "数据集为空,无法识别"},
    #     #                   headers={"Content-Type": "application/json; charset=UTF-8"}, verify=False)
    #     # print('成功结果反馈接口', r)
    #
    #     # url = 'http://192.168.22.160:8088/pushResult'
    #     # args['resourceFile'] = {}
    #     # args['status'] = 2
    #     # url_requests(args)
    #     return "数据预处理后为空"

    if path_enhance is not None:
        opt.source = path_enhance  # 修改待预测文件夹
    # else:
    #     opt.source = path_ori
    # opt.weights = r'C:\Users\ljy\Downloads\best (8).pt'  # 参数区分不同的项目模型
    main(opt)
    return "执行完毕"
    pass


def url_requests(param):
    """
        请求回调
    :param param:
    :return:
    """
    # url = 'http://192.168.22.160:8088/pushResult'
    url = 'http://0.0.0.0:8088/pushResult'
    r = requests.post(url=url, json=param,
                      headers={"Content-Type": "application/json; charset=UTF-8"}, verify=False)
    print('成功结果反馈接口', r)
    pass


def imitate_data():
    """
        模拟数据
    :return:
    """
    # df = pd.DataFrame([
    #     ['20230720-052434-612.txt', 'detect', 1393, 1646, 1658, 1924, 89.1478, 2560, 2560],
    #     ['20230720-052434-649.txt', 'detect', 583, 843, 1620, 1880, 88.1255, 2560, 2560],
    #     ['20230720-052434-949.txt', 'detect', 680, 930, 1690, 1954, 87.869, 2560, 2560],
    #     ['20230720-052436-149.txt', 'detect', 1453, 1692, 1697, 1962, 88.462, 2560, 2560],
    #     ['20230720-052436-599.txt', 'detect', 500, 756, 1663, 1936, 80.7031, 2560, 2560],
    #     ['20230720-052439-162.txt', 'detect', 1038, 1275, 1916, 2168, 82.0699, 2560, 2560],
    #     ['20230720-052439-812.txt', 'detect', 602, 835, 1017, 1259, 89.1259, 2560, 2560],
    #     ['20230720-052440-337.txt', 'detect', 993, 1230, 1078, 1329, 91.1373, 2560, 2560],
    #     ['20230720-052440-624.txt', 'detect', 1668, 1904, 1157, 1416, 88.1957, 2560, 2560]
    # ], columns=['picture_name', 'type_', 'distance_left', 'distance_right', 'distance_top', 'distance_bottom', 'conf',
    #             'im_w', 'im_h'])
    #
    # random_row = df.sample(n=1)

    random_row = pd.DataFrame(
        columns=['picture_name', 'type_', 'distance_left', 'distance_right', 'distance_top', 'distance_bottom', 'conf',
                 'im_w', 'im_h'])

    return random_row


main_process()
name='exp'
project=ROOT / 'runs/detect',
save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
# get_data(path1=r"F:\PythonProject\Job\data_a_b_cut\Dataset-原始\val\labels",
#          path2=str(save_dir)+r"labels2\result.csv")

#
# if __name__ == "__main__":
#     # app.run(host='192.168.22.160', port=5000, debug=True)
#     app.run(host='0.0.0.0', port=5000, debug=True)
