# # # import sys
# # # import os
# # # import time
# # # import argparse
# # #
# # # import torch
# # # import torch.backends.cudnn as cudnn
# # # from torch.autograd import Variable
# # #
# # # from PIL import Image
# # #
# # # import cv2
# # # import numpy as np
# # # import craft_utils
# # # import imgproc
# # # import file_utils
# # #
# # # from craft import CRAFT
# # #
# # # from collections import OrderedDict
# # # from pdf2image import convert_from_path
# # # import easyocr
# # #
# # #
# # # def copyStateDict(state_dict):
# # #     if list(state_dict.keys())[0].startswith("module"):
# # #         start_idx = 1
# # #     else:
# # #         start_idx = 0
# # #     new_state_dict = OrderedDict()
# # #     for k, v in state_dict.items():
# # #         name = ".".join(k.split(".")[start_idx:])
# # #         new_state_dict[name] = v
# # #     return new_state_dict
# # #
# # #
# # # def str2bool(v):
# # #     return v.lower() in ("yes", "y", "true", "t", "1")
# # #
# # #
# # # parser = argparse.ArgumentParser(description='CRAFT Text Detection')
# # # parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
# # # parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
# # # parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
# # # parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
# # # parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
# # # parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
# # # parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
# # # parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
# # # parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
# # # parser.add_argument('--test_folder', default='testImage/', type=str, help='folder path to input images')
# # # parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
# # # parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str,
# # #                     help='pretrained refiner model')
# # # parser.add_argument('--pdf_folder', default='C:/Users/samsung/Desktop/python 가상환경/CRAFT-pytorch/pdfFile', type=str,
# # #                     help='folder path to PDF files')
# # # parser.add_argument('--poppler_path', default='C:/Users/samsung/Desktop/python 가상환경/poppler-24.07.0/Library/bin',
# # #                     type=str, help='path to poppler bin')
# # # parser.add_argument('--output_folder', default='C:/Users/samsung/Desktop/python 가상환경/CRAFT-pytorch/testImage', type=str,
# # #                     help='folder path to save images')
# # # parser.add_argument('--result_folder', default='result/', type=str, help='folder path to save results')
# # # parser.add_argument('--text_result_folder', default='textResult/', type=str, help='folder path to save text results')
# # #
# # # args = parser.parse_args()
# # #
# # #
# # # def pdf_to_images(pdf_folder, output_folder, poppler_path, image_format='jpeg'):
# # #     if not os.path.exists(output_folder):
# # #         os.makedirs(output_folder)
# # #
# # #     pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
# # #     all_image_paths = []
# # #
# # #     for pdf_file in pdf_files:
# # #         pdf_path = os.path.join(pdf_folder, pdf_file)
# # #         images = convert_from_path(pdf_path, poppler_path=poppler_path)
# # #         for i, image in enumerate(images):
# # #             image_path = os.path.join(output_folder, f'{os.path.splitext(pdf_file)[0]}_page_{i + 1}.{image_format}')
# # #             image.save(image_path, image_format.upper())
# # #             all_image_paths.append(image_path)
# # #
# # #     return all_image_paths
# # #
# # #
# # # def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
# # #     t0 = time.time()
# # #
# # #     # resize
# # #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
# # #                                                                           interpolation=cv2.INTER_LINEAR,
# # #                                                                           mag_ratio=args.mag_ratio)
# # #     ratio_h = ratio_w = 1 / target_ratio
# # #
# # #     # preprocessing
# # #     x = imgproc.normalizeMeanVariance(img_resized)
# # #     x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
# # #     x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
# # #     if cuda:
# # #         x = x.cuda()
# # #
# # #     # forward pass
# # #     with torch.no_grad():
# # #         y, feature = net(x)
# # #
# # #     # make score and link map
# # #     score_text = y[0, :, :, 0].cpu().data.numpy()
# # #     score_link = y[0, :, :, 1].cpu().data.numpy()
# # #
# # #     # refine link
# # #     if refine_net is not None:
# # #         with torch.no_grad():
# # #             y_refiner = refine_net(y, feature)
# # #         score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
# # #
# # #     t0 = time.time() - t0
# # #     t1 = time.time()
# # #
# # #     # Post-processing
# # #     boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
# # #
# # #     # coordinate adjustment
# # #     boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
# # #     polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
# # #     for k in range(len(polys)):
# # #         if polys[k] is None: polys[k] = boxes[k]
# # #
# # #     t1 = time.time() - t1
# # #
# # #     # render results (optional)
# # #     render_img = score_text.copy()
# # #     render_img = np.hstack((render_img, score_link))
# # #     # ret_score_text = imgproc.cvt2HeatmapImg(render_img)
# # #
# # #     if args.show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
# # #
# # #     return boxes, polys, None  # ret_score_text
# # #
# # #
# # # def extract_text_from_boxes(image, boxes, text_result_path):
# # #     reader = easyocr.Reader(['en', 'ko'])  # EasyOCR Reader 객체 생성
# # #     result_text = ""
# # #     for box in boxes:
# # #         x_min = int(min(box[:, 0]))
# # #         x_max = int(max(box[:, 0]))
# # #         y_min = int(min(box[:, 1]))
# # #         y_max = int(max(box[:, 1]))
# # #
# # #         cropped_img = image[y_min:y_max, x_min:x_max]
# # #         result = reader.readtext(cropped_img)
# # #         for res in result:
# # #             result_text += res[1] + "\n"
# # #
# # #     with open(text_result_path, 'w', encoding='utf-8') as f:
# # #         f.write(result_text)
# # #
# # #
# # # def detecting(image_list):
# # #     result_folder = args.result_folder
# # #     text_result_folder = args.text_result_folder
# # #     if not os.path.exists(result_folder):
# # #         os.makedirs(result_folder)
# # #     if not os.path.exists(text_result_folder):
# # #         os.makedirs(text_result_folder)
# # #
# # #     result = []
# # #     args.cuda = False  # GPU 사용시 변경
# # #     # load net
# # #     net = CRAFT()  # initialize
# # #
# # #     print('Loading weights from checkpoint (' + args.trained_model + ')')
# # #     if args.cuda:
# # #         print("GPU detected!")
# # #         net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
# # #     else:
# # #         net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
# # #
# # #     if args.cuda:
# # #         net = net.cuda()
# # #         net = torch.nn.DataParallel(net)
# # #         cudnn.benchmark = False
# # #
# # #     net.eval()
# # #
# # #     # LinkRefiner
# # #     refine_net = None
# # #     if args.refine:
# # #         from refinenet import RefineNet
# # #         refine_net = RefineNet()
# # #         print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
# # #         if args.cuda:
# # #             refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
# # #             refine_net = refine_net.cuda()
# # #             refine_net = torch.nn.DataParallel(refine_net)
# # #         else:
# # #             refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
# # #
# # #         refine_net.eval()
# # #         args.poly = True
# # #
# # #     t = time.time()
# # #
# # #     # load data
# # #     for k, image_path in enumerate(image_list):
# # #         print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
# # #         image = imgproc.loadImage(image_path)
# # #
# # #         bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text,
# # #                                              args.cuda, args.poly, refine_net)
# # #
# # #         # save results
# # #         filename, file_ext = os.path.splitext(os.path.basename(image_path))
# # #         result_image_path = os.path.join(result_folder, f'res_{filename}.jpg')
# # #         file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)
# # #         result.append(polys)
# # #
# # #         # Extract text and save to textResult folder
# # #         text_result_path = os.path.join(text_result_folder, f'{filename}.txt')
# # #         extract_text_from_boxes(image, polys, text_result_path)
# # #
# # #         # Optional: save the mask file
# # #         # mask_file = os.path.join(result_folder, f'res_{filename}_mask.jpg')
# # #         # cv2.imwrite(mask_file, score_text)
# # #
# # #     print("elapsed time : {}s".format(time.time() - t))
# # #     result.append(image_list)
# # #     return result
# # #
# # #
# # # def main():
# # #     image_list = pdf_to_images(args.pdf_folder, args.output_folder, args.poppler_path)
# # #     detecting(image_list)
# # #
# # #
# # # if __name__ == "__main__":
# # #     main()
#
# import sys
# import os
# import time
# import argparse
#
# import torch
# import torch.backends.cudnn as cudnn
# from torch.autograd import Variable
#
# from PIL import Image
#
# import cv2
# import numpy as np
# import craft_utils
# import imgproc
# import file_utils
#
# from craft import CRAFT
#
# from collections import OrderedDict
# from pdf2image import convert_from_path
# import easyocr
#
#
# def copyStateDict(state_dict):
#     if list(state_dict.keys())[0].startswith("module"):
#         start_idx = 1
#     else:
#         start_idx = 0
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = ".".join(k.split(".")[start_idx:])
#         new_state_dict[name] = v
#     return new_state_dict
#
#
# def str2bool(v):
#     return v.lower() in ("yes", "y", "true", "t", "1")
#
#
# parser = argparse.ArgumentParser(description='CRAFT Text Detection')
# parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
# parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
# parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
# parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
# parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
# parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
# parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
# parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
# parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
# parser.add_argument('--test_folder', default='testImage/', type=str, help='folder path to input images')
# parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
# parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str,
#                     help='pretrained refiner model')
# parser.add_argument('--pdf_folder', default='C:/Users/samsung/Desktop/python 가상환경/CRAFT-pytorch/pdfFile', type=str,
#                     help='folder path to PDF files')
# parser.add_argument('--poppler_path', default='C:/Users/samsung/Desktop/python 가상환경/poppler-24.07.0/Library/bin',
#                     type=str, help='path to poppler bin')
# parser.add_argument('--output_folder', default='C:/Users/samsung/Desktop/python 가상환경/CRAFT-pytorch/testImage', type=str,
#                     help='folder path to save images')
# parser.add_argument('--result_folder', default='result/', type=str, help='folder path to save results')
# parser.add_argument('--text_result_folder', default='textResult/', type=str, help='folder path to save text results')
#
# args = parser.parse_args()
#
#
# def pdf_to_images(pdf_folder, output_folder, poppler_path, image_format='jpeg'):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
#     all_image_paths = []
#
#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(pdf_folder, pdf_file)
#         images = convert_from_path(pdf_path, poppler_path=poppler_path)
#         for i, image in enumerate(images):
#             image_path = os.path.join(output_folder, f'{os.path.splitext(pdf_file)[0]}_page_{i + 1}.{image_format}')
#             image.save(image_path, image_format.upper())
#             all_image_paths.append(image_path)
#
#     return all_image_paths
#
#
# def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
#     t0 = time.time()
#
#     # resize
#     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
#                                                                           interpolation=cv2.INTER_LINEAR,
#                                                                           mag_ratio=args.mag_ratio)
#     ratio_h = ratio_w = 1 / target_ratio
#
#     # preprocessing
#     x = imgproc.normalizeMeanVariance(img_resized)
#     x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
#     x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
#     if cuda:
#         x = x.cuda()
#
#     # forward pass
#     with torch.no_grad():
#         y, feature = net(x)
#
#     # make score and link map
#     score_text = y[0, :, :, 0].cpu().data.numpy()
#     score_link = y[0, :, :, 1].cpu().data.numpy()
#
#     # refine link
#     if refine_net is not None:
#         with torch.no_grad():
#             y_refiner = refine_net(y, feature)
#         score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
#
#     t0 = time.time() - t0
#     t1 = time.time()
#
#     # Post-processing
#     boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
#
#     # coordinate adjustment
#     boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
#     polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
#     for k in range(len(polys)):
#         if polys[k] is None: polys[k] = boxes[k]
#
#     t1 = time.time() - t1
#
#     # render results (optional)
#     render_img = score_text.copy()
#     render_img = np.hstack((render_img, score_link))
#     # ret_score_text = imgproc.cvt2HeatmapImg(render_img)
#
#     if args.show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
#
#     return boxes, polys, None  # ret_score_text
#
#
# def extract_text_from_boxes(image, boxes, text_result_path):
#     reader = easyocr.Reader(['en', 'ko'])  # EasyOCR Reader 객체 생성
#     results = []
#
#     for box in boxes:
#         x_min = int(min(box[:, 0]))
#         x_max = int(max(box[:, 0]))
#         y_min = int(min(box[:, 1]))
#         y_max = int(max(box[:, 1]))
#
#         cropped_img = image[y_min:y_max, x_min:x_max]
#         result = reader.readtext(cropped_img)
#         results.extend(result)
#
#     # y 좌표를 기준으로 정렬
#     results = sorted(results, key=lambda r: r[0][0][1])
#
#     with open(text_result_path, 'w', encoding='utf-8') as f:
#         for res in results:
#             f.write(res[1] + "\n")
#
#
# def detecting(image_list):
#     result_folder = args.result_folder
#     text_result_folder = args.text_result_folder
#     if not os.path.exists(result_folder):
#         os.makedirs(result_folder)
#     if not os.path.exists(text_result_folder):
#         os.makedirs(text_result_folder)
#
#     result = []
#     args.cuda = False  # GPU 사용시 변경
#     # load net
#     net = CRAFT()  # initialize
#
#     print('Loading weights from checkpoint (' + args.trained_model + ')')
#     if args.cuda:
#         print("GPU detected!")
#         net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
#     else:
#         net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
#
#     if args.cuda:
#         net = net.cuda()
#         net = torch.nn.DataParallel(net)
#         cudnn.benchmark = False
#
#     net.eval()
#
#     # LinkRefiner
#     refine_net = None
#     if args.refine:
#         from refinenet import RefineNet
#         refine_net = RefineNet()
#         print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
#         if args.cuda:
#             refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
#             refine_net = refine_net.cuda()
#             refine_net = torch.nn.DataParallel(refine_net)
#         else:
#             refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
#
#         refine_net.eval()
#         args.poly = True
#
#     t = time.time()
#
#     # load data
#     for k, image_path in enumerate(image_list):
#         print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
#         image = imgproc.loadImage(image_path)
#
#         bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text,
#                                              args.cuda, args.poly, refine_net)
#
#         # save results
#         filename, file_ext = os.path.splitext(os.path.basename(image_path))
#         result_image_path = os.path.join(result_folder, f'res_{filename}.jpg')
#         file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)
#         result.append(polys)
#
#         # Extract text and save to textResult folder
#         text_result_path = os.path.join(text_result_folder, f'{filename}.txt')
#         extract_text_from_boxes(image, polys, text_result_path)
#
#         # Optional: save the mask file
#         # mask_file = os.path.join(result_folder, f'res_{filename}_mask.jpg')
#         # cv2.imwrite(mask_file, score_text)
#
#     print("elapsed time : {}s".format(time.time() - t))
#     result.append(image_list)
#     return result
#
#
# def main():
#     image_list = pdf_to_images(args.pdf_folder, args.output_folder, args.poppler_path)
#     detecting(image_list)
#
#
# if __name__ == "__main__":
#     main()
import sys
import os
import time
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
import numpy as np
import craft_utils
import imgproc
import file_utils

from craft import CRAFT

from collections import OrderedDict
from pdf2image import convert_from_path
import easyocr


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='testImage/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
parser.add_argument('--pdf_folder', default='C:/Users/samsung/Desktop/python 가상환경/CRAFT-pytorch/pdfFile', type=str, help='folder path to PDF files')
parser.add_argument('--poppler_path', default='C:/Users/samsung/Desktop/python 가상환경/poppler-24.07.0/Library/bin', type=str, help='path to poppler bin')
parser.add_argument('--output_folder', default='C:/Users/samsung/Desktop/python 가상환경/CRAFT-pytorch/testImage', type=str, help='folder path to save images')
parser.add_argument('--result_folder', default='result/', type=str, help='folder path to save results')
parser.add_argument('--text_result_folder', default='textResult/', type=str, help='folder path to save text results')

args = parser.parse_args()


def pdf_to_images(pdf_folder, output_folder, poppler_path, image_format='jpeg'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    all_image_paths = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f'{os.path.splitext(pdf_file)[0]}_page_{i + 1}.{image_format}')
            image.save(image_path, image_format.upper())
            all_image_paths.append(image_path)

    return all_image_paths


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    # ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, None  # ret_score_text


def extract_text_from_boxes(image, boxes):
    reader = easyocr.Reader(['en', 'ko'])  # EasyOCR Reader 객체 생성
    results = []

    for box in boxes:
        x_min = int(min(box[:, 0]))
        x_max = int(max(box[:, 0]))
        y_min = int(min(box[:, 1]))
        y_max = int(max(box[:, 1]))

        cropped_img = image[y_min:y_max, x_min:x_max]
        result = reader.readtext(cropped_img)
        results.extend(result)

    # y 좌표를 기준으로 정렬
    results = sorted(results, key=lambda r: r[0][0][1])

    return [res[1] for res in results]


def detecting(image_list):
    result_folder = args.result_folder
    text_result_folder = args.text_result_folder
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(text_result_folder):
        os.makedirs(text_result_folder)

    result = []
    args.cuda = False  # GPU 사용시 변경
    # load net
    net = CRAFT()  # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        print("GPU detected!")
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    all_text_results = []

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text,
                                             args.cuda, args.poly, refine_net)

        # save results
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        result_image_path = os.path.join(result_folder, f'res_{filename}.jpg')
        file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)
        result.append(polys)

        # Extract text and save to textResult folder
        text_result_path = os.path.join(text_result_folder, f'{filename}.txt')
        extracted_text = extract_text_from_boxes(image, polys)
        all_text_results.append(" ".join(extracted_text))
        with open(text_result_path, 'w', encoding='utf-8') as f:
            f.write(" ".join(extracted_text))

    with open(os.path.join(text_result_folder, 'final_text_results.txt'), 'w', encoding='utf-8') as f:
        f.write(str(all_text_results))

    print("elapsed time : {}s".format(time.time() - t))
    result.append(image_list)
    return result


def main():
    image_list = pdf_to_images(args.pdf_folder, args.output_folder, args.poppler_path)
    detecting(image_list)


if __name__ == "__main__":
    main()
