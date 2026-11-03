
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision


def xyxy2xywh(x):

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def box_iou(box1, box2):

    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):

        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)


    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    return inter / (area1[:, None] + area2 - inter)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):

    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):

    boxes[:, 0].clamp_(0, img_shape[1])
    boxes[:, 1].clamp_(0, img_shape[0])
    boxes[:, 2].clamp_(0, img_shape[1])
    boxes[:, 3].clamp_(0, img_shape[0])


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):

    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]
    coords[:, :10] /= gain

    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    coords[:, 4].clamp_(0, img0_shape[1])
    coords[:, 5].clamp_(0, img0_shape[0])
    coords[:, 6].clamp_(0, img0_shape[1])
    coords[:, 7].clamp_(0, img0_shape[0])
    coords[:, 8].clamp_(0, img0_shape[1])
    coords[:, 9].clamp_(0, img0_shape[0])
    return coords


def show_results(img, xywh, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def make_divisible(x, divisor):

    return (x // divisor) * divisor


def non_max_suppression_face(prediction, conf_thres=0.5, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 15
    xc = prediction[..., 4] > conf_thres


    min_wh, max_wh = 2, 4096

    redundant = True
    multi_label = nc > 1
    merge = False


    output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):


        x = x[xc[xi]]


        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 15), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 15] = 1.0
            x = torch.cat((x, v), 0)


        if not x.shape[0]:
            continue


        x[:, 15:] *= x[:, 4:5]


        box = xywh2xyxy(x[:, :4])


        if multi_label:
            i, j = (x[:, 15:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 15, None], x[i, 5:15] ,j[:, None].float()), 1)
        else:
            conf, j = x[:, 15:].max(1, keepdim=True)
            x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > conf_thres]


        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]


        n = x.shape[0]
        if not n:
            continue


        c = x[:, 15:16] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)


        if merge and (1 < n < 3E3):

            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]



    return output


class YoloFace():
    def __init__(self, pt_path=None, confThreshold=0.5, nmsThreshold=0.45, device='cuda'):
        pt_path = pt_path or Path(os.environ.get('PSYMM_WEIGHTS', Path(__file__).resolve().parents[4] / 'weights')) / 'dicetalk_weights/yolov5m-face.pt'
        assert os.path.exists(pt_path)

        self.inpSize = 416
        self.conf_thres = confThreshold
        self.iou_thres = nmsThreshold
        self.test_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(pt_path).to(self.test_device)
        self.last_w = 416
        self.last_h = 416
        self.grids = None

    @torch.no_grad()
    def detect(self, srcimg):


        h0, w0 = srcimg.shape[:2]
        r = self.inpSize / min(h0, w0)
        h1 = int(h0*r+31)//32*32
        w1 = int(w0*r+31)//32*32

        img = cv2.resize(srcimg, (w1,h1), interpolation=cv2.INTER_LINEAR)


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        img = torch.from_numpy(img).to(self.test_device).permute(2,0,1)
        img = img.float()/255
        if img.ndimension() == 3:
            img = img.unsqueeze(0)


        if h1 != self.last_h or w1 != self.last_w or self.grids is None:
            grids = []
            for scale in [8,16,32]:
                ny = h1//scale
                nx = w1//scale
                yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
                grid = torch.stack((xv, yv), 2).view((1,1,ny, nx, 2)).float()
                grids.append(grid.to(self.test_device))
            self.grids = grids
            self.last_w = w1
            self.last_h = h1

        pred = self.model(img, self.grids).cpu()


        det = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)[0]


        bboxes = np.zeros((det.shape[0], 4))
        kpss = np.zeros((det.shape[0], 5, 2))
        scores = np.zeros((det.shape[0]))


        det = det.cpu().numpy()

        for j in range(det.shape[0]):

            bboxes[j, 0] = det[j, 0] * w0/w1
            bboxes[j, 1] = det[j, 1] * h0/h1
            bboxes[j, 2] = det[j, 2] * w0/w1 - bboxes[j, 0]
            bboxes[j, 3] = det[j, 3] * h0/h1 - bboxes[j, 1]
            scores[j] = det[j, 4]

            kpss[j, :, :] = det[j, 5:15].reshape(5, 2) * np.array([[w0/w1,h0/h1]])


        return bboxes, kpss, scores



if __name__ == '__main__':
    import time

    imgpath = 'test.png'

    yoloface = YoloFace(pt_path=Path(os.environ.get('PSYMM_WEIGHTS', Path(__file__).resolve().parents[4] / 'weights')) / 'dicetalk_weights/yoloface_v5m.pt')
    srcimg = cv2.imread(imgpath)


    bboxes, kpss, scores = yoloface.detect(srcimg)
    bboxes, kpss, scores = yoloface.detect(srcimg)
    bboxes, kpss, scores = yoloface.detect(srcimg)

    t1 = time.time()
    for _ in range(10):
        bboxes, kpss, scores = yoloface.detect(srcimg)
    t2 = time.time()
    print('total time: {} ms'.format((t2 - t1) * 1000))
    for i in range(bboxes.shape[0]):
        xmin, ymin, xamx, ymax = int(bboxes[i, 0]), int(bboxes[i, 1]), int(bboxes[i, 0] + bboxes[i, 2]), int(bboxes[i, 1] + bboxes[i, 3])
        cv2.rectangle(srcimg, (xmin, ymin), (xamx, ymax), (0, 0, 255), thickness=2)
        for j in range(5):
            cv2.circle(srcimg, (int(kpss[i, j, 0]), int(kpss[i, j, 1])), 1, (0, 255, 0), thickness=5)
    cv2.imwrite('test_yoloface.jpg', srcimg)
