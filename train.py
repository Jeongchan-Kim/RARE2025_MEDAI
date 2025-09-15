#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

import os
import argparse
import csv
from datetime import datetime
from collections import Counter, defaultdict
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet50
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

import cv2
import wandb
import math
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# 로컬 유틸
from utils import progress_bar, ModelEMA, _make_pos_weight_tensor, split_dataset
from metrics import compute_metrics, log_confusion_matrix_wandb

# -----------------------------
# 인자
# -----------------------------


parser = argparse.ArgumentParser(description='RARE25 mixup Training with EMA + Cosine LR')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--optimizer', default='sgd', type=str, choices=['adam', 'adamw', 'sgd'], help='optimizer type')
parser.add_argument('--sched', default='cosine', type=str, choices=['none','step','cosine','cosine_wr'], help='LR scheduler')
parser.add_argument('--warmup_epochs', default=0, type=int, help='linear warmup epochs (0 to disable)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default="ResNet50", type=str, help='model type')
parser.add_argument('--seed', default=2024, type=int, help='random seed')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='total epochs')
parser.add_argument('--augment', action='store_true', help='use data augmentation') 
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float, help='mixup alpha')
parser.add_argument('--data_folder', default='~/data', type=str, help='dataset root')
parser.add_argument('--num_workers', default=8, type=int, help='dataloader workers')
parser.add_argument('--pretrained', action='store_true', help='use torchvision pretrained (ignored if pretrained-path set)')
parser.add_argument('--pretrained_path', default=None, type=str, help='Path to pretrained model weights')
parser.add_argument('--sampling', default='none', type=str, choices=['none', 'weighted'], help='sampler')
parser.add_argument('--ema', action='store_true', help='use EMA')
parser.add_argument('--weighted_bce_loss', action='store_true', help='BCE with pos_weight')
parser.add_argument('--bn_recalc', action='store_true', help='recalculate BN stats before eval (EMA)')
parser.add_argument('--mixup', default='vanilla', type=str, choices=['none', 'vanilla', 'balanced'],
                    help='mixup mode: none | vanilla | balanced')

parser.add_argument('--la', action='store_true', help='enable Logit Adjustment')
parser.add_argument('--la_tau', default=1.0, type=float, help='LA temperature (scale)')
parser.add_argument('--als', action='store_true', help='enable Asymmetric Label Smoothing')
parser.add_argument('--als_pos_eps', default=0.0, type=float, help='ALS epsilon for positive targets')
parser.add_argument('--als_neg_eps', default=0.05, type=float, help='ALS epsilon for negative targets')

parser.add_argument('--mode', default='bce',
    choices=['bce','cosface','arcface','amsoftmax','platt','bce_fp_weight','ohnm'], help='loss / inference mode')

parser.add_argument('--ckpt_backbone', type=str, default=None,
    help='(선택) 기존 BCE/기타 체크포인트에서 백본만 이어받기')

parser.add_argument('--val_ratio', default=0, type=float,
                    help='fraction of training data used as validation (0 disables split)')

# margin-softmax 공통 하이퍼
parser.add_argument('--margin-m', type=float, default=0.35, help='margin for CosFace/AM-Softmax (additive cosine margin)')
parser.add_argument('--margin-arc', type=float, default=0.50, help='margin for ArcFace (additive angular margin, radians)')
parser.add_argument('--scale-s', type=float, default=30.0, help='scale for margin-softmax heads')

# --- OHNM 하이퍼 (top-k% 선별) ---
parser.add_argument('--ohnm-topk', type=float, default=0.4,
    help='fraction (0~1): select top-k%% hardest negatives per batch')
parser.add_argument('--ohnm-metric', type=str, default='prob',
    choices=['prob','loss'],
    help="how to rank negatives: 'prob' (higher sigmoid prob) or 'loss' (higher per-sample loss)")

# negative FP loss
parser.add_argument('--fpw-enable', action='store_true',
    help='False Positive-weighting on negatives (label=0 & score>=th → *gamma)')
parser.add_argument('--fpw-thresh', type=float, default=0.5,
    help='score threshold to flag negative as suspected FP')
parser.add_argument('--fpw-gamma', type=float, default=3.0,
    help='loss weight for suspected negative FPs')
parser.add_argument('--fpw-warmup-epochs', type=int, default=0,
    help='epochs to skip FP-weighting at the beginning')

# --- Calibration options ---
parser.add_argument('--calib', default='none',
    choices=['none','platt','temp','isotonic','ccc','bbq','beta'],
    help='post-hoc calibration method (validated on VAL, applied to TEST)')
parser.add_argument('--holdout_val_ratio', type=float, default=0.5,
    help='fraction of holdout(test_dir) used as VAL; rest as TEST')
parser.add_argument('--bbq_bins', type=int, default=10, help='number of bins for BBQ')
parser.add_argument('--iso_out_of_bounds', default='clip', choices=['clip','nearest'],
    help='how to extrapolate isotonic outside fitted range')


args = parser.parse_args()

use_cuda = torch.cuda.is_available()
if args.seed != 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# -----------------------------
# W&B
# -----------------------------
def wandb_init(args):
    wandb.init(
        entity="medai-endoscopy",
        project="RARE-Challenge-jeongchan-cali",
        name=args.model_name,
        config=vars(args)
    )

# ---------- target helpers ----------
def get_targets_any(ds) -> Sequence[int]:
    """ImageFolder / Subset 모두 대응해서 [0/1,...] 반환"""
    if hasattr(ds, 'indices') and hasattr(ds, 'dataset'):  # Subset
        base = ds.dataset; idxs = ds.indices
        if hasattr(base, 'targets'):
            return [int(base.targets[i]) for i in idxs]
        else:
            return [int(base[i][1]) for i in idxs]
    elif hasattr(ds, 'targets'):
        return [int(t) for t in ds.targets]
    else:
        return [int(ds[i][1]) for i in range(len(ds))]

def underlying_len(ds) -> int:
    """Subset이면 원본 길이, 아니면 현재 길이"""
    return len(ds.dataset) if (hasattr(ds, 'dataset') and hasattr(ds, 'indices')) else len(ds)

def build_class_to_indices_local(ds):
    """로컬 인덱스(0..len(ds)-1) 기준 class→indices"""
    c2i = defaultdict(list)
    for i in range(len(ds)):
        samp = ds[i]
        if isinstance(samp, (list, tuple)):
            y = int(samp[1])
        else:
            raise RuntimeError("Unexpected dataset item format")
        c2i[y].append(i)
    return c2i

# ===== Transforms =====
class ForceRGB:
    def __call__(self, im):
        return im.convert("RGB") if isinstance(im, Image.Image) and im.mode != "RGB" else im

to_rgb = ForceRGB()

if args.augment:
    transform_train = transforms.Compose([
        to_rgb,
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
else:
    transform_train = transforms.Compose([
        to_rgb,
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

transform_valtest = transforms.Compose([
    to_rgb,
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

# ===== Dataset & Split =====
train_dir = os.path.join(os.path.expanduser(args.data_folder), 'train')
test_dir  = os.path.join(os.path.expanduser(args.data_folder), 'test')
has_test_dir = os.path.isdir(test_dir)

# 기본 트랜스폼은 그대로 사용
imgfolder_train = datasets.ImageFolder(root=train_dir, transform=transform_train)

# 기본값: 기존 코드와 동일(= train에서 split)
imgfolder_val_fallback   = datasets.ImageFolder(root=train_dir, transform=transform_valtest)
imgfolder_test_fallback  = datasets.ImageFolder(root=train_dir, transform=transform_valtest)

# ---- 정책 ----
# - test/ 폴더가 있고, --val_ratio 0 이면: 외부 test 를 Val/Test 둘 다 사용 (val=test 모드)
# - 그 외에는: 기존처럼 train 안에서 stratified split
use_external_test_as_val = has_test_dir and (args.val_ratio == 0)

if use_external_test_as_val:
    # 외부 test 폴더를 Val/Test 둘 다로
    imgfolder_val  = datasets.ImageFolder(root=test_dir,  transform=transform_valtest)
    imgfolder_test = datasets.ImageFolder(root=test_dir,  transform=transform_valtest)

    # train 은 전체 사용, val/test 는 test 전체 사용
    train_idx = np.arange(len(imgfolder_train))
    val_idx   = np.arange(len(imgfolder_val))
    test_idx  = np.arange(len(imgfolder_test))
else:
    # 기존 경로: train 안에서 split
    imgfolder_val  = imgfolder_val_fallback
    imgfolder_test = imgfolder_test_fallback

    targets_full = [int(t) for t in getattr(imgfolder_train, 'targets', [])]
    n_total = len(targets_full)

    if args.val_ratio > 0:
        train_idx, valtest_idx = train_test_split(
            np.arange(n_total),
            test_size=args.val_ratio,
            stratify=targets_full,
            random_state=args.seed,
        )
    else:
        train_idx = np.arange(n_total)
        valtest_idx = []

    if len(valtest_idx) > 0:
        vt_targets = [targets_full[i] for i in valtest_idx]
        val_idx, test_idx = train_test_split(
            valtest_idx,
            test_size=args.holdout_val_ratio,
            stratify=vt_targets,
            random_state=args.seed,
        )
    else:
        val_idx, test_idx = [], []


# ===== SubsetWithIndex: transform 건드리지 않고 인덱스만 추가 =====
# ===== Indexed subset (no inheritance from Subset) =====
class IndexedImageFolderSubset(torch.utils.data.Dataset):
    """
    ImageFolder의 파일 리스트(samples)와 loader/transform을 그대로 쓰되,
    우리가 고른 indices만 접근하고 (img, label, real_idx) 3-튜플을 반환한다.
    """
    def __init__(self, imagefolder: datasets.ImageFolder, indices, transform=None):
        self.base = imagefolder
        self.indices = np.asarray(indices, dtype=np.int64)
        # transform 우선순위: 인자로 주면 그걸, 아니면 base.transform
        self.transform = transform if transform is not None else imagefolder.transform
        self.loader = imagefolder.loader
        self.samples = imagefolder.samples  # [(path, class), ...]
        assert self.transform is not None, "IndexedImageFolderSubset needs a non-None transform."

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, i):
        real_idx = int(self.indices[i])
        path, target = self.samples[real_idx]
        img = self.loader(path)  # PIL.Image

        # transform 적용 → Tensor 보장
        img = self.transform(img)
        if not torch.is_tensor(img):
            raise TypeError(f"[Transform Bug] Expected Tensor, got {type(img)} for path: {path}")

        return img, int(target), real_idx


# 최종 Dataset (※ transform은 ImageFolder에 이미 들어가 있지만, 여기서도 명시해도 무방)
# 최종 Dataset
train_dataset = IndexedImageFolderSubset(imgfolder_train, train_idx, transform=transform_train)

# 주의: 외부 test 를 쓰는 모드면 val/test 둘 다 "test 폴더" 기반
val_dataset   = IndexedImageFolderSubset(imgfolder_val,   val_idx,   transform=transform_valtest)
test_dataset  = IndexedImageFolderSubset(imgfolder_test,  test_idx,  transform=transform_valtest)

# datasets 만든 다음
train_targets = [int(imgfolder_train.targets[i]) for i in train_idx]
class_counts  = Counter(train_targets)

pos_ratio = float(sum(train_targets)) / max(1, len(train_targets))
pos_ratio = float(np.clip(pos_ratio, 1e-6, 1 - 1e-6))
la_logit_bias = args.la_tau * float(np.log(pos_ratio / (1.0 - pos_ratio)))

class_to_indices = build_class_to_indices_local(train_dataset)
print("[balanced] class_to_indices sizes:",
      {k: len(v) for k, v in class_to_indices.items()})

# ===== Sampler & Loaders (여기 '한 번만' 생성!) =====
train_targets = [int(imgfolder_train.targets[i]) for i in train_idx]
class_counts = Counter(train_targets)

def make_loader(ds, shuffle, sampler=None):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

if args.sampling == 'weighted':
    class_weights = {cls: 1.0 / cnt for cls, cnt in class_counts.items()}
    weights = [class_weights[int(y)] for y in train_targets]
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )
else:
    train_sampler = None

trainloader       = make_loader(train_dataset, shuffle=(train_sampler is None), sampler=train_sampler)
valloader         = make_loader(val_dataset,   shuffle=False)
testloader        = make_loader(test_dataset,  shuffle=False)
train_eval_loader = make_loader(train_dataset, shuffle=False)

# 1) dataset 타입
print("trainloader.dataset:", type(trainloader.dataset).__name__)
# 2) 단일 샘플 3-튜플 확인
s0 = train_dataset[0]
print("sample[0] types:", type(s0), len(s0), type(s0[0]), type(s0[1]), type(s0[2]))
# 3) 배치가 (x,y,idx)로 나오는지
xb, yb, ib = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=0)))
print("batch lens:", len((xb, yb, ib)))
print("xb:", xb.shape, xb.dtype, "yb:", yb.shape, yb.dtype, "ib:", ib.shape, ib.dtype)


print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Test size:", len(test_dataset))



# FPW 초기화: 원본 인덱스 길이(Subset 대비 안전)
FPW_VEC = torch.ones(underlying_len(train_dataset), dtype=torch.float32)  # CPU tensor

# =======================
# Margin-Softmax 헤드
# =======================

class L2Norm(nn.Module):
    def forward(self, x, eps=1e-7):
        return x / (x.norm(p=2, dim=1, keepdim=True) + eps)

class CosFaceHead(nn.Module):
    def __init__(self, feat_dim, num_classes=2, s=30.0, m=0.35):
        super().__init__()
        self.s, self.m = s, m
        self.W = nn.Parameter(torch.randn(feat_dim, num_classes))
        nn.init.xavier_uniform_(self.W)
        self.l2 = L2Norm()
    def forward(self, x, y):
        x = self.l2(x)                       # [B, D]
        W = self.l2(self.W)                  # [D, C]
        cos = x @ W                          # [B, C]
        # 정답 로짓에 margin 빼기
        if y is not None:
            cos = cos.scatter_add(1, y.view(-1,1),
                                  (-self.m) * torch.ones_like(y, dtype=cos.dtype, device=cos.device).view(-1,1))
        return self.s * cos

class ArcFaceHead(nn.Module):
    def __init__(self, feat_dim, num_classes=2, s=30.0, m=0.50):
        super().__init__()
        self.s, self.m = s, m
        self.W = nn.Parameter(torch.randn(feat_dim, num_classes))
        nn.init.xavier_uniform_(self.W)
        self.l2 = L2Norm()
        self.register_buffer('cos_m', torch.tensor(math.cos(m)))
        self.register_buffer('sin_m', torch.tensor(math.sin(m)))
    def forward(self, x, y):
        x = self.l2(x); W = self.l2(self.W)
        cos = x @ W
        if y is not None:
            theta_y = cos.gather(1, y.view(-1,1))
            sin_theta = torch.sqrt(torch.clamp(1.0 - theta_y**2, min=1e-7))
            cos_theta_m = theta_y * self.cos_m - sin_theta * self.sin_m
            cos = cos.scatter(1, y.view(-1,1), cos_theta_m)
        return self.s * cos
class AMSoftmaxHead(nn.Module):
    def __init__(self, feat_dim, num_classes=2, s=30.0, m=0.35):
        super().__init__()
        self.s, self.m = s, m
        self.W = nn.Parameter(torch.randn(feat_dim, num_classes))
        nn.init.xavier_uniform_(self.W)
        self.l2 = L2Norm()
    def forward(self, x, y):
        x = self.l2(x); W = self.l2(self.W)
        cos = x @ W
        if y is not None:
            cos = cos.scatter_add(1, y.view(-1,1),
                                  (-self.m) * torch.ones_like(y, dtype=cos.dtype, device=cos.device).view(-1,1))
        return self.s * cos

def mixup_class(inputs, targets_long, alpha=1.0, use_cuda=True):
    """
    inputs: [B, C, H, W], targets_long: [B] (int64)
    return: mixed_x, y_a_long, y_b_long, lam(float)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    b = inputs.size(0)
    index = torch.randperm(b).cuda() if use_cuda else torch.randperm(b)
    mixed_x = lam * inputs + (1 - lam) * inputs[index, :]
    y_a = targets_long                     # [B]
    y_b = targets_long[index]              # [B]
    return mixed_x, y_a, y_b, float(lam)


# =======================
# NEW: Calibration utils
# =======================
def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def _logit_np(p, eps=1e-6):
    p = np.clip(p, eps, 1.0-eps)
    return np.log(p/(1.0-p))

# ----- Isotonic (robust PAV) -----
def _pav_sorted(x: np.ndarray, y: np.ndarray, w: np.ndarray = None):
    """
    Robust stack-based PAV.
    x, y are sorted by x (ascending). w is optional weights (default 1).
    Returns:
      right_edges: monotonically increasing breakpoints (right edge of each block)
      means: block means (non-decreasing)
    """
    n = len(x)
    if w is None:
        w = np.ones(n, dtype=np.float64)
    # blocks stored as [sum_wy, sum_w, right_edge_index]
    sum_wy = []
    sum_w  = []
    r_edge = []

    for i in range(n):
        # create new block with singleton i
        sum_wy.append(w[i] * y[i])
        sum_w.append(w[i])
        r_edge.append(i)
        # merge while last two blocks violate monotonicity
        while len(sum_wy) >= 2 and (sum_wy[-2] / sum_w[-2]) > (sum_wy[-1] / sum_w[-1]):
            # merge last two into second last
            sum_wy[-2] += sum_wy[-1]
            sum_w[-2]  += sum_w[-1]
            r_edge[-2]  = r_edge[-1]
            # pop last
            sum_wy.pop()
            sum_w.pop()
            r_edge.pop()

    # construct right_edges (x at right edge) and means
    right_edges = np.array([x[j] for j in r_edge], dtype=np.float64)
    means = np.array([sum_wy[k] / sum_w[k] for k in range(len(sum_w))], dtype=np.float64)

    # ensure strictly increasing right_edges to make searchsorted stable
    # if there are ties in x (identical probabilities), nudge by tiny eps
    eps = 1e-12
    for i in range(1, len(right_edges)):
        if right_edges[i] <= right_edges[i-1]:
            right_edges[i] = right_edges[i-1] + eps

    return right_edges, means


class IsotonicCalibrator:
    def __init__(self, out_of_bounds='clip'):
        self.edges = None   # right edges of blocks
        self.values = None  # block means
        self.out_of_bounds = out_of_bounds

    def fit(self, p, y):
        p = np.asarray(p, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # guard: degenerate cases
        if len(p) < 2 or np.unique(p).size < 2:
            # not enough support to fit: identity
            self.edges = np.array([0.0, 1.0])
            self.values = np.array([y.mean(), y.mean()])
            return self

        if (y == 0).all() or (y == 1).all():
            # all one class → constant mapping
            const = float(y.mean())
            self.edges = np.array([0.0, 1.0])
            self.values = np.array([const, const])
            return self

        # sort by p
        order = np.argsort(p)
        ps = p[order]
        ys = y[order]

        # robust PAV
        edges, means = _pav_sorted(ps, ys)
        # store edges and means
        # prepend a left edge to cover values < edges[0] during clipping
        self.edges = edges
        self.values = means
        return self

    def predict(self, p):
        p = np.asarray(p, dtype=np.float64)
        if self.edges is None or self.values is None:
            return p  # identity if not fitted

        x = p.copy()
        if self.out_of_bounds == 'clip':
            # clip to [min_edge, max_edge]
            x = np.clip(x, self.edges[0], self.edges[-1])
        # right-constant step function: find block by right edge
        idx = np.searchsorted(self.edges, x, side='right') - 1
        idx = np.clip(idx, 0, len(self.values)-1)
        return self.values[idx]

# ----- Platt (Az+B) -----
class PlattCalibrator:
    def __init__(self):
        self.A = 0.0
        self.B = 0.0
    def fit(self, p, y, max_iter=100):
        z = _logit_np(p)
        A = torch.zeros(1, requires_grad=True)
        B = torch.zeros(1, requires_grad=True)
        x_t = torch.from_numpy(z).float()
        y_t = torch.from_numpy(y.astype(np.float32)).float()
        opt = torch.optim.LBFGS([A,B], lr=1.0, max_iter=max_iter, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            q = torch.sigmoid(A*x_t + B)
            loss = F.binary_cross_entropy(q, y_t)
            loss.backward()
            return loss
        opt.step(closure)
        self.A = float(A.detach().item())
        self.B = float(B.detach().item())
        return self
    def predict(self, p):
        z = _logit_np(p)
        return _sigmoid_np(self.A*z + self.B)

# ----- Temperature (z/T) -----
class TempScaler:
    def __init__(self):
        self.T = 1.0
    def fit(self, p, y, max_iter=100):
        # recover logits
        z = _logit_np(p)
        T = torch.ones(1, requires_grad=True)
        z_t = torch.from_numpy(z).float()
        y_t = torch.from_numpy(y.astype(np.float32)).float()
        opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            q = torch.sigmoid(z_t / torch.clamp(T, min=1e-3))
            loss = F.binary_cross_entropy(q, y_t)
            loss.backward()
            return loss
        opt.step(closure)
        self.T = float(torch.clamp(T, min=1e-3).detach().item())
        return self
    def predict(self, p):
        z = _logit_np(p)
        return _sigmoid_np(z / max(self.T, 1e-3))

# ----- Beta Calibration -----
class BetaCalibrator:
    def __init__(self):
        self.alpha = 1.0; self.beta = 1.0; self.gamma = 0.0
    def fit(self, p, y, max_iter=200):
        eps = 1e-6
        p = np.clip(p, eps, 1-eps)
        lp = np.log(p); lq = np.log(1-p)
        a = torch.zeros(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        g = torch.zeros(1, requires_grad=True)
        lp_t = torch.from_numpy(lp).float()
        lq_t = torch.from_numpy(lq).float()
        y_t  = torch.from_numpy(y.astype(np.float32)).float()
        opt = torch.optim.LBFGS([a,b,g], lr=1.0, max_iter=max_iter, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            q = torch.sigmoid(a*lp_t + b*lq_t + g)
            loss = F.binary_cross_entropy(q, y_t)
            loss.backward()
            return loss
        opt.step(closure)
        self.alpha = float(a.detach().item())
        self.beta  = float(b.detach().item())
        self.gamma = float(g.detach().item())
        return self
    def predict(self, p):
        eps=1e-6
        p = np.clip(p, eps, 1-eps)
        z = self.alpha*np.log(p) + self.beta*np.log(1-p) + self.gamma
        return _sigmoid_np(z)

# ----- BBQ (Jeffreys-smoothed histogram binning) -----
class BBQCalibrator:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.edges = None
        self.post_mean = None
    def fit(self, p, y):
        # quantile bins
        qs = np.linspace(0,1,self.n_bins+1)
        edges = np.quantile(p, qs)
        edges[0] = 0.0; edges[-1] = 1.0
        # avoid duplicates
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.linspace(0,1, min(self.n_bins,3))
        self.edges = edges
        post = []
        for i in range(len(edges)-1):
            l, r = edges[i], edges[i+1]
            mask = (p>=l) & (p<=r) if i==len(edges)-2 else (p>=l) & (p<r)
            n = mask.sum()
            k = y[mask].sum()
            # Jeffreys prior Beta(0.5,0.5)
            post.append((k+0.5)/(n+1.0) if n>0 else np.nan)
        # fill empty with linear interp
        post = np.array(post, dtype=np.float64)
        if np.any(np.isnan(post)):
            notnan = ~np.isnan(post)
            post[~notnan] = np.interp(np.where(~notnan)[0], np.where(notnan)[0], post[notnan])
        self.post_mean = post
        return self
    def predict(self, p):
        p = np.asarray(p, dtype=np.float64)
        idx = np.searchsorted(self.edges, p, side='right')-1
        idx = np.clip(idx, 0, len(self.post_mean)-1)
        return self.post_mean[idx]

# ----- CCC (binary, Platt per-class + renormalize) -----
class CCCPlattCalibrator:
    def __init__(self):
        self.A1 = 0.0; self.B1 = 0.0  # for positive class logit z
        self.A0 = 0.0; self.B0 = 0.0  # for negative class logit(-z)
    def fit(self, p, y, max_iter=200):
        z = _logit_np(p)
        A1 = torch.zeros(1, requires_grad=True); B1 = torch.zeros(1, requires_grad=True)
        A0 = torch.zeros(1, requires_grad=True); B0 = torch.zeros(1, requires_grad=True)
        z_t = torch.from_numpy(z).float()
        y_t = torch.from_numpy(y.astype(np.float32)).float()
        opt = torch.optim.LBFGS([A1,B1,A0,B0], lr=1.0, max_iter=max_iter, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            p1p = torch.sigmoid(A1*z_t + B1)
            p0p = torch.sigmoid(A0*(-z_t) + B0)
            q = p1p / torch.clamp(p1p + p0p, min=1e-6)
            loss = F.binary_cross_entropy(q, y_t)
            loss.backward()
            return loss
        opt.step(closure)
        self.A1 = float(A1.detach().item()); self.B1 = float(B1.detach().item())
        self.A0 = float(A0.detach().item()); self.B0 = float(B0.detach().item())
        return self
    def predict(self, p):
        z = _logit_np(p)
        p1p = _sigmoid_np(self.A1*z + self.B1)
        p0p = _sigmoid_np(self.A0*(-z) + self.B0)
        return p1p / np.clip(p1p + p0p, 1e-6, None)

# ----- factory -----
def fit_calibrator(method, probs, labels, bbq_bins=10, iso_oob='clip'):
    m = method.lower()
    if m == 'none':
        return {'type':'none'}
    if m == 'platt':
        cal = PlattCalibrator().fit(probs, labels)
        return {'type':'platt', 'A':cal.A, 'B':cal.B}
    if m == 'temp':
        cal = TempScaler().fit(probs, labels)
        return {'type':'temp', 'T':cal.T}
    if m == 'isotonic':
        cal = IsotonicCalibrator(out_of_bounds=iso_oob).fit(probs, labels)
        return {'type':'isotonic', 'edges':cal.edges.tolist(), 'values':cal.values.tolist(), 'oob':iso_oob}
    if m == 'bbq':
        cal = BBQCalibrator(n_bins=bbq_bins).fit(probs, labels)
        return {'type':'bbq', 'edges':cal.edges.tolist(), 'post':cal.post_mean.tolist()}
    if m == 'beta':
        cal = BetaCalibrator().fit(probs, labels)
        return {'type':'beta', 'alpha':cal.alpha, 'beta':cal.beta, 'gamma':cal.gamma}
    if m == 'ccc':
        cal = CCCPlattCalibrator().fit(probs, labels)
        return {'type':'ccc', 'A1':cal.A1, 'B1':cal.B1, 'A0':cal.A0, 'B0':cal.B0}
    raise ValueError(f"Unknown calibration method: {method}")

def apply_calibrator(calib, probs):
    t = calib.get('type','none')
    if t=='none': return probs
    p = probs
    if t=='platt':
        z = _logit_np(p); return _sigmoid_np(calib['A']*z + calib['B'])
    if t=='temp':
        z = _logit_np(p); return _sigmoid_np(z / max(calib['T'],1e-3))
    if t=='isotonic':
        edges = np.array(calib['edges']); vals = np.array(calib['values'])
        oob = calib.get('oob','clip')
        p2 = np.clip(p, edges[0], edges[-1]) if oob=='clip' else p
        idx = np.searchsorted(edges, p2, side='right') - 1
        idx = np.clip(idx, 0, len(vals)-1)
        return vals[idx]
    if t=='bbq':
        edges = np.array(calib['edges']); post = np.array(calib['post'])
        idx = np.searchsorted(edges, p, side='right')-1
        idx = np.clip(idx, 0, len(post)-1)
        return post[idx]
    if t=='beta':
        eps=1e-6; p2 = np.clip(p,eps,1-eps)
        z = calib['alpha']*np.log(p2) + calib['beta']*np.log(1-p2) + calib['gamma']
        return _sigmoid_np(z)
    if t=='ccc':
        z = _logit_np(p)
        p1p = _sigmoid_np(calib['A1']*z + calib['B1'])
        p0p = _sigmoid_np(calib['A0']*(-z) + calib['B0'])
        return p1p / np.clip(p1p + p0p, 1e-6, None)
    raise ValueError(f"Unknown calib type: {t}")

# -----------------------------
# 모델
# -----------------------------
print('==> Building model..')
# 백본: resnet50에서 fc 직전까지 꺼내기
def build_backbone():
    m = resnet50(pretrained=args.pretrained)
    feat_dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, feat_dim

# 기존 BCE 체크포인트에서 백본만 이어받기 (헤드 키 무시)
def load_backbone_from_ckpt(backbone, ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in sd: sd = sd['state_dict']
    new_sd = {}
    for k,v in sd.items():
        k2 = k.replace('module.', '')
        if k2.startswith('fc.'):   # 헤드는 무시
            continue
        new_sd[k2] = v
    missing, unexpected = backbone.load_state_dict(new_sd, strict=False)
    print(f"[backbone ckpt] missing={list(missing)} unexpected={list(unexpected)}")

# 모델 빌드
if args.mode in ['cosface','arcface','amsoftmax']:
    backbone, feat_dim = build_backbone()
    if args.ckpt_backbone:
        load_backbone_from_ckpt(backbone, args.ckpt_backbone)
    if args.mode == 'cosface':
        head = CosFaceHead(feat_dim, num_classes=2, s=30.0, m=0.35)
    elif args.mode == 'arcface':
        head = ArcFaceHead(feat_dim, num_classes=2, s=30.0, m=0.50)
    else:
        head = AMSoftmaxHead(feat_dim, num_classes=2, s=30.0, m=0.35)

    class MarginModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, x, y=None):
            feat = self.backbone(x)
            return self.head(feat, y)

    net = MarginModel(backbone, head)
else:
    # 기존 BCE 경로
    net = resnet50(pretrained=args.pretrained)
    net.fc = nn.Linear(net.fc.in_features, 1)
    if args.pretrained_path is not None:
        print(f"Loading weights from: {args.pretrained_path}")
        sd = torch.load(args.pretrained_path, map_location='cpu')
        if 'state_dict' in sd:
            sd = sd['state_dict']
        new_sd = {}
        for k, v in sd.items():
            nk = k.replace('module.', '') if k.startswith('module.') else k
            new_sd[nk] = v
        missing, unexpected = net.load_state_dict(new_sd, strict=False)
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

if not os.path.isdir('results'):
    os.mkdir('results')

# 모델 이름 태깅 시작
tags = []

# mixup
tags.append(f"{args.mixup}_a{args.alpha}")

# Logit Adjustment
if args.la:
    tags.append(f"LA{args.la_tau}")
else:
    tags.append("noLA")

# ALS
if args.als:
    tags.append(f"ALSpos{args.als_pos_eps}_neg{args.als_neg_eps}")
else:
    tags.append("noALS")

# Optimizer, LR
tags.append(f"lr{args.lr}_{args.optimizer}")

# Sampling
if args.sampling != 'none':
    tags.append(args.sampling)

# EMA
if args.ema:
    tags.append("ema")

# Weighted BCE
if args.weighted_bce_loss:
    tags.append("bce-weighted")

# FPW
if args.fpw_enable:
    tags.append(f"fpw_th{args.fpw_thresh}_g{args.fpw_gamma}")

# OHNM
if args.mode == 'ohnm':
    tk = f"{args.ohnm_topk:.2f}".rstrip('0').rstrip('.')  # 예: 0.2 -> "0.2"
    tags.append(f"ohnm_top{tk}_{args.ohnm_metric}")

# Calibration
if args.calib != 'none':
    tags.append(f"calib-{args.calib}")
else:
    tags.append("noCalib")

# 최종 model_name 조립
args.model_name = f"{args.model}_{args.mode}_" + "_".join(tags)

logname = ('results/log_' + net.__class__.__name__ + '_' + args.model_name + '_' + str(args.seed) + '.csv')

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    print('Using CUDA with', torch.cuda.device_count(), 'GPUs')

# 항상 base_model로 파라미터 접근 일관화
base_model = net.module if isinstance(net, nn.DataParallel) else net

# -----------------------------
# 손실함수 / 옵티마이저
# -----------------------------
# === 손실함수 설정 부분 교체 ===
if args.mode in ['cosface','arcface','amsoftmax']:
    criterion = nn.CrossEntropyLoss()
else:
    if args.weighted_bce_loss:
        pos_weight_tensor, pos_weight_value = _make_pos_weight_tensor(class_counts, use_cuda)
    else:
        pos_weight_tensor = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction='none')

if args.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
elif args.optimizer == 'adamw':
    # AdamW: BN/편향(no_decay) 분리
    decay_params, no_decay_params = [], []
    for name, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):  # BN/LN/편향 등
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
    )

# -----------------------------
# 스케줄러 (cosine / warm restarts / step / none)
# -----------------------------
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

def get_lr(optim):
    for pg in optim.param_groups:
        return pg['lr']

scheduler = None
if args.sched == 'cosine':
    # epoch 단위 cosine (최저 lr은 1e-6로)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)
elif args.sched == 'cosine_wr':
    # warm restarts: 첫 주기 T_0= max(10, epoch//3), 이후 2배씩
    t0 = max(10, max(1, args.epoch // 3))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=2, eta_min=1e-6)
# step 스케줄은 기존 adjust_learning_rate 사용

# -----------------------------
# EMA
# -----------------------------
if args.ema:
    ema_model = ModelEMA(base_model, decay=0.9997)
    if use_cuda:
        ema_model.ema_model.to(torch.device('cuda'))

# W&B
wandb.init(entity="medai-endoscopy", project="RARE-Challenge-jeongchan-cali", name=args.model_name, config=vars(args))
if args.weighted_bce_loss:
    wandb.log({"Train/pos_weight_value": float(pos_weight_value)}, step=0)

best_acc = 0.0
start_epoch = 0

# -----------------------------
# EMA decay 램프업 / BN 재추정
# -----------------------------
global_step = 0
steps_per_epoch = len(trainloader)
# 동적 warmup: 1~2 epoch 범위 내에서 200~2000 사이로 제한
ema_warmup = max(200, min(2000, 2 * steps_per_epoch))

def ema_decay(step, base=0.9997, warmup=ema_warmup):
    return min(base, 1.0 - 1.0/(step+1)) if step < warmup else base

@torch.no_grad()
def bn_recalc(model, loader, num_batches=100):
    """EMA 모델 평가 전 BN running stats 재추정(선택 권장)"""
    was_training = model.training
    model.train()
    dev = next(model.parameters()).device
    for i, (x, *_) in enumerate(loader):
        if i >= num_batches:
            break
        model(x.to(dev))
    if not was_training:
        model.eval()

# -----------------------------
# Mixup 유틸
# -----------------------------
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def denormalize(imgs: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(imgs.device)
    std = STD.to(imgs.device)
    x = imgs * std + mean
    return x.clamp(0, 1)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    b = x.size(0)
    index = torch.randperm(b).cuda() if use_cuda else torch.randperm(b)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def save_mixup_samples(mixed_x, y_a, y_b, lam, outdir="./mixup_samples", n_save=10):
    os.makedirs(outdir, exist_ok=True)
    with torch.no_grad():
        b = min(n_save, mixed_x.size(0))
        imgs = denormalize(mixed_x[:b]).cpu()
        y_a = y_a[:b].long().view(-1).cpu().numpy()
        y_b = y_b[:b].long().view(-1).cpu().numpy()
        for i in range(b):
            fname = f"sample_{i}_lam{lam:.2f}_ya{y_a[i]}_yb{y_b[i]}.jpg"
            save_image(imgs[i], os.path.join(outdir, fname))

def balanced_batch_from_dataset(dataset, class_to_indices, batch_size, device):
    """클래스 균등 샘플링으로 길이가 B인 (x_C, y_C) 배치를 만든다.
       dataset이 (img, label, idx) 또는 (img, label)을 반환해도 동작."""
    xs, ys = [], []
    classes = list(class_to_indices.keys())
    if len(classes) == 0:
        raise RuntimeError("class_to_indices가 비어 있습니다. train_targets를 확인하세요.")

    for _ in range(batch_size):
        c = random.choice(classes)                 # 클래스 균등
        idx = random.choice(class_to_indices[c])   # 해당 클래스에서 임의 인덱스
        sample = dataset[idx]

        # dataset이 (img, label, idx) 또는 (img, label) 반환 모두 대응
        if isinstance(sample, (list, tuple)):
            if len(sample) == 3:
                x, y, _ = sample
            elif len(sample) == 2:
                x, y = sample
            else:
                raise RuntimeError(f"dataset[idx] 형식 이상: len={len(sample)}")
        else:
            raise RuntimeError("dataset[idx]는 (img,label[,idx]) 형태여야 합니다.")

        xs.append(x)
        ys.append(int(y))

    x_tensor = torch.stack(xs, dim=0).to(device)
    y_tensor = torch.tensor(ys, dtype=torch.float32, device=device).unsqueeze(1)
    return x_tensor, y_tensor


def balanced_mixup(x_I, y_I, x_C, y_C, alpha=0.2):
    """Balanced-MixUp: lam~Beta(alpha,1), soft-label로 반환."""
    B = x_I.size(0)
    if alpha > 0:
        lam = torch.distributions.Beta(concentration1=torch.tensor([alpha], device=x_I.device),
                                       concentration0=torch.tensor([1.0], device=x_I.device)).sample((B,))
    else:
        lam = torch.ones(B, device=x_I.device)
    lam_x = lam.view(B,1,1,1)
    lam_y = lam.view(B,1)
    x = lam_x * x_I + (1.0 - lam_x) * x_C
    y = lam_y * y_I + (1.0 - lam_y) * y_C   # BCE soft label
    return x, y, lam



# =======================
# NEW: ALS / LA / Viz
# =======================

import matplotlib.pyplot as plt

def apply_als(y: torch.Tensor, eps_pos: float, eps_neg: float) -> torch.Tensor:
    """
    Asymmetric Label Smoothing:
    - y: soft/hard targets in [0,1], shape [B,1]
    - positive keeps (1-eps_pos), negative lifts by eps_neg
    """
    return y * (1.0 - eps_pos) + (1.0 - y) * eps_neg

def apply_la_logits(outputs: torch.Tensor, la_bias_scalar: float) -> torch.Tensor:
    if la_bias_scalar == 0.0:
        return outputs
    return outputs + torch.tensor(la_bias_scalar, device=outputs.device, dtype=outputs.dtype)

def log_sigmoid_hist_to_wandb(labels_np, scores_np, epoch: int, split_name: str = "Val"):
    pos_scores = scores_np[labels_np == 1]
    neg_scores = scores_np[labels_np == 0]

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6,4))
    plt.hist(neg_scores, bins=50, range=(0,1), alpha=0.6, label='Negative (y=0)')
    plt.hist(pos_scores, bins=50, range=(0,1), alpha=0.6, label='Positive (y=1)')
    plt.legend()
    plt.title(f'{split_name}: Sigmoid score distribution by class')
    plt.xlabel('sigmoid(score)')
    plt.ylabel('count')
    import wandb
    wandb.log({f"Viz/{split_name}/Sigmoid_Hist_Overlay": wandb.Image(fig)}, step=epoch)
    plt.close(fig)

    if len(pos_scores) > 0:
        wandb.log({f"Viz/{split_name}/PosScoresHist": wandb.Histogram(pos_scores)}, step=epoch)
    if len(neg_scores) > 0:
        wandb.log({f"Viz/{split_name}/NegScoresHist": wandb.Histogram(neg_scores)}, step=epoch)

# -----------------------------
# Train / Eval
# -----------------------------
def apply_warmup(optimizer, epoch):
    """선형 워밍업: epoch < warmup_epochs일 때만"""
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        warmup_lr = args.lr * float(epoch + 1) / float(args.warmup_epochs)
        for pg in optimizer.param_groups:
            pg['lr'] = warmup_lr

@torch.no_grad()
def compute_fp_weights(model, loader, num_samples, thresh, gamma, device, mode='bce'):
    model.eval()
    w = torch.ones(num_samples, dtype=torch.float32)
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if mode in ['cosface','arcface','amsoftmax']:
            logits = model(x, y=None)                # [B,2]
            prob1  = torch.softmax(logits, dim=1)[:,1]
        else:
            z = model(x)                             # ← 여기!
            prob1 = torch.sigmoid(z).squeeze(1)
        mask = (y == 0) & (prob1 >= thresh)
        w[idx[mask].cpu()] = gamma
    return w

def train(epoch, fpw_vec=None):
    global global_step
    net.train()
    train_loss, correct, total = 0.0, 0.0, 0

    for batch_idx, (inputs, targets_long, idx) in enumerate(trainloader):
        if use_cuda:
            inputs = inputs.cuda()
            targets_long = targets_long.cuda()
        # BCE용은 float(0/1), CE용은 long(0/1)
        targets_float = targets_long.float().unsqueeze(1)

        optimizer.zero_grad()

        if args.mode in ['cosface','arcface','amsoftmax']:
            # --- margin-softmax + mixup 지원 ---
            if args.mixup == 'none':
                # 기존 경로 (그대로)
                logits2 = net(inputs, y=targets_long)
                loss = criterion(logits2, targets_long)

                # 모니터링용 preds는 margin없는 로짓으로 계산해도 무방
                with torch.no_grad():
                    logits_nomargin = net(inputs, y=None)
                    preds = logits_nomargin.argmax(dim=1)

            elif args.mixup == 'vanilla':
                # (A) vanilla mixup: lam은 scalar
                mixed_x, y_a_long, y_b_long, lam = mixup_class(
                    inputs, targets_long, alpha=args.alpha, use_cuda=use_cuda
                )
                # 두 번 forward: 각 라벨에 margin 적용
                logits_a = net(mixed_x, y=y_a_long)
                logits_b = net(mixed_x, y=y_b_long)

                # per-sample CE 후 lam 가중합
                ce = nn.CrossEntropyLoss(reduction='none')
                loss = (lam * ce(logits_a, y_a_long) + (1.0 - lam) * ce(logits_b, y_b_long)).mean()

                # 모니터링용 정확도: margin 없이 argmax
                with torch.no_grad():
                    logits_nomargin = net(mixed_x, y=None)
                    preds = logits_nomargin.argmax(dim=1)

            elif args.mixup == 'balanced':
                # (B) balanced mixup: lam은 [B] 벡터
                # 1) 클래스 균등으로 보조 배치 샘플
                x_C, y_C = balanced_batch_from_dataset(
                    train_dataset, class_to_indices,
                    batch_size=inputs.size(0), device=inputs.device
                )
                y_C_long = y_C.view(-1).long()

                # 2) 이미지 혼합 및 개별 lam 추출
                # balanced_mixup은 (x, y_soft, lam) 반환 → 여기선 x만 쓰고 lam만 활용
                mixed_x, _y_soft_ignored, lam_vec = balanced_mixup(
                    inputs, targets_long.float().unsqueeze(1), x_C, y_C, alpha=args.alpha
                )
                lam_vec = lam_vec.view(-1)  # [B]

                # 3) 두 번 forward: 각 라벨에 margin 적용
                logits_a = net(mixed_x, y=targets_long)  # 원 배치 라벨
                logits_b = net(mixed_x, y=y_C_long)      # 보조 배치 라벨

                # 4) per-sample CE 후 lam_vec로 가중합
                ce = nn.CrossEntropyLoss(reduction='none')
                loss = (lam_vec * ce(logits_a, targets_long) + (1.0 - lam_vec) * ce(logits_b, y_C_long)).mean()

                # 모니터링용 정확도
                with torch.no_grad():
                    logits_nomargin = net(mixed_x, y=None)
                    preds = logits_nomargin.argmax(dim=1)

            else:
                raise ValueError(f"Unsupported mixup for margin-softmax: {args.mixup}")

            # 공통: 정확도 누적
            correct += (preds == targets_long).sum().item()
            total   += targets_long.numel()

        else:
            # === BCE 파이프라인 (MixUp + OHNM 지원) ===
            use_ohnm = (args.mode == 'ohnm')

            if args.mixup == 'vanilla':
                # vanilla는 y_a, y_b를 soft로 합쳐서 per-sample loss/마스크 계산
                inputs_m, y_a, y_b, lam = mixup_data(inputs, targets_float, args.alpha, use_cuda)
                y_soft = lam * y_a + (1.0 - lam) * y_b            # [B,1] in [0,1]
                outputs = net(inputs_m)
                if args.la:
                    outputs = apply_la_logits(outputs, la_logit_bias)
                if args.als:
                    y_soft = apply_als(y_soft, args.als_pos_eps, args.als_neg_eps)

                raw_loss = criterion(outputs, y_soft)              # [B,1] or [B]

            elif args.mixup == 'balanced':
                x_C, y_C = balanced_batch_from_dataset(
                    train_dataset, class_to_indices,
                    batch_size=inputs.size(0), device=inputs.device
                )
                inputs_m, y_soft, _ = balanced_mixup(inputs, targets_float, x_C, y_C, alpha=args.alpha)  # y_soft:[B,1]
                outputs = net(inputs_m)
                if args.la:
                    outputs = apply_la_logits(outputs, la_logit_bias)
                if args.als:
                    y_soft = apply_als(y_soft, args.als_pos_eps, args.als_neg_eps)
                raw_loss = criterion(outputs, y_soft)              # [B,1] or [B]

            else:
                # mixup 사용 안 함
                outputs = net(inputs)
                if args.la:
                    outputs = apply_la_logits(outputs, la_logit_bias)
                y_soft = targets_float                               # [B,1]
                if args.als:
                    y_soft = apply_als(y_soft, args.als_pos_eps, args.als_neg_eps)
                raw_loss = criterion(outputs, y_soft)                # [B,1] or [B]

            # shape 정리 -> [B]
            if raw_loss.ndim == 2 and raw_loss.size(1) == 1:
                raw_loss = raw_loss.squeeze(1)                      # [B]
            y_soft_vec = y_soft.view(-1)                            # [B]

            if use_ohnm:
                # --- OHNM: pos는 모두 포함, neg는 top-k%만 포함 ---
                with torch.no_grad():
                    probs = torch.sigmoid(outputs).view(-1)         # [B]
                pos_mask = (y_soft_vec >= 0.5)
                neg_mask = (y_soft_vec < 0.5)

                num_neg = int(neg_mask.sum().item())
                if num_neg > 0:
                    k = max(1, int(num_neg * float(args.ohnm_topk)))
                    if args.ohnm_metric == 'prob':
                        neg_scores = probs[neg_mask]                # 확률 높을수록 hard
                    else:  # 'loss'
                        neg_scores = raw_loss[neg_mask]             # loss 클수록 hard
                    topk_vals, topk_idx = torch.topk(neg_scores, k=k, largest=True, sorted=False)

                    selected_neg_mask = torch.zeros_like(neg_mask)
                    neg_indices = torch.nonzero(neg_mask, as_tuple=False).view(-1)
                    selected_neg_mask[neg_indices[topk_idx]] = True
                else:
                    selected_neg_mask = torch.zeros_like(neg_mask)

                selected = pos_mask | selected_neg_mask
                if selected.sum().item() == 0:                      # 안전장치
                    selected = pos_mask if pos_mask.any() else neg_mask
                loss = raw_loss[selected].mean()

            else:
                # --- OHNM이 아니면 FPW(옵션) 또는 평균 ---
                if args.fpw_enable and (fpw_vec is not None):
                    bw = fpw_vec[idx.cpu()].to(outputs.device)      # [B]
                    loss = (raw_loss * bw).mean()
                else:
                    loss = raw_loss.mean()

            # 모니터링용 정확도(0.5) (soft 라벨이어도 threshold는 확률 기반)
            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct += preds.eq(targets_long.view_as(preds)).sum().item()
                total += targets_long.size(0)

        loss.backward()
        optimizer.step()

        # EMA
        if args.ema:
            global_step += 1
            ema_model.decay = ema_decay(global_step)
            ema_model.update(net.module if isinstance(net, nn.DataParallel) else net)

        train_loss += float(loss.item())
        progress_bar(batch_idx, len(trainloader),
                     f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.0*correct/max(1,total):.2f}%')

    wandb.log({
        "Train/Loss": train_loss / (batch_idx + 1),
        "Train/Accuracy": 100.0 * correct / max(1,total),
        "LR": get_lr(optimizer)
    }, step=epoch)

    return (train_loss / (batch_idx + 1), 0.0, 100.0 * correct / max(1,total))

def _platt_fit(scores_np, labels_np, max_iter=100):
    # scores_np: (N,) raw probabilities or logits? → 보편적으로 "logits"가 안정적.
    # 여기선 "logits"로 피팅하겠습니다.
    # logits = log(p/(1-p)) 변환이 필요할 수 있으니, 입력이 prob이면 clamp 후 logit으로 변환:
    eps = 1e-6
    x = np.clip(scores_np, eps, 1-eps)
    logits = np.log(x/(1-x))
    y = labels_np.astype(np.float32)

    A = torch.zeros(1, requires_grad=True)
    B = torch.zeros(1, requires_grad=True)
    optim_pl = torch.optim.LBFGS([A,B], lr=1.0, max_iter=max_iter, line_search_fn='strong_wolfe')

    x_t = torch.from_numpy(logits).float()
    y_t = torch.from_numpy(y).float()

    def closure():
        optim_pl.zero_grad()
        z = A * x_t + B
        p = torch.sigmoid(z)
        # BCE
        loss = F.binary_cross_entropy(p, y_t)
        loss.backward()
        return loss

    optim_pl.step(closure)
    with torch.no_grad():
        return float(A.item()), float(B.item())

def _platt_apply(prob_np, A, B):
    # 학습은 logits 기준이었으므로, 여기서도 prob→logit→변환→prob
    eps = 1e-6
    p = np.clip(prob_np, eps, 1-eps)
    z = np.log(p/(1-p))
    z_cal = A*z + B
    return 1.0 / (1.0 + np.exp(-z_cal))

@torch.no_grad()
def _forward_to_probs_and_logits(model_to_eval, loader):
    """return: probs(np[N]), logits(np[N]), labels(np[N])  (binary positive-class)"""
    model_to_eval.eval()
    if args.ema and args.bn_recalc:
        bn_recalc(model_to_eval, trainloader, num_batches=100)
        model_to_eval.eval()

    all_probs, all_logits, all_labels = [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets, _idx) in enumerate(loader):
            inputs  = inputs.cuda()  if use_cuda else inputs
            targets = targets.cuda() if use_cuda else targets

            if args.mode in ['cosface','arcface','amsoftmax']:
                logits2 = model_to_eval(inputs, y=None)            # [B,2]
                prob1   = torch.softmax(logits2, dim=1)[:,1]       # [B]
                # binary logit as log-odds of class1
                lg1 = torch.log(torch.clamp(prob1, 1e-6, 1-1e-6) / (1-prob1))
                all_logits.append(lg1.detach().cpu().numpy())
                all_probs.append(prob1.detach().cpu().numpy())
            else:
                outputs = model_to_eval(inputs)                    # [B,1] logit
                if args.la:
                    outputs = apply_la_logits(outputs, la_logit_bias)
                prob1 = torch.sigmoid(outputs).squeeze(1)         # [B]
                all_probs.append(prob1.detach().cpu().numpy())
                all_logits.append(outputs.squeeze(1).detach().cpu().numpy())

            all_labels.append(targets.detach().cpu().numpy())

            progress_bar(batch_idx, len(loader), 'Collecting logits/probs..')

    probs  = np.concatenate(all_probs, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(int)
    return probs, logits, labels


# --- split별 히스토그램 로깅 (Val/Test 등) ---
def log_sigmoid_hist_to_wandb(labels_np: np.ndarray, scores_np: np.ndarray, epoch: int, split_name: str = "Val"):
    """
    labels_np: (N,) in {0,1}
    scores_np: (N,) sigmoid probabilities in [0,1]
    split_name: "Val" or "Test" 등
    """
    pos_scores = scores_np[labels_np == 1]
    neg_scores = scores_np[labels_np == 0]

    # Matplotlib overlay figure
    fig = plt.figure(figsize=(6,4))
    plt.hist(neg_scores, bins=50, range=(0,1), alpha=0.6, label='Negative (y=0)')
    plt.hist(pos_scores, bins=50, range=(0,1), alpha=0.6, label='Positive (y=1)')
    plt.legend()
    plt.title(f'[{split_name}] Sigmoid score distribution by class')
    plt.xlabel('sigmoid(score)')
    plt.ylabel('count')
    wandb.log({f"{split_name}/SigmoidHistOverlay": wandb.Image(fig)}, step=epoch)
    plt.close(fig)

    # W&B native histograms
    if len(pos_scores) > 0:
        wandb.log({f"{split_name}/PosScoresHist": wandb.Histogram(pos_scores)}, step=epoch)
    if len(neg_scores) > 0:
        wandb.log({f"{split_name}/NegScoresHist": wandb.Histogram(neg_scores)}, step=epoch)


@torch.no_grad()
def evaluate_and_calibrate(epoch, split_name, loader, fit_calib=False, existing_calib=None):
    """
    split_name: "Val" / "Test" 등
    fit_calib:  True면 이 split에서 캘리브레이션 파라미터 학습
    existing_calib: 주어지면 학습 없이 그걸 적용
    반환값: (F1, calib_dict or None)
    """
    # 1) 모델 선택(EMA 여부 반영) + 확률/로짓/라벨 수집
    model_to_eval = ema_model.ema_model if args.ema else net
    probs, logits, labels = _forward_to_probs_and_logits(model_to_eval, loader)

    # 2) 캘리브레이션 학습/적용
    calib = None
    probs_for_metric = probs.copy()
    if existing_calib is not None:
        probs_for_metric = apply_calibrator(existing_calib, probs_for_metric)
        calib = existing_calib
    elif fit_calib and args.calib != 'none':
        calib = fit_calibrator(args.calib, probs, labels,
                               bbq_bins=args.bbq_bins, iso_oob=args.iso_out_of_bounds)
        probs_for_metric = apply_calibrator(calib, probs_for_metric)

    # 3) 시각화(히스토그램: split_name으로 라우팅)
    log_sigmoid_hist_to_wandb(labels, probs_for_metric, epoch, split_name=split_name)

    # 4) 메트릭 계산 + 로깅
    acc = ( (probs_for_metric > 0.5).astype(int) == labels ).mean() * 100.0
    metric_dict = compute_metrics(labels, probs_for_metric)  # 이미 F1, AUC, ECE 등 포함 가정
    metric_dict.update({f"{split_name}/Accuracy": acc})
    metric_prefixed = {f"{split_name}/{k}": v for k, v in metric_dict.items()}
    wandb.log(metric_prefixed, step=epoch)

    # 5) Confusion Matrix 및 기본 카운트 로깅
    try:
        if labels.size == 0:
            print(f"[WARN] {split_name}: no samples -> skip confusion matrix")
        else:
            preds = (probs_for_metric >= 0.5).astype(int)
            cm = confusion_matrix(labels, preds, labels=[0,1])

            tp = int(cm[1,1]); tn = int(cm[0,0]); fp = int(cm[0,1]); fn = int(cm[1,0])
            wandb.log({
                f"{split_name}/CM/TP": tp,
                f"{split_name}/CM/TN": tn,
                f"{split_name}/CM/FP": fp,
                f"{split_name}/CM/FN": fn,
            }, step=epoch)

            # 1) 당신의 유틸(있다면)
            try:
                log_confusion_matrix_wandb(
                    cm=cm,
                    class_names=["Negative","Positive"],
                    split_name=split_name,
                    step=epoch
                )
            except Exception as e_util:
                print(f"[INFO] custom CM logger failed, fallback to image: {e_util}")

                # 2) 폴백: matplotlib 이미지로 로깅
                import matplotlib.pyplot as plt
                import numpy as np
                fig, ax = plt.subplots(figsize=(3.2, 3.2))
                im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set(xticks=np.arange(2), yticks=np.arange(2),
                       xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"],
                       ylabel='True', xlabel='Pred')
                # 텍스트
                thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")
                ax.set_title(f"{split_name} Confusion Matrix")
                wandb.log({f"{split_name}/ConfusionMatrixImage": wandb.Image(fig)}, step=epoch)
                plt.close(fig)

            # 3) (선택) W&B 인터랙티브 CM
            try:
                wandb.log({
                    f"{split_name}/ConfusionMatrix": wandb.plot.confusion_matrix(
                        y_true=labels.tolist(),
                        preds=preds.tolist(),
                        class_names=["Negative","Positive"]
                    )
                }, step=epoch)
            except Exception as e_wb:
                print(f"[INFO] wandb.plot.confusion_matrix failed: {e_wb}")

    except Exception as e:
        print(f"[WARN] Confusion matrix logging failed: {e}")

    return metric_dict.get("F1", 0.0), calib


def checkpoint(f1, epoch, calib=None):
    print(f"Saving.. (Best VAL F1: {f1:.4f})")
    state = {
        'state_dict': (net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()),
        'ema_state_dict': (ema_model.ema_model.state_dict() if args.ema else None),
        'f1_val': f1,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'calib': calib if calib is not None else {'type':'none'},
        'args': vars(args),                # inference에서 재현용
        'la_logit_bias': float(la_logit_bias),  # LA 상수
        'pos_ratio_train': float(pos_ratio),    # 참고용
    }
    os.makedirs('checkpoint', exist_ok=True)
    torch.save(state, f'./checkpoint/{args.model_name}.pth')

def adjust_learning_rate_step(optimizer, epoch):
    """기존 step 스케줄 (SGD에 주로 적합)"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# -----------------------------
# 디버그: 첫 배치 mixup 샘플 저장
# -----------------------------
# try:
#     inputs0, targets0 = next(iter(trainloader))
#     if use_cuda:
#         inputs0, targets0 = inputs0.cuda(), targets0.cuda()
#     targets0 = targets0.float().unsqueeze(1)

#     if args.mixup == 'vanilla':
#         mixed_x0, y_a0, y_b0, lam0 = mixup_data(inputs0, targets0, args.alpha, use_cuda)
#         save_mixup_samples(mixed_x0, y_a0, y_b0, lam0, outdir="./mixup_samples", n_save=10)
#     elif args.mixup == 'balanced':
#         xC0, yC0 = balanced_batch_from_dataset(train_dataset, class_to_indices, inputs0.size(0), inputs0.device)
#         mixed_x0, y_soft0, lamv0 = balanced_mixup(inputs0, targets0, xC0, yC0, alpha=args.alpha)
#         # 파일명에 soft 라벨을 모두 기록하기 어렵기 때문에 lam 평균만 부여
#         save_mixup_samples(mixed_x0, (y_soft0>0.5).long(), (y_soft0<=0.5).long(),
#                            lam=float(lamv0.mean().item()),
#                            outdir="./mixup_samples", n_save=10)
# except Exception as e:
#     print(f"[WARN] mixup sample save skipped: {e}")


# -----------------------------
# 로그 파일 헤더
# -----------------------------
if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc', 'test loss', 'test acc'])

# -----------------------------
# 학습 루프
# -----------------------------
best_f1 = 0.0
best_acc = 0.0
best_calib = None
start_epoch = 0

for epoch in range(start_epoch, args.epoch):
    print(f"Epoch [{epoch+1}/{args.epoch}]")

    # (E1) FP-weighting 벡터 갱신
    if args.fpw_enable and epoch >= args.fpw_warmup_epochs:
        model_to_eval = ema_model.ema_model if args.ema else net
        device = torch.device('cuda' if use_cuda else 'cpu')
        FPW_VEC = compute_fp_weights(
            model=model_to_eval, loader=train_eval_loader,
            num_samples=underlying_len(train_dataset),   # <<< 여기!
            thresh=args.fpw_thresh, gamma=args.fpw_gamma,
            device=device, mode=args.mode
        )
        wandb.log({"FPW/mean": float(FPW_VEC.mean().item()),
                   "FPW/max": float(FPW_VEC.max().item())}, step=epoch)

    apply_warmup(optimizer, epoch)
    train_loss, reg_loss, train_acc = train(epoch, FPW_VEC)

    # ---- VAL: fit calibration here ----
    f1_val, calib_now = evaluate_and_calibrate(
        epoch, split_name="Val", loader=valloader, fit_calib=True
    )

    # ---- TEST: apply the same calibration learned on VAL ----
    if calib_now is not None:
        # 캘리브레이션 파라미터 로깅
        wandb.log({f"Calib/method": args.calib}, step=epoch)
        for k,v in calib_now.items():
            if k == 'type': continue
            if isinstance(v, (int,float)):
                wandb.log({f"Calib/param_{k}": v}, step=epoch)
    f1_test, _ = evaluate_and_calibrate(
        epoch, split_name="Test", loader=testloader, fit_calib=False, existing_calib=calib_now
    )


    # 스케줄러
    if args.sched == 'step':
        adjust_learning_rate_step(optimizer, epoch)
    elif args.sched in ['cosine', 'cosine_wr']:
        scheduler.step()

    # CSV 로그
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc,
                            # 아래 두 값은 편의상 마지막 evaluate의 Loss가 아니라 F1과 Acc대신 F1/Acc 넣고 싶다면 수정 가능
                            f1_val, f1_test])

    # 최고 VAL F1 갱신 시 체크포인트 저장 (캘리브레이션 파라미터 포함)
    if f1_val > best_f1:
        best_f1 = f1_val
        best_calib = calib_now
        checkpoint(f1_val, epoch, calib=best_calib)
