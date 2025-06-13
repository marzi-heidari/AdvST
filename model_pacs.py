from __future__ import print_function, absolute_import, division

import os

from transformers import LlavaProcessor, LlavaForConditionalGeneration


from clip import clip
from transformers import AutoModelForCausalLM, AutoTokenizer

from minigpt4.common import registry
from minigpt4.common.config import Config

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import kornia
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms


from torch.nn import DataParallel
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from common.pacs import PACS, PACSMultiple
from common.utils import (
    fix_all_seed,
    write_log,
    compute_accuracy,
    Averager,
)
from config import PACS_DATA_FOLDER
from models.resnet_vanilla import resnet18

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# https://github.com/HAHA-DL/Episodic-DG
def bn_eval(model):
    for name, param in model.named_parameters():
        if 'conv1' in name or 'bn1' in name:
            param.requires_grad = False

    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            # m.requires_grad_(False)
            # pass


# https://github.com/mil-tokyo/dg_mmld/blob/aef26b2745beabc6356accd183ff3e17f71657ce/util/scheduler.py
from torch.optim.lr_scheduler import _LRScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features_q, features_k):
        """
        features_q: tensor of shape (batch_size, feature_dim)
        features_k: tensor of shape (batch_size, feature_dim)
        Positive pairs are (features_q[i], features_k[i]).
        """
        batch_size = features_q.shape[0]

        # Normalize the features
        features_q = F.normalize(features_q, dim=1)
        features_k = F.normalize(features_k, dim=1)

        # Compute logits: (batch_size, batch_size)
        logits = torch.matmul(features_q, features_k.T) / self.temperature

        # Targets: diagonal are positives
        labels = torch.arange(batch_size, device=features_q.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss


class inv_lr_scheduler(_LRScheduler):
    def __init__(self, optimizer, alpha, beta, total_epoch, last_epoch=-1):
        self.alpha = alpha
        self.beta = beta
        self.total_epoch = total_epoch
        super(inv_lr_scheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr
            * ((1 + self.alpha * self.last_epoch / self.total_epoch) ** (-self.beta))
            for base_lr in self.base_lrs
        ]


class DataPool:
    def __init__(self, pool_size):
        self.data = [[]] * pool_size
        self.pool_size = pool_size
        self.count = 0
        self.num = 0

    def add(self, x):
        if self.count < self.pool_size:
            self.data[self.count] = x
            self.count += 1
        else:
            self.count = 0
            self.data[self.count] = x
            self.count += 1
        if self.num < self.pool_size:
            self.num += 1

    def get(self, num=-1):
        if self.num == 0:
            return []
        if num < 0:
            return self.data[0: self.num]
        else:
            num = min(num, self.num)
            indexes = list(range(self.num))
            random.shuffle(indexes)
            sel_indexes = indexes[0:num]
            return [self.data[i] for i in sel_indexes]


def hsv_aug(x, hsv):
    rgb2hsv = kornia.color.RgbToHsv()
    hsv2rgb = kornia.color.HsvToRgb()
    B = x.shape[0]
    hsv_img = rgb2hsv(x) + hsv.view(B, 3, 1, 1)
    rgb_img = hsv2rgb(hsv_img)
    return torch.clamp(rgb_img, -10, 10)


def rotate_aug(x, angle):
    rgb_img = kornia.geometry.transform.rotate(x, torch.clamp(angle, 0.01, 1) * 360)
    return rgb_img


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        """
        features: (N, D)
        labels: (N,)
        """
        centers_batch = self.centers[labels]
        return ((features - centers_batch) ** 2).sum(dim=1).mean()


def posterize_aug(x, factor):
    bits = torch.randint(0, 9, size=(len(x),)).to(x.device)
    nx = kornia.enhance.posterize(x, bits)
    return nx.detach() + x - x.detach()


def cutout(img_batch, num_holes, hole_size, fill_value=0):
    img_batch = img_batch.clone()
    B = len(img_batch)
    height, width = img_batch.shape[-2:]
    masks = torch.zeros_like(img_batch)
    for _n in range(num_holes):
        if height == hole_size:
            y1 = torch.tensor([0])
        else:
            y1 = torch.randint(0, height - hole_size, (1,))
        if width == hole_size:
            x1 = torch.tensor([0])
        else:
            x1 = torch.randint(0, width - hole_size, (1,))
        y2 = y1 + hole_size
        x2 = x1 + hole_size
        masks[:, :, y1:y2, x1:x2] = 1.0
        # img_batch[:, :, y1:y2, x1:x2] = fill_value
    img_batch = (1.0 - masks) * img_batch + masks * fill_value.view(B, 3, 1, 1)
    return img_batch


def cutout_fixed_num_holes(x, factor, num_holes=8, image_shape=(84, 84)):
    height, width = image_shape
    min_size = min(height, width)
    hole_size = max(int(min_size * 0.2), 1)
    return cutout(x, num_holes=num_holes, hole_size=hole_size, fill_value=factor)


class SemanticAugment(nn.Module):
    def __init__(self, batch_size, op_tuples, op_label):
        super(SemanticAugment, self).__init__()
        self.ops = [op[0] for op in op_tuples]
        self.op_label = op_label
        params = []
        for tup in op_tuples:
            min_val = tup[1][0]
            max_val = tup[1][1]
            num = tup[2]
            init_val = torch.rand(batch_size, num) * (max_val - min_val) + min_val
            init_val = init_val.squeeze(1)
            params.append(torch.nn.Parameter(init_val))
        self.params = nn.ParameterList(params)

    def forward(self, x):
        for i, op in enumerate(self.ops):
            x = torch.clamp(op(x, self.params[i]), 0, 1)
        return x


class Counter:
    def __init__(self):
        self.v = 0
        self.c = 0

    def add(self, x):
        self.v += x
        self.c += 1

    def avg(self):
        if self.c == 0:
            return 0
        return self.v / self.c


class ModelBaseline(object):
    def __init__(self, flags):
        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)

    def get_images(
            self, images, labels, save_path, shuffle=False, sel_indexes=None, nsamples=10
    ):
        class_dict = {}
        for i, l in enumerate(labels):
            if class_dict.get(l, None) is None:
                class_dict[l] = [images[i]]
            else:
                class_dict[l].append(images[i])
        num_classes = len(class_dict)
        total_num_per_class = np.array([len(class_dict[i]) for i in class_dict]).min()
        nsamples = min(nsamples, total_num_per_class)
        indexes = list(range(total_num_per_class))
        if shuffle:
            random.shuffle(indexes)
        if sel_indexes is None:
            sel_indexes = np.array(indexes[0:nsamples])
        else:
            assert nsamples >= len(sel_indexes), "sel_indexes too long"
        data_matrix = []
        keys = sorted(list(class_dict.keys()))
        for c in keys:
            data_matrix.append(np.array(class_dict[c])[sel_indexes])
        data_matrix = np.concatenate(data_matrix, axis=0)
        self.vis_image(data_matrix, nsamples, save_path)
        return sel_indexes

    def vis_image(self, data, max_per_row=10, save_path="./"):
        num = len(data)
        nrow = int(np.ceil(num / max_per_row))

        fig, ax = plt.subplots(figsize=(max_per_row, nrow))
        demo = []
        for i in range(len(data)):
            demo.append(torch.tensor(data[i]))
        demo = torch.stack(demo)

        grid_img = torchvision.utils.make_grid(demo, nrow=max_per_row)
        grid_img = grid_img.permute(1, 2, 0).detach().cpu().numpy()
        ax.imshow(grid_img, interpolation="nearest")
        ax.axis("off")
        fig.savefig(save_path)
        plt.close(fig)

    def setup(self, flags):
        torch.backends.cudnn.deterministic = flags.deterministic
        print("torch.backends.cudnn.deterministic:", torch.backends.cudnn.deterministic)
        fix_all_seed(flags.seed)
        if flags.model == "resnet18":
            self.network = resnet18(
                pretrained=True,
                num_classes=flags.num_classes,
                contrastive=flags.train_mode,
            )
        self.network = DataParallel(self.network)
        self.network = self.network.cuda()

        print(self.network)
        print("flags:", flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)
        flag_str = (
                "--------Parameters--------\n"
                + "\n".join(["{}={}".format(k, flags.__dict__[k]) for k in flags.__dict__])
                + "\n--------------------"
        )
        print("flags:", flag_str)
        flags_log = os.path.join(flags.logs, "flags_log.txt")
        write_log(flag_str, flags_log)

    def whitening_loss(self, feat):
        C = torch.cov(feat)
        C_diag = torch.diag(C, 0)
        diag_ = 0.5 * (torch.norm(C, 'fro') ** 2 - torch.norm(C_diag, 2) ** 2)
        return diag_

    def setup_path(self, flags):
        root_folder = PACS_DATA_FOLDER
        dataset_names = ["art_painting", "cartoon", "photo", "sketch"]
        seen_index = flags.seen_index
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(288),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.train_transform = transforms.Compose([
            transforms.Resize(288),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0),
            # transforms.RandomAffine(0, shear=20, fill=128),
            # transforms.RandomPosterize(bits=3, p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        if type(seen_index) == list:
            names = [dataset_names[i] for i in seen_index]
            self.train_name = "+".join(names)
            self.train_dataset = PACSMultiple(root_folder, names, "train")
            self.val_dataset = PACSMultiple(root_folder, names, "val")
            self.test_loaders = []
            for index, name in enumerate(dataset_names):
                if index not in seen_index:
                    dataset = PACS(root_folder, name, "test")
                    loader = DataLoader(
                        dataset,
                        batch_size=flags.batch_size,
                        shuffle=False,
                        num_workers=flags.num_workers,
                        pin_memory=False,
                    )
                    self.test_loaders.append((name, loader))

        else:
            self.train_dataset = PACS(
                root_folder, dataset_names[seen_index], "train", ratio=flags.ratio
            )
            # if flags.algorithm == "ERM":
            self.train_dataset.transform = self.preprocess
            self.val_dataset = PACS(root_folder, dataset_names[seen_index], "val")
            self.test_loaders = []
            self.train_name = dataset_names[seen_index]
            for index, name in enumerate(dataset_names):
                if index != seen_index:
                    dataset = PACS(root_folder, name, "test")
                    loader = DataLoader(
                        dataset,
                        batch_size=flags.batch_size,
                        shuffle=False,
                        num_workers=flags.num_workers,
                        pin_memory=False
                    )

                    self.test_loaders.append((name, loader))

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=flags.batch_size,
            shuffle=True,
            num_workers=flags.num_workers,
            pin_memory=False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=flags.batch_size,
            shuffle=False,
            num_workers=flags.num_workers,
            pin_memory=False,
        )

    def configure(self, flags):
        for name, param in self.network.named_parameters():
            print(name, param.size())
        parameter_list = []
        # classifier_param = list(map(id, self.network.fc.parameters()))
        # backbone_param = filter(lambda p: id(p) not in classifier_param and p.requires_grad, self.network.parameters())
        # parameter_list.append({'params': backbone_param, 'lr': flags.lr})
        parameter_list.append({'params': self.network.parameters(), 'lr': flags.lr})
        self.optimizer = torch.optim.SGD(
            parameter_list,
            weight_decay=flags.weight_decay,
            momentum=flags.momentum,
            nesterov=True,
        )
        # self.optimizer = torch.optim.AdamW(parameter_list, lr=3e-4, weight_decay=1e-2)
        # self.optimizer = Lookahead(self.optimizer)

        if flags.model == "resnet18":
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, flags.train_epochs * len(self.train_loader)
            )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.cnt_loss = InfoNCELoss()
        self.loss_per_ele = torch.nn.CrossEntropyLoss(reduction="none")

    def save_model(self, file_name, flags):
        if not os.path.exists(flags.model_path):
            os.makedirs(flags.model_path)
        outfile = os.path.join(flags.model_path, file_name)
        torch.save({"state": self.network.state_dict(), "args": flags}, outfile)

    def entropy_loss(self, x):
        out = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        out = -1.0 * out.sum(dim=1)
        return out.mean()

    def standardize(self, x, dim=0, eps=1e-6):
        """
        Standardize input tensor along the specified dimension.
        Args:
            x: input tensor
            dim: dimension along which to compute mean and std
            eps: small constant to avoid division by zero
        Returns:
            standardized tensor
        """
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True)
        return (x - mean) / (std + eps)

    def projection_3(self, Z, ph_t):
        ph_t = ph_t.to(Z.dtype)
        ndot = (ph_t @ ph_t.transpose(-2, -1))[0][0]
        P = (ph_t.transpose(-2, -1) @ ph_t) / ndot
        return Z @ P
        # return F.normalize(Z @ P, p=2, dim=1) * Z.norm(p=2, dim=1).mean()

    def orthogonality_loss(self, features, labels):
        """
        Encourages orthogonality between class prototypes.

        Args:
            features: (N, D) tensor
            labels: (N,) tensor
        Returns:
            loss: scalar
        """
        unique_labels = torch.unique(labels)
        class_prototypes = []

        for lbl in unique_labels:
            mask = labels == lbl
            if mask.sum() < 2:
                continue
            class_feat = features[mask]
            proto = class_feat.mean(dim=0)
            proto = F.normalize(proto, dim=0)
            class_prototypes.append(proto)

        if len(class_prototypes) < 2:
            return torch.tensor(0.0, device=features.device)

        prototypes = torch.stack(class_prototypes)  # (C, D)
        sim_matrix = torch.matmul(prototypes, prototypes.T)  # (C, C)
        identity = torch.eye(sim_matrix.size(0), device=features.device)
        off_diag = sim_matrix - identity  # zero out diagonals

        return (off_diag ** 2).mean()

    def projection2_V(self, Z, phi_t, energy_threshold=0.5):
        """
        Implements Projection 1.
        Z: (n, d) tensor of backbone features.
        phi_s: (d,) tensor for the source domain.
        phi_t: (d,) tensor for the target domain.

        Computes:
            Omega_t = Z * diag(phi_t)
            Omega_s = Z * diag(phi_s)
        then SVD on each and
            Ẑ_proj = U_t U_sᵀ Z V_s V_tᵀ
        """
        B, d = Z.shape  # Batch size, feature dimension

        # Ensure phi_s and phi_t are broadcastable
        # phi_t = phi_t.view(1, d)  # Reshape to (1, d) for broadcasting
        # phi_s = phi_s.view(1, d)

        # D_phi_t = torch.diag(phi_t).to(Z.dtype)
        mu_z = Z.mean(dim=0, keepdim=True)
        sigma_z = Z.std(dim=0, unbiased=False, keepdim=False) + 1e-6  # prevent division by zero
        Z_norm = (Z - mu_z)
        # @ torch.diag(1 / sigma_z).to(Z.device))
        D_phi = torch.diag(phi_t).to(Z.dtype)
        Omega_t = Z_norm + phi_t.to(Z.dtype)

        # # # Compute Omega matrices: (n,d) = (n,d) @ (d,d)
        # Omega_t = Z @ D_phi_t
        # Omega_t = self.standardize(Omega_t, dim=1)

        # Compute SVD
        U_t, S_t, Vh_t = torch.linalg.svd(Omega_t, full_matrices=False)  # (B, d, d)

        # Extract top k_t singular vectors from target and bottom k_s singular vectors from source
        V_top = Vh_t.transpose(-2, -1)[..., :int(0.1 * 512)]  # Top k_t singular vectors (B, d, k_t)
        # V_bottom = Vh_s.transpose(-2, -1)[..., -k:]  # Bottom k_s singular vectors (B, d, k_s)

        # Compute projection matrices
        P_t = V_top @ V_top.transpose(-2, -1)  # Shape: (B, d, d)

        # Compute projected feature matrix
        z_pos = Z @ P_t

        Z_proj = z_pos
        # loss += 0.01

        return (F.normalize(Z_proj, p=2, dim=-1) * Z.norm(p=2, dim=1).mean()), S_t

    ###############################
    # Projection 1 Function
    ###############################
    def projection1_b(self, Z, phi_s, phi_t):
        """
        Implements Projection 1.
        Z: (n, d) tensor of backbone features.
        phi_s: (d,) tensor for the source domain.
        phi_t: (d,) tensor for the target domain.

        Computes:
            Omega_t = Z * diag(phi_t)
            Omega_s = Z * diag(phi_s)
        then SVD on each and
            Ẑ_proj = U_t U_sᵀ Z V_s V_tᵀ
        """
        # # # Create explicit diagonal matrices from phi vectors
        D_phi_t = torch.diag(phi_t).to(Z.dtype)
        D_phi_s = torch.diag(phi_s).to(Z.dtype)
        # #
        # # # Compute Omega matrices: (n,d) = (n,d) @ (d,d)
        Omega_t = Z @ D_phi_t
        Omega_s = Z @ D_phi_s
        # # Create explicit diagonal matrices from phi vectors

        #
        # # Compute Omega matrices: (n,d) = (n,d) @ (d,d)
        # Omega_t = self.elementwise_pearson_corr(phi_t, Z)
        # Omega_s = self.elementwise_pearson_corr(phi_s, Z)
        # Create diagonal matrix from u
        # diag_u = torch.diag(u)  # Shape: (n, n)
        # D_phi_t = torch.diag(phi_t).to(Z.dtype)
        # D_phi_s = torch.diag(phi_s).to(Z.dtype)
        #
        # # Compute the outer product v^T v for each batch element
        # outer_products = torch.bmm(Z.unsqueeze(2), Z.unsqueeze(1))  # Shape: (B, n, n)
        #
        # # Add the diagonal matrix to each batch element
        # Omega_t = D_phi_t + outer_products  # Broadcasting over batch
        # Omega_s = D_phi_s + outer_products  # Broadcasting over batch

        # Compute SVD on Omega_t and Omega_s (reduced SVD)
        U_t, _, Vh_t = torch.linalg.svd(Omega_t, full_matrices=False)
        U_s, _, Vh_s = torch.linalg.svd(Omega_s, full_matrices=False)

        # Get V matrices (transpose of Vh)
        k = int(0.5 * 512)
        V_t = Vh_t.transpose(-2, -1)
        V_s = Vh_s.transpose(-2, -1)
        V_t = V_t[..., :k]
        V_s = V_s[..., :k]
        U_t = U_t[..., :k]
        U_s = U_s[..., :k]

        # V_t = V_t.transpose(-2, -1)
        # V_s = V_s

        # Compute the projection: Ẑ_proj = U_t U_sᵀ Z V_s V_tᵀ
        hat_Z = U_t @ ((U_s.transpose(0, 1) @ Z) @ V_s) @ V_t.transpose(0, 1)
        # hat_Z = Z @ V_s @ V_t.transpose(0, 1)

        return F.normalize(hat_Z, p=2, dim=-1) * Z.norm(p=2, dim=1).mean() + Z

    def projection1(self, Z, phi_s, phi_t):
        """
        Implements Projection 1.
        Z: (n, d) tensor of backbone features.
        phi_s: (d,) tensor for the source domain.
        phi_t: (d,) tensor for the target domain.

        Computes:
            Omega_t = Z * diag(phi_t)
            Omega_s = Z * diag(phi_s)
        then SVD on each and
            Ẑ_proj = U_t U_sᵀ Z V_s V_tᵀ
        """
        # # # Create explicit diagonal matrices from phi vectors
        D_phi_t = torch.diag(phi_t).to(Z.dtype)
        D_phi_s = torch.diag(phi_s).to(Z.dtype)
        # #
        # # # Compute Omega matrices: (n,d) = (n,d) @ (d,d)
        Omega_t = Z @ D_phi_t
        Omega_s = Z @ D_phi_s
        # # Create explicit diagonal matrices from phi vectors

        #
        # # Compute Omega matrices: (n,d) = (n,d) @ (d,d)
        # Omega_t = self.elementwise_pearson_corr(phi_t, Z)
        # Omega_s = self.elementwise_pearson_corr(phi_s, Z)
        # Create diagonal matrix from u
        # diag_u = torch.diag(u)  # Shape: (n, n)
        # D_phi_t = torch.diag(phi_t).to(Z.dtype)
        # D_phi_s = torch.diag(phi_s).to(Z.dtype)
        #
        # # Compute the outer product v^T v for each batch element
        # outer_products = torch.bmm(Z.unsqueeze(2), Z.unsqueeze(1))  # Shape: (B, n, n)
        #
        # # Add the diagonal matrix to each batch element
        # Omega_t = D_phi_t + outer_products  # Broadcasting over batch
        # Omega_s = D_phi_s + outer_products  # Broadcasting over batch

        # Compute SVD on Omega_t and Omega_s (reduced SVD)
        U_t, _, Vh_t = torch.linalg.svd(Omega_t, full_matrices=False)
        U_s, _, Vh_s = torch.linalg.svd(Omega_s, full_matrices=False)

        # Get V matrices (transpose of Vh)
        k = int(0.5 * 512)
        V_t = Vh_t.transpose(-2, -1)
        V_s = Vh_s.transpose(-2, -1)
        V_t = V_t[..., :k]
        V_s = V_s[..., :k]

        # V_t = V_t.transpose(-2, -1)
        # V_s = V_s

        # Compute the projection: Ẑ_proj = U_t U_sᵀ Z V_s V_tᵀ
        # hat_Z = U_t @ ((U_s.transpose(0, 1) @ Z) @ V_s) @ V_t.transpose(0, 1)
        hat_Z = Z @ V_s @ V_t.transpose(0, 1)

        return F.normalize(hat_Z, p=2, dim=-1) * Z.norm(p=2, dim=1).mean() + Z

    def alignment_loss(self, edited_Z, phi_s, phi_t, tau):
        """
        Alignment loss using cosine similarity.
        For each sample, compute:
            L_align = -log( exp(cos(Ẑ,phi_t)/τ) / (exp(cos(Ẑ,phi_t)/τ) + exp(cos(Ẑ,phi_s)/τ) ) )
        """

        # Compute similarities
        # Normalize vectors to ensure cosine similarity calculation
        phi_t = F.normalize(phi_t, dim=0)
        phi_s = F.normalize(phi_s, dim=0)
        edited_Z = F.normalize(edited_Z, dim=1)
        sim_t = torch.matmul(edited_Z, phi_t.to(edited_Z.dtype)) / tau  # Positive similarities
        sim_s = torch.matmul(edited_Z, phi_s.to(edited_Z.dtype)) / tau  # Negative similarities

        # Compute loss
        numerator = torch.exp(sim_t)  # exp(sim(phi_t, z))
        denominator = numerator + torch.exp(sim_s).sum(dim=0)  # exp(sim(phi_t, z)) + sum(exp(sim(phi_s, z)))

        loss = -torch.mean(torch.log(numerator / denominator))

        return loss.mean()

    def cosine_loss_with_temperature(self, student, teacher, temperature=1.0):
        student = F.normalize(student, dim=-1)
        teacher = F.normalize(teacher, dim=-1)
        cosine_sim = (student * teacher).sum(dim=-1)
        loss = 1 - cosine_sim / temperature
        return loss.mean()

    def prototype_alignment_loss(self, features_clip, features_resnet, labels):
        """
        Aligns class-wise prototypes (centroids) between CLIP and ResNet.

        Args:
            features_clip: (N, D) tensor
            features_resnet: (N, D) tensor
            labels: (N,) tensor of ints
        Returns:
            loss: scalar tensor
        """
        unique_labels = torch.unique(labels)
        loss = 0.0
        for label in unique_labels:
            mask = labels == label
            clip_proto = features_clip[mask].mean(dim=1)
            resnet_proto = features_resnet[mask].mean(dim=1)
            loss += F.mse_loss(resnet_proto, clip_proto)
        return loss.mean()

    def train(self, flags):
        os.makedirs(flags.model_path, exist_ok=True)
        best_val_acc = -1
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
        # model, preprocess = clip.load("ViT-B/32")
        model_id = "llava-hf/llava-1.5-7b-hf"
        processor = LlavaProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)

        # Move to DataParallel on multiple GPUs
        model = torch.nn.DataParallel(model).cuda()

        # New BLIP loading
        # processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        # model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").cuda()
        model.eval()

        # t_s = clip.tokenize(
        #     [
        #         "A higdh-resolution, lifelike image of an object captured with a camera, displaying realistic colors, shadows, and textures."]).to(
        #     f'cuda:{0}')
        # t_t = clip.tokenize(
        #     [
        #         "Artistic renditions of objects and scenes, often with visible brushstrokes, varied textures, and abstract or exaggerated forms"]).to(
        #     f'cuda:{0}')
        # t_t = clip.tokenize(
        #     [
        #         "A colorful and stylized cartoon drawing of an object, with bold outlines, simplified details, and exaggerated proportions in a comic or animated style.-["]).to(
        #     f'cuda:{0}')
        t_s = clip.tokenize(
            ["Realistic"]).to(
            f'cuda:{0}')
        t_t = clip.tokenize(
            ["Sketch drawing"]).to(
            f'cuda:{0}')
        # ----------- Encode Text -----------

        # Encode text
        text = 'realistic'
        text_inputs = processor(text=text, return_tensors="pt").to("cuda", torch.float16)


        # text_inputs = processor(text=["Realistic"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            # text_outputs = model.text_encoder(**text_inputs)
            # phi_s = text_outputs.last_hidden_state[:, 0, :]  # CLS token

            # t_t = clip.tokenize(["Sketch drawing"]).to(f'cuda:{0}')
            # phi_t = model.encode_text(t_t)

            # text_inputs = processor(text=["Sketch drawing"], return_tensors="pt").to("cuda")
            #
            # text_outputs = model.text_encoder(**text_inputs)
            # phi_t = text_outputs.last_hidden_state[:, 0, :]  # CLS token
            # phi_s = model.encode_text(t_s)
            # phi_t = model.encode_text(t_t)
            text_outputs = model.language_model.model.embed_tokens(text_inputs["input_ids"])
            # Shape: [batch_size, seq_len, hidden_dim]
            phi_s = text_outputs.mean(dim=1)  # Simple average pooling
            phi_t = text_outputs.mean(dim=1)  # Simple average pooling



        for epoch in range(flags.train_epochs):
            loss_avger = Averager()
            self.network.train()
            bn_eval(self.network)

            for images_train, images_train_org, labels_train, _ in self.train_loader:
                inputs, labels = images_train.cuda(), labels_train.cuda()
                images_train_org = images_train_org.cuda()
                o, out = self.network(x=inputs)
                o1, out1 = self.network(x=images_train_org)
                Z = out['Embedding']
                with torch.no_grad():
                    # Z_clip = model.encode_image(inputs)
                    Z_clip = vis_processor(inputs)
                    # img_inputs = processor(images=images_train_org, return_tensors="pt", do_rescale=False).to("cuda")
                    # vision_outputs = model.vision_model(**img_inputs)
                    # Z_clip = vision_outputs.last_hidden_state[:, 0, :]  # CLS token
                #
                proj1, s1 = self.projection2_V(Z, phi_t.squeeze(0))
                proj2, s2 = self.projection2_V(out1['Embedding'], phi_t.squeeze(0))
                # proj1 = F.dropout(proj1, p=0.4, training=self.network.training)
                outputs = self.network(0.8 * proj1 + 0.2 * Z, classifier=True)
                alpha = 2.0

                # Sample lambda from Beta distribution
                lam = torch.distributions.Beta(alpha, alpha).sample().item()

                # Ensure lambda is on the same device as features
                # lam = lam.to(Z.device)
                # Shuffle Z_prime along the batch dimension
                indices = torch.randperm(Z.size(0))
                # Z_prime_shuffled = Z[indices]

                # Perform Mixup
                proj1 = (proj1 + proj2) / 2
                labels_one_hot = F.one_hot(labels, flags.num_classes).to(Z.dtype)
                Z_mix = lam * proj1 + (1 - lam) * Z[indices]
                Y_mix = lam * labels_one_hot + (1 - lam) * labels_one_hot[indices]
                logit_mix = self.network(Z_mix, classifier=True)
                log_probs = torch.log_softmax(logit_mix + 1e-6, dim=1)
                #
                # # Gather log probabilities corresponding to true labels
                loss_mixup = -(Y_mix * log_probs).sum(dim=1).mean()
                # phi_t_mat = phi_t.repeat(Z.size(0), 1).to(Z.dtype)  # Shape: (B, d)
                # phi_s_mat = phi_s.repeat(Z.size(0), 1).to(Z.dtype)  #
                Z_norm = Z / Z.norm(dim=1, keepdim=True)
                Z_clip = Z_clip / Z_clip.norm(dim=1, keepdim=True)
                Z_clip = Z_clip.to(Z.dtype)
                align_loss = F.mse_loss(Z_norm, Z_clip.to(Z.dtype)).mean()
                # proj1 = proj1.reshape(inputs.shape[0], -1, proj1.shape[-1])

                loss = self.loss_fn(outputs,
                                    labels) + loss_mixup + 0.1 * F.mse_loss(proj1, Z) + 0.1 * align_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizer.step()
                loss_avger.add(loss.item(), len(labels_train))
                if flags.model == "resnet18":
                    self.scheduler.step()

            flags_log = os.path.join(flags.logs, "loss_log.txt")

            val_acc = self.batch_test(self.val_loader, phi_t)
            msg = "[epoch {}] loss {:.4f}, lr {:.4f}, val_acc {:.4f}".format(
                epoch + 1, loss_avger.item(), self.scheduler.get_last_lr()[0], val_acc
            )
            acc_arr = self.batch_test_workflow(phi_s, phi_s)
            mean_acc = np.array(acc_arr).mean()
            names = [n for n, _ in self.test_loaders]
            res = (
                    "\n{} ".format(self.train_name)
                    + " ".join(["{}:{:.6f}".format(n, a) for n, a in zip(names, acc_arr)])
                    + " Mean:{:.6f}".format(mean_acc)
            )
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                msg += " (best)"
                self.save_model("best_model_{}.tar".format(self.train_name), flags)
            msg += res
            print(msg)
            write_log(msg, flags_log)
        self.save_model("latest_model_{}.tar".format(self.train_name), flags)

    # def test_workflow(self, flags, tag=None):
    #     accuracies = []
    #     if tag is not None:
    #         log_dir = os.path.join(flags.logs, tag)
    #     else:
    #         log_dir = flags.logs
    #     for name, loader in enumerate(self.test_loaders):
    #         accuracy_test = self.test(
    #             loader, log_dir=flags.logs, log_prefix="test_{}".format(name)
    #         )
    #         accuracies.append(accuracy_test)
    #
    #     mean_acc = np.mean(accuracies)
    #     f = open(os.path.join(flags.logs, "acc_test.txt"), mode="a")
    #     f.write("test accuracy:{}\n".format(mean_acc))
    #     f.close()

    # def test(self, test_loader, log_prefix, log_dir="logs/"):
    #     # switch on the network test mode
    #     self.network.eval()
    #     predictions = []
    #     labels = []
    #     with torch.no_grad():
    #         for images_test, labels_test, _ in tqdm(
    #                 test_loader, leave=False, desc="test"
    #         ):
    #             images_test, labels_test = images_test.cuda(), labels_test.cuda()
    #             out, _ = self.network(images_test)
    #             predictions.append(torch.argmax(out, -1).detach().cpu())
    #             labels.append(labels_test.detach().cpu())
    #         predictions = torch.cat(predictions)
    #         labels = torch.cat(labels)
    #     accuracy = compute_accuracy(predictions=predictions, labels=labels)
    #
    #     if not os.path.exists(log_dir):
    #         os.makedirs(log_dir)
    #
    #     f = open(os.path.join(log_dir, "{}.txt".format(log_prefix)), mode="a")
    #     f.write("test accuracy:{}\n".format(accuracy))
    #     f.close()
    #     return accuracy

    def batch_test(self, ood_loader, phi_t):
        # switch on the network test mode
        self.network.eval()
        test_image_preds = []
        test_labels = []
        with torch.no_grad():
            for images_test, _, labels_test, _ in ood_loader:
                # if images_test.size(0) < 512:
                #     break
                images_test, labels_test = images_test.cuda(), labels_test.cuda()

                out, end_points = self.network(images_test)
                proj, _ = self.projection2_V(end_points['Embedding'], phi_t.squeeze(0))
                #
                predictions = self.network(proj, classifier=True)
                # predictions = end_points['Predictions']
                # predictions = F.normalize(out, p=2, dim=1)
                predictions = predictions.cpu().data.numpy()
                test_image_preds.append(predictions)
                test_labels.append(labels_test.cpu().data.numpy())
        predictions = np.concatenate(test_image_preds)
        test_labels = np.concatenate(test_labels)

        accuracy = compute_accuracy(predictions=predictions, labels=test_labels)

        return accuracy

    def batch_test_workflow(self, phi_s, phi_t):
        accuracies = []
        with torch.no_grad():
            for name, test_loader in self.test_loaders:
                accuracy_test = self.batch_test(test_loader, phi_t)
                accuracies.append(accuracy_test)
        return accuracies
