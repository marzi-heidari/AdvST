from __future__ import print_function, absolute_import, division

import clip
from info_nce import InfoNCE, info_nce
import kornia
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import cosine_similarity
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.autoaugment import ImageNetPolicy
from common.pacs import PACS, PACSMultiple
from common.utils import *
from common.utils import (
    fix_all_seed,
    write_log,
    compute_accuracy,
    Averager,
)
from config import PACS_DATA_FOLDER
from models.resnet_vanilla import resnet18


# https://github.com/HAHA-DL/Episodic-DG
def bn_eval(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            # m.requires_grad_(False)
            # pass


# https://github.com/mil-tokyo/dg_mmld/blob/aef26b2745beabc6356accd183ff3e17f71657ce/util/scheduler.py
from torch.optim.lr_scheduler import _LRScheduler


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


def translate_aug(x, trans):
    h = x.shape[-1] * 0.1
    rgb_img = kornia.geometry.transform.translate(x, torch.clamp(trans, -1, 1) * h)
    return rgb_img


def invert_aug(x, max_val):
    x = torch.clamp(max_val, 0.5, 1.0).view(len(max_val), 1, 1, 1) - x
    return x


def shear_aug(x, val):
    x = kornia.geometry.transform.shear(x, val)
    return x


def contrast_aug(x, con):
    rgb_img = kornia.enhance.adjust_contrast(x, torch.clamp(con, 0.1, 1.9))
    return rgb_img


def sharpness_aug(x, factor):
    x = kornia.enhance.sharpness(x, torch.clamp(factor, 0, 1))
    return x


def scale_aug(x, factor):
    factor = factor.view(len(factor), 1)  # for kornia 0.4
    x = kornia.geometry.transform.scale(x, torch.clamp(factor, 0.5, 2.0))
    return x


def solarize_aug(x, factor):
    x = kornia.enhance.solarize(x, additions=torch.clamp(factor, -0.499, 0.499))
    return x


def equalize_aug(x, factor):
    ex = kornia.enhance.equalize(torch.clamp(x, 0.001, 1.0))
    return ex.detach() + x - x.detach()


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

    def setup_path(self, flags):
        root_folder = PACS_DATA_FOLDER
        dataset_names = ["art_painting", "cartoon", "photo", "sketch"]
        seen_index = flags.seen_index
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.train_transform = transforms.Compose(
            [
                # transforms.Resize(256),
                # ImageNetPolicy(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
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
        classifier_param = list(map(id, self.network.fc.parameters()))
        backbone_param = filter(lambda p: id(p) not in classifier_param and p.requires_grad, self.network.parameters())
        parameter_list.append({'params': backbone_param, 'lr': flags.lr})
        parameter_list.append({'params': self.network.fc.parameters(), 'lr': flags.lr})
        self.optimizer = torch.optim.SGD(
            parameter_list,
            weight_decay=flags.weight_decay,
            momentum=flags.momentum,
            nesterov=True,
        )

        if flags.model == "resnet18":
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, flags.train_epochs * len(self.train_loader)
            )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.cnt_loss = InfoNCE()
        self.loss_per_ele = torch.nn.CrossEntropyLoss(reduction="none")

    def save_model(self, file_name, flags):
        if not os.path.exists(flags.model_path):
            os.makedirs(flags.model_path)
        outfile = os.path.join(flags.model_path, file_name)
        torch.save({"state": self.network.state_dict(), "args": flags}, outfile)

    def projection1_test_b(self, Z, phi_t):
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
        D_phi_t = torch.diag(phi_t).to(Z.dtype)
        # D_phi_s = torch.diag(phi_s).to(Z.dtype)
        #
        # # Compute Omega matrices: (n,d) = (n,d) @ (d,d)
        # Omega_t = self.elementwise_pearson_corr(phi_t, Z)
        Omega_t = Z @ D_phi_t
        # D_phi_s = torch.diag(phi_s).to(Z.dtype)

        # Compute the outer product v^T v for each batch element
        # outer_products = torch.bmm(Z.unsqueeze(2), Z.unsqueeze(1))  # Shape: (B, n, n)

        # Add the diagonal matrix to each batch element
        # Omega_t = D_phi_t + outer_products  # Broadcasting over batch
        # Omega_s = D_phi_s + outer_products  # Broadcasting over batch

        # Compute SVD on Omega_t and Omega_s (reduced SVD)
        U_t, _, Vh_t = torch.linalg.svd(Omega_t, full_matrices=False)
        k = int(0.5 * 512)
        V_t = Vh_t.transpose(-2, -1)
        V_t = V_t[..., :k]

        # Get V matrices (transpose of Vh)
        # V_t = V_t.transpose(-2, -1)

        # Compute the projection: Ẑ_proj = U_t U_sᵀ Z V_s V_tᵀ
        # hat_Z = U_t @ (U_t.transpose(0, 1) @ Z) @ (V_t @ V_t.transpose(0, 1))
        hat_Z = Z @ V_t @ V_t.transpose(0, 1)

        return F.normalize(hat_Z, p=2, dim=-1) * Z.norm(p=2, dim=1).mean() + Z

    def projection1_test(self, Z, phi_t):
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
        D_phi_t = torch.diag(phi_t).to(Z.dtype)
        # D_phi_s = torch.diag(phi_s).to(Z.dtype)
        #
        # # Compute Omega matrices: (n,d) = (n,d) @ (d,d)
        # Omega_t = self.elementwise_pearson_corr(phi_t, Z)
        Omega_t = Z @ D_phi_t
        # D_phi_s = torch.diag(phi_s).to(Z.dtype)

        # Compute the outer product v^T v for each batch element
        # outer_products = torch.bmm(Z.unsqueeze(2), Z.unsqueeze(1))  # Shape: (B, n, n)

        # Add the diagonal matrix to each batch element
        # Omega_t = D_phi_t + outer_products  # Broadcasting over batch
        # Omega_s = D_phi_s + outer_products  # Broadcasting over batch

        # Compute SVD on Omega_t and Omega_s (reduced SVD)
        U_t, _, Vh_t = torch.linalg.svd(Omega_t, full_matrices=False)
        k = int(0.5 * 512)
        V_t = Vh_t.transpose(-2, -1)
        V_t = V_t[..., :k]
        U_t = U_t[..., :k]
        # U_s = U_s[..., :k]

        # Get V matrices (transpose of Vh)
        # V_t = V_t.transpose(-2, -1)

        # Compute the projection: Ẑ_proj = U_t U_sᵀ Z V_s V_tᵀ
        hat_Z = U_t @ (U_t.transpose(0, 1) @ Z) @ (V_t @ V_t.transpose(0, 1))
        # hat_Z = Z @ V_t @ V_t.transpose(0, 1)

        return F.normalize(hat_Z, p=2, dim=-1) * Z.norm(p=2, dim=1).mean() + Z

    def elementwise_pearson_corr(self, v: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Computes the Pearson correlation between each element in Z with each element in v,
        returning an m x n matrix of correlations.

        Parameters:
        v (torch.Tensor): A 1D tensor of shape (n,).
        Z (torch.Tensor): A 2D tensor of shape (m, n).

        Returns:
        torch.Tensor: A 2D tensor of shape (m, n), containing correlation values for each element.
        """
        assert v.shape[0] == Z.shape[1], "v must have the same number of elements as the columns of Z"
        v = v.repeat(Z.size(0), 1).to(Z.dtype)
        # Center v and Z (subtract mean)
        v_mean = v.mean()
        Z_mean = Z.mean(dim=1, keepdim=True)

        v_centered = v - v_mean
        Z_centered = Z - Z_mean

        # Compute standard deviations
        v_std = v.std()
        Z_std = Z.std(dim=1, keepdim=True)

        # Pearson correlation (element-wise)
        correlation = (Z_centered * v_centered) / (v_std * Z_std)

        return correlation

    def projection2_U(self, Z, phi_s, phi_t, k):
        """
        Implements Projection 2 using U instead of V.

        Args:
            Z (torch.Tensor): Shape (B, d) - feature tensor for batch.
            phi_s (torch.Tensor): Shape (d,) - source domain representation.
            phi_t (torch.Tensor): Shape (d,) - target domain representation.

        Returns:
            torch.Tensor: Shape (B, d) - projected feature tensor.
        """

        B, d = Z.shape  # Batch size, feature dimension

        # Ensure phi_s and phi_t are broadcastable
        # phi_t = phi_t.view(1, d)  # Reshape to (1, d) for broadcasting
        # phi_s = phi_s.view(1, d)

        # Compute Omega matrices
        # Omega_t = Z + phi_t.repeat(B, 1).to(Z.dtype)  # Shape: (B, d)
        # Omega_s = Z + phi_s.repeat(B, 1).to(Z.dtype)  # Shape: (B, d)
        Omega_t = self.elementwise_pearson_corr(phi_t, Z)
        Omega_s = self.elementwise_pearson_corr(phi_s, Z)

        # Compute SVD
        U_t, S_t, Vh_t = torch.linalg.svd(Omega_t, full_matrices=False)  # (B, d, d)
        U_s, S_s, Vh_s = torch.linalg.svd(Omega_s, full_matrices=False)  # (B, d, d)

        # Extract top k_t singular vectors from target and bottom k_s singular vectors from source
        U_t_top = U_t[..., :k]  # Top k_t singular vectors (B, d, k_t)
        U_s_bottom = U_s[..., -k:]  # Bottom k_s singular vectors (B, d, k_s)

        # Compute projection matrices
        P_t = U_t_top @ U_t_top.transpose(-2, -1)  # Shape: (B, d, d)
        P_s = U_s_bottom @ U_s_bottom.transpose(-2, -1)  # Shape: (B, d, d)

        # Compute projected feature matrix
        z_pos = (P_t @ Z)
        z_neg = P_s @ Z
        Z_proj = z_pos + (z_neg)  # Shape: (B, d)

        return (F.normalize(Z_proj, p=2, dim=-1) * Z.norm(p=2, dim=1).mean()
                ), z_pos, z_neg
        # return Z_proj

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

    def projection2_U_test(self, Z, phi_s, phi_t):
        """
        Implements Projection 2 using U instead of V.

        Args:
            Z (torch.Tensor): Shape (B, d) - feature tensor for batch.
            phi_s (torch.Tensor): Shape (d,) - source domain representation.
            phi_t (torch.Tensor): Shape (d,) - target domain representation.

        Returns:
            torch.Tensor: Shape (B, d) - projected feature tensor.
        """

        B, d = Z.shape  # Batch size, feature dimension

        # Ensure phi_s and phi_t are broadcastable
        # phi_t = phi_t.view(1, d)  # Reshape to (1, d) for broadcasting
        # phi_s = phi_s.view(1, d)
        #
        # # Compute Omega matrices
        # Omega_t = Z + phi_t.repeat(B, 1).to(Z.dtype)  # Shape: (B, d)
        # Omega_s = Z + phi_s.repeat(B, 1).to(Z.dtype)  # Shape: (B, d)
        Omega_t = self.elementwise_pearson_corr(phi_t, Z)
        Omega_s = self.elementwise_pearson_corr(phi_s, Z)

        # Compute SVD
        U_t, S_t, Vh_t = torch.linalg.svd(Omega_t, full_matrices=False)  # (B, d, d)
        U_s, S_s, Vh_s = torch.linalg.svd(Omega_s, full_matrices=False)  # (B, d, d)
        k = int(0.5 * d)
        # Extract top k_t singular vectors from target and bottom k_s singular vectors from source
        U_t_top = U_t[..., -k:]  # Top k_t singular vectors (B, d, k_t)
        U_s_bottom = U_s[..., :k]  # Bottom k_s singular vectors (B, d, k_s)

        # Compute projection matrices
        P_t = U_t_top @ U_t_top.transpose(-2, -1)  # Shape: (B, d, d)
        P_s = U_s_bottom @ U_s_bottom.transpose(-2, -1)  # Shape: (B, d, d)

        # Compute projected feature matrix
        Z_proj = (P_t @ Z)  # Shape: (B, d)

        return (F.normalize(Z_proj, p=2, dim=-1) * Z.norm(p=2, dim=1).mean())
        # return Z_proj

    def projection_3(self, Z, ph_t):
        ndot = (ph_t @ ph_t.transpose(-2, -1))[0][0]
        P = (ph_t.transpose(-2, -1) @ ph_t) / ndot
        return F.normalize(Z @ P, p=2, dim=1) * Z.norm(p=2, dim=1).mean()

    def projection2_V_test(self, Z, phi_t):
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
        D_phi_t = torch.diag(phi_t).to(Z.dtype)
        # D_phi_s = torch.diag(phi_s).to(Z.dtype)
        # #
        # # # Compute Omega matrices: (n,d) = (n,d) @ (d,d)
        Omega_t = Z @ D_phi_t
        # Omega_s = Z @ D_phi_s
        # Compute Omega matrices
        # Omega_t = Z * phi_t  # Shape: (B, d)
        # Omega_s = Z * phi_s  # Shape: (B, d)
        # Omega_t = self.elementwise_pearson_corr(phi_t, Z)

        # EMA update for Omega_t and Omega_s

        # Compute SVD
        U_t, S_t, Vh_t = torch.linalg.svd(Omega_t, full_matrices=False)  # (B, d), (B, d, d)
        # U_s, S_s, Vh_s = torch.linalg.svd(Omega_s, full_matrices=False)  # (B, d), (B, d, d)

        # Get V matrices
        V_t = Vh_t.transpose(-2, -1)  # Shape: (B, d, d)
        # V_s = Vh_s.transpose(-2, -1)  # Shape: (B, d, d)
        k = int(0.8 * d)
        # Extract top k_t singular vectors from target and bottom k_s singular vectors from source
        V_t_top = V_t[..., :k]  # Top k_t singular vectors (B, d, k_t)
        # V_s_bottom = V_s[..., :k]  # Bottom k_s singular vectors (B, d, k_s)

        # Compute projection matrices
        P_t = V_t_top @ V_t_top.transpose(-2, -1)  # Shape: (B, d, d)
        # P_s = V_s_bottom @ V_s_bottom.transpose(-2, -1)  # Shape: (B, d, d)

        # Compute projected feature matrix
        Z_proj = Z @ P_t + Z @ P_t  # Shape: (B, d)

        return F.normalize(Z_proj, p=2, dim=-1) * torch.norm(Z, p=2, dim=-1).mean()

    def projection2_V(self, Z, phi_t):
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

        # Compute Omega matrices
        # Omega_t = Z + phi_t.repeat(B, 1).to(Z.dtype)  # Shape: (B, d)
        # Omega_s = Z + phi_s.repeat(B, 1).to(Z.dtype)  # Shape: (B, d)
        # Omega_t = self.elementwise_pearson_corr(phi_t, Z)
        # Omega_s = self.elementwise_pearson_corr(phi_s, Z)
        D_phi_t = torch.diag(phi_t).to(Z.dtype)

        # # # Compute Omega matrices: (n,d) = (n,d) @ (d,d)
        Omega_t = Z @ D_phi_t
        # Omega_t = self.standardize(Omega_t, dim=1)


        # Compute SVD
        U_t, S_t, Vh_t = torch.linalg.svd(Omega_t, full_matrices=False)  # (B, d, d)
        # U_s, S_s, Vh_s = torch.linalg.svd(Omega_s, full_matrices=False)  # (B, d, d)
        k = int(0.5 * d)
        # Extract top k_t singular vectors from target and bottom k_s singular vectors from source
        V_top = Vh_t.transpose(-2, -1)[..., :k]  # Top k_t singular vectors (B, d, k_t)
        # V_bottom = Vh_s.transpose(-2, -1)[..., -k:]  # Bottom k_s singular vectors (B, d, k_s)

        # Compute projection matrices
        P_t = V_top @ V_top.transpose(-2, -1)  # Shape: (B, d, d)


        # Compute projected feature matrix
        z_pos = Z @ P_t

        Z_proj = z_pos

        return (F.normalize(Z_proj, p=2, dim=-1) * Z.norm(p=2, dim=1).mean())

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
        model, preprocess = clip.load("ViT-B/32", device=f'cuda:{0}')

        t_s = clip.tokenize(
            [
                "A high-resolution, lifelike image of an object captured with a camera, displaying realistic colors, shadows, and textures."]).to(
            f'cuda:{0}')
        t_t = clip.tokenize(
            [
                "Artistic renditions of objects and scenes, often with visible brushstrokes, varied textures, and abstract or exaggerated forms"]).to(
            f'cuda:{0}')
        # t_t = clip.tokenize(
        #     [
        #         "A colorful and stylized cartoon drawing of an object, with bold outlines, simplified details, and exaggerated proportions in a comic or animated style.-["]).to(
        #     f'cuda:{0}')
        # t_s = clip.tokenize(
        #     ["Realistic"]).to(
        #     f'cuda:{0}')
        # t_t = clip.tokenize(
        #     ["Art painting"]).to(
        #     f'cuda:{0}')

        with torch.no_grad():
            phi_s = model.encode_text(t_s)
            phi_t = model.encode_text(t_t)

        for epoch in range(flags.train_epochs):
            loss_avger = Averager()
            self.network.train()
            bn_eval(self.network)

            for images_train, labels_train, _ in self.train_loader:
                inputs, labels = images_train.cuda(), labels_train.cuda()
                o, out = self.network(x=inputs)
                Z = out['Embedding']
                with torch.no_grad():
                    Z_clip = model.encode_image(inputs)
                #
                proj1 = self.projection2_V(Z, phi_t.squeeze(0))
                outputs = self.network(proj1, classifier=True)
                alpha = 0.2

                # Sample lambda from Beta distribution
                lam = torch.distributions.Beta(alpha, alpha).sample().item()

                # Ensure lambda is on the same device as features
                # lam = lam.to(Z.device)
                # Shuffle Z_prime along the batch dimension
                indices = torch.randperm(Z.size(0))
                # Z_prime_shuffled = Z[indices]

                # Perform Mixup
                labels_one_hot = F.one_hot(labels, flags.num_classes).to(Z.dtype)
                Z_mix = lam * proj1 + (1 - lam) * Z[indices]
                Y_mix = lam * labels_one_hot + (1 - lam) * labels_one_hot[indices]
                log_probs = torch.log_softmax(self.network(Z_mix, classifier=True) + 1e-6, dim=1)
                #
                # # Gather log probabilities corresponding to true labels
                loss_mixup = -(Y_mix * log_probs).sum(dim=1).mean()
                # phi_t_mat = phi_t.repeat(Z.size(0), 1).to(Z.dtype)  # Shape: (B, d)
                # phi_s_mat = phi_s.repeat(Z.size(0), 1).to(Z.dtype)  #
                align_loss = F.mse_loss(Z, Z_clip.to(Z.dtype)).mean()
                # align_loss = self.cosine_similarity_loss(Z, Z_clip.to(Z.dtype))
                # align_loss = self.prototype_alignment_loss(Z_clip.to(Z.dtype), Z, labels)
                #
                loss = self.loss_fn(outputs, labels) + loss_mixup + 0.1 * align_loss

                # + 0.1 * self.alignment_loss(Z, phi_s.squeeze(0), phi_t.squeeze(0), 1))
                # + 0.1 * self.alignment_cnt_loss(proj1, z_pos, z_neg, 1))

                self.optimizer.zero_grad()
                loss.backward()
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
            for images_test, labels_test, _ in ood_loader:
                # if images_test.size(0) < 512:
                #     break
                images_test, labels_test = images_test.cuda(), labels_test.cuda()

                out, end_points = self.network(images_test)
                proj = self.projection2_V(end_points['Embedding'],phi_t.squeeze(0))
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
