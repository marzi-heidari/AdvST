# import pandas as pd
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import k_means
from transformers import RobertaModel


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassificationConcat(nn.Module):
    """reference: transformers.RobertaForSequenceClassification"""

    def __init__(self, hidden_dropout_prob=0.0, num_labels=2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False)
        self.hidden_size = self.roberta.config.hidden_size
        self.classifier = RobertaClassificationHead(2 * self.hidden_size, hidden_dropout_prob, num_labels)

    def forward(self, encoded_input):
        output = self.roberta(**encoded_input)
        last_hidden_state = output.last_hidden_state  # same as output[0]
        features = last_hidden_state[:, 0, :]  # take <s> token (equiv. to [CLS])
        features_cat = torch.cat([features, features], dim=-1)
        logits = self.classifier(features_cat)

        return features, logits

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def from_pretrained(self, path):
        self.load_state_dict(torch.load(path))


class RobertaForSequenceClassification(nn.Module):
    """reference: transformers.RobertaForSequenceClassification"""

    def __init__(self, hidden_dropout_prob=0.0, num_labels=2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False)
        self.hidden_size = self.roberta.config.hidden_size
        self.classifier = RobertaClassificationHead(self.hidden_size, hidden_dropout_prob, num_labels)

    def forward(self, encoded_input):
        output = self.roberta(**encoded_input)
        last_hidden_state = output.last_hidden_state  # same as output[0]
        features = last_hidden_state[:, 0, :]  # take <s> token (equiv. to [CLS])

        logits = self.classifier(features)

        return features, logits

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def from_pretrained(self, path):
        self.load_state_dict(torch.load(path))


class AdversarialFeatureMemoryBank:
    def __init__(self, memory_size=1024):
        self.memory_size = memory_size
        self.memory_bank = None

    def init_memory_bank(self, model, trainloader, device, n_clusters=10, num_classes=2, epoch=1):
        model.eval()

        all_features = []
        all_labels = []
        with torch.no_grad():
            for encoded_inputs, labels, _ in trainloader:
                encoded_inputs, labels = encoded_inputs.to(device), labels.to(device)
                # noisy_input = encoded_inputs + torch.randn_like(encoded_inputs) * 0.0001
                _, output = model(encoded_inputs)
                features = output['Embedding']
                all_features.append(features)
                all_labels.append(labels)

            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            _, cluster_label, _ = k_means(all_features.cpu().numpy(), n_clusters)

            size_per_cluster = np.histogram(cluster_label, bins=n_clusters)[0]
            sample_per_cluster = (size_per_cluster / len(cluster_label) * self.memory_size).astype(np.int32)
            max_cluster = np.argmax(size_per_cluster)
            sample_per_cluster[max_cluster] += self.memory_size - np.sum(sample_per_cluster)

            selected_indices = []
            for i in range(n_clusters):
                indices_i = np.where(cluster_label == i)[0]
                selected_indices.append(np.random.choice(indices_i, sample_per_cluster[i], replace=True))

            selected_indices = np.concatenate(selected_indices)
            feats = all_features[selected_indices]
            lbs = all_labels[selected_indices]
        adv_feats = self.langevin_process_cat(model.fc, feats, lbs, num_iters=5, epoch=epoch)
        model.train()
        # adv_feats_ = adv_feats / torch.norm(adv_feats, dim=-1, keepdim=True)
        self.memory_bank = (adv_feats, lbs)

    def update_memory_bank_cat(self, model, trainloader, device, n_clusters=10, num_classes=2, epoch=1):
        model.eval()

        all_features = []
        all_labels = []
        with torch.no_grad():
            for encoded_inputs, labels, _ in trainloader:
                encoded_inputs, labels = encoded_inputs.to(device), labels.to(device)
                # noisy_input = encoded_inputs + torch.randn_like(encoded_inputs) * 0.001
                _, output = model(encoded_inputs)
                features = output['Embedding']
                all_features.append(features)
                all_labels.append(labels)

            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            update_rate = 0.7
            selected_indices = np.random.choice(range(self.memory_size), size=int(self.memory_size * update_rate),
                                                replace=False)

            feats = all_features[selected_indices]
            lbs = all_labels[selected_indices]
        adv_feats = self.langevin_process_cat(model.fc, feats, lbs, num_iters=5, epoch=epoch)
        memory_features, memory_labels = self.memory_bank
        model.train()
        # adv_feats_ = adv_feats / torch.norm(adv_feats, dim=-1, keepdim=True)
        with torch.no_grad():
            mem_cat = torch.cat([memory_features, memory_features], dim=-1)
            logits = model.fc(mem_cat)

            probs = F.softmax(logits, dim=-1)
            entropy = - torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

            _, indices = torch.topk(entropy, k=int(self.memory_size * update_rate), largest=False)

            memory_features[indices] = adv_feats
            memory_labels[indices] = lbs

    def update_memory_bank(self, classifier, features, labels, epoch, num_iters=5):
        adv_features = self.langevin_process(classifier, features, labels, num_iters, epoch)

        memory_features, memory_labels = self.memory_bank
        with torch.no_grad():
            logits = classifier(memory_features)
            probs = F.softmax(logits, dim=-1)
            entropy = - torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

            _, indices = torch.topk(entropy, k=features.size(0), largest=False)

            memory_features[indices] = adv_features
            memory_labels[indices] = labels

    def langevin_process(self, model, x, y, num_iters, epoch):
        eta = lambda t: epoch * 10 / (t + 1)

        for t in range(num_iters):
            x.requires_grad = True

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            # Compute gradients
            loss.backward()
            grad = x.grad

            # grad = grad / grad.norm(p=2, dim=-1, keepdim=True)

            x.requires_grad = False

            # Langevin step update
            with torch.no_grad():
                noise = torch.randn_like(x)
                x = x + grad * eta(t) + noise * torch.sqrt(torch.tensor(2 * eta(t)))

        return x

    def langevin_process_cat(self, model, x, y, num_iters, epoch):

        eta = lambda t: math.log(epoch) * 0.1 / (t + 1)
        for param in model.parameters():
            param.requires_grad = False
        for t in range(num_iters):
            # x = x.detach()
            x.requires_grad = True
            x_cat = torch.cat([x, x], dim=-1)

            logits = model(x_cat)
            loss = F.cross_entropy(logits, y)

            # Compute gradients

            loss.backward()
            grad = x.grad

            # grad = grad / grad.norm(p=2, dim=-1, keepdim=True)

            x.requires_grad = False

            # Langevin step update

            with torch.no_grad():
                noise = torch.randn_like(x)
                ro = x.norm(p=2, dim=-1, keepdim=True) / (grad.norm(p=2, dim=-1, keepdim=True) + 1e-6)
                x = x + grad * eta(t) * ro + noise * torch.sqrt(
                    torch.tensor(2 * eta(t)))
        for param in model.parameters():
            param.requires_grad = True
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                # m.requires_grad_(False)
                m.eval()
        return x

    def langevin_process2(self, model, x, y, num_iters, epoch):
        # eta = 0.1 / (epoch + 1)
        eta = 0.1

        x.requires_grad = True
        optimizer = torch.optim.SGD([x], lr=eta)

        for t in range(100):
            logits = model(x)
            loss = - F.cross_entropy(logits, y)

            # Compute gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return x.requires_grad_(False)

    def generate_random_orthogonal_matrix(self, n):
        # Generate a random matrix
        A = torch.randn(n, n)

        # Perform QR decomposition
        Q, R = torch.linalg.qr(A)

        # Ensure Q is an orthogonal matrix (Q * Q^T = I)
        Q = Q if torch.det(R) > 0 else -Q

        return Q

    def langevin_process3(self, model, x, y, num_iters, epoch):
        random_orthogonal_matrix = self.generate_random_orthogonal_matrix(x.size(1))

        x_hat = torch.matmul(x, random_orthogonal_matrix.to(x.device))

        x_hat.requires_grad = True
        optimizer = torch.optim.SGD([x_hat], lr=20)

        for t in range(100):

            logits = model(x_hat)
            loss = F.cross_entropy(logits, y)

            # Compute gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss.item())

            if loss.item() < 0.1:
                break

        # print(t)
        return x_hat.requires_grad_(False)


class QueueFeatureMemoryBank:
    def __init__(self, memory_size=1024):
        self.memory_size = memory_size
        self.memory_features = None
        self.memory_labels = None

    def update_memory_bank(self, features, labels):
        alpha = torch.randn_like(features) + 1
        beta = torch.randn_like(features)

        features = alpha * features + beta
        if self.memory_features is None:
            self.memory_features = features
            self.memory_labels = labels
        else:
            self.memory_features = torch.cat([self.memory_features, features], dim=0)
            self.memory_labels = torch.cat([self.memory_labels, labels], dim=0)

        if len(self.memory_features) > self.memory_size:
            self.memory_features = self.memory_features[-self.memory_size:]
            self.memory_labels = self.memory_labels[-self.memory_size:]


class FeatureAugmentationNetwork(nn.Module):
    def __init__(self, hidden_size, merge_coeff=None):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.merge_coeff = merge_coeff

    def forward(self, features, memory_features, labels=None, memory_labels=None, num_classses=2):
        tau = 1.0
        q = self.q_proj(features)
        k = self.k_proj(memory_features)
        attn = torch.matmul(q, k.t())
        attn = torch.softmax(attn / tau, dim=-1)
        augment_features = torch.matmul(attn, memory_features)
        if self.merge_coeff is None:
            # beta distribution
            merge_coeff = np.random.beta(0.5, 0.5, size=(features.size(0), 1))
            merge_coeff = torch.tensor(merge_coeff, dtype=torch.float32).to(features.device)
        else:
            merge_coeff = self.merge_coeff

        augmented_features = merge_coeff * features + (1 - merge_coeff) * augment_features
        if mixup_label:
            augment_labels = torch.matmul(attn.detach(), F.one_hot(memory_labels, num_classes=num_classses).float())
            augmented_labels = merge_coeff * F.one_hot(labels, num_classes=num_classses).float() + (
                    1 - merge_coeff) * augment_labels
        else:
            augmented_labels = labels

        return augmented_features, augmented_labels


class FeatureAugmentationNetworkCat(nn.Module):
    def __init__(self, hidden_size, merge_coeff=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.merge_coeff = merge_coeff

    def forward(self, features, memory_features, labels=None, memory_lables=None, device=None):
        tau = 1.0
        q = self.q_proj(features)
        k = self.k_proj(memory_features)
        # q = features
        # k = memory_features
        attn = torch.matmul(q, k.t())
        sqrt = torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        attn = torch.softmax(attn/sqrt, dim=-1)
        v = memory_features
        augment_features = torch.matmul(attn, v)
        # augment_features = F.normalize(augment_features, dim=-1)
        # augment_features_ = augment_features / torch.norm(augment_features, dim=-1, keepdim=True)
        augmented_features = torch.cat([features, augment_features], dim=-1)
        if labels is None:
            return augmented_features
        one_hot = F.one_hot(memory_lables, num_classes=7).float()
        one_hot_o = F.one_hot(labels, num_classes=7).float()
        # augmented_labels = 0.5 * torch.matmul(attn, one_hot) + one_hot_o * 0.5
        # augmented_labels = augmented_labels.to(device)
        augmented_labels = one_hot_o

        return augmented_features, augmented_labels

    def get_only_aug(self, features, memory_features, labels=None, memory_lables=None, device=None):
        tau = 1.0
        q = self.q_proj(features)
        k = self.q_proj(memory_features)
        attn = torch.matmul(q, k.t())
        sqrt = torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        attn = torch.softmax(attn / sqrt, dim=-1)
        v = memory_features
        augment_features = torch.matmul(attn, v)
        # augment_features = F.normalize(augment_features, dim=-1)
        # augment_features_ = augment_features / torch.norm(augment_features, dim=-1, keepdim=True)
        augmented_features = torch.cat([augment_features, augment_features], dim=-1)
        if labels is None:
            return augmented_features
        one_hot = F.one_hot(memory_lables, num_classes=7).float()
        one_hot_o = F.one_hot(labels, num_classes=7).float()
        augmented_labels = torch.matmul(attn, one_hot)
        # augmented_labels = augmented_labels.to(device)
        # augmented_labels = one_hot_o

        return augmented_features, augmented_labels


class FeatureAugmentationNetwork2(nn.Module):
    def __init__(self, hidden_size, merge_coeff=None):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.merge_coeff = merge_coeff

    def forward(self, features, memory_features):
        tau = 1.0
        q = self.q_proj(features)
        k = self.k_proj(memory_features)
        attn = torch.matmul(q, k.t())
        attn = torch.softmax(attn / tau, dim=-1)
        augment_features = torch.matmul(attn, memory_features)
        if self.merge_coeff is None:
            # beta distribution
            merge_coeff = np.random.beta(0.5, 0.5, size=(features.size(0), 1))
            merge_coeff = torch.tensor(merge_coeff, dtype=torch.float32).to(features.device)
        else:
            merge_coeff = self.merge_coeff

        augmented_features = merge_coeff * features + (1 - merge_coeff) * augment_features

        return augmented_features


def feature_augmentation(features, labels, memory_features, memory_labels, mixup_label, num_classes=2):
    tau = 0.01
    q = features
    k = memory_features
    attn = torch.matmul(q, k.t())
    attn = torch.softmax(attn / tau, dim=-1)
    augment_features = torch.matmul(attn, memory_features)
    # beta distribution
    merge_coeff = np.random.beta(0.5, 0.5, size=(features.size(0), 1))
    merge_coeff = torch.tensor(merge_coeff, dtype=torch.float32).to(features.device)

    augmented_features = merge_coeff * features + (1 - merge_coeff) * augment_features
    if mixup_label:
        augment_labels = torch.matmul(attn.detach(), F.one_hot(memory_labels, num_classes=num_classes).float())
        augmented_labels = merge_coeff * F.one_hot(labels, num_classes=num_classes).float() + (
                1 - merge_coeff) * augment_labels
    else:
        augmented_labels = labels

    return augmented_features, augmented_labels
