from torchreid.data.transforms import build_transforms
from PIL import Image
import torchreid
import torch
from torchreid import metrics
import torch


class Reidentificator:
    def __init__(self, cfg):
        self.device = torch.device("cpu")
        self.model = torchreid.models.build_model(
            name='resnet50',
            num_classes=1,
            loss='softmax',
            pretrained=True,
            use_gpu=True
        )
        torchreid.utils.load_pretrained_weights(
            self.model, cfg['TORCHREID_MODEL_PATH'])
        self.model = self.model.to(self.device)
        self.optimizer = torchreid.optim.build_optimizer(
            self.model,
            optim='adam',
            lr=0.0003
        )
        self.scheduler = torchreid.optim.build_lr_scheduler(
            self.optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        _, self.transform_te = build_transforms(
            height=256, width=128,
            random_erase=False,
            color_jitter=False,
            color_aug=False
        )
        self.dist_metric = 'euclidean'
        self.model.eval()

    @torch.no_grad()
    def get_image_features(self, query_img):
        # Convert query image to tensor and extract its feature
        query_img = Image.fromarray(query_img.astype('uint8')).convert('RGB')
        query_img = self.transform_te(query_img)
        query_img = torch.unsqueeze(query_img, 0)
        query_img = query_img.to(self.device)
        query_feature = self._extract_features(query_img)
        query_feature = query_feature.data.cpu()
        return query_feature

    @torch.no_grad()
    def get_features_v2(self, query_feature, gallery_features):
        # Concatenate gallery features into a tensor
        gf = torch.stack([torch.from_numpy(f) for f in gallery_features])

        # Compute distance matrix between query feature and gallery features
        distmat = metrics.compute_distance_matrix(
            query_feature, gf, self.dist_metric)

        # Return distance matrix and query feature as a numpy array
        return distmat.numpy()

    @torch.no_grad()
    def get_features(self, query_imgs, gallery_imgs):
        qf = []
        for img in query_imgs:
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            img = self.transform_te(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(self.device)
            features = self._extract_features(img)
            features = features.data.cpu()
            qf.append(features)
        qf = torch.cat(qf, 0)

        gf = []
        for img in gallery_imgs:
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            img = self.transform_te(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(self.device)
            features = self._extract_features(img)
            features = features.data.cpu()
            gf.append(features)
        gf = torch.cat(gf, 0)

        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)

        return distmat.numpy()

    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)
