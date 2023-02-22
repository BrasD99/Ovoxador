from PARE.pare.models import PARE
from PARE.pare.core.config import update_hparams
from PARE.pare.utils.vibe_image_utils import get_single_image_crop_demo
import torch
from collections import OrderedDict
import cv2
import numpy as np
from PARE.pare.utils.smooth_pose import smooth_pose

class PoseEstimator:
    
    def __init__(self, cfg):
        self.pare_config = cfg['PARE_CFG']
        self.pare_ckpt = cfg['PARE_CKPT']
        self.model_cfg = update_hparams(self.pare_config)
        self.device = torch.device('cpu')
        self.model = PARE(
            backbone=self.model_cfg.PARE.BACKBONE,
            num_joints=self.model_cfg.PARE.NUM_JOINTS,
            softmax_temp=self.model_cfg.PARE.SOFTMAX_TEMP,
            num_features_smpl=self.model_cfg.PARE.NUM_FEATURES_SMPL,
            focal_length=self.model_cfg.DATASET.FOCAL_LENGTH,
            img_res=self.model_cfg.DATASET.IMG_RES,
            pretrained=self.model_cfg.TRAINING.PRETRAINED,
            iterative_regression=self.model_cfg.PARE.ITERATIVE_REGRESSION,
            num_iterations=self.model_cfg.PARE.NUM_ITERATIONS,
            iter_residual=self.model_cfg.PARE.ITER_RESIDUAL,
            shape_input_type=self.model_cfg.PARE.SHAPE_INPUT_TYPE,
            pose_input_type=self.model_cfg.PARE.POSE_INPUT_TYPE,
            pose_mlp_num_layers=self.model_cfg.PARE.POSE_MLP_NUM_LAYERS,
            shape_mlp_num_layers=self.model_cfg.PARE.SHAPE_MLP_NUM_LAYERS,
            pose_mlp_hidden_size=self.model_cfg.PARE.POSE_MLP_HIDDEN_SIZE,
            shape_mlp_hidden_size=self.model_cfg.PARE.SHAPE_MLP_HIDDEN_SIZE,
            use_keypoint_features_for_smpl_regression=self.model_cfg.PARE.USE_KEYPOINT_FEATURES_FOR_SMPL_REGRESSION,
            use_heatmaps=self.model_cfg.DATASET.USE_HEATMAPS,
            use_keypoint_attention=self.model_cfg.PARE.USE_KEYPOINT_ATTENTION,
            use_postconv_keypoint_attention=self.model_cfg.PARE.USE_POSTCONV_KEYPOINT_ATTENTION,
            use_scale_keypoint_attention=self.model_cfg.PARE.USE_SCALE_KEYPOINT_ATTENTION,
            keypoint_attention_act=self.model_cfg.PARE.KEYPOINT_ATTENTION_ACT,
            use_final_nonlocal=self.model_cfg.PARE.USE_FINAL_NONLOCAL,
            use_branch_nonlocal=self.model_cfg.PARE.USE_BRANCH_NONLOCAL,
            use_hmr_regression=self.model_cfg.PARE.USE_HMR_REGRESSION,
            use_coattention=self.model_cfg.PARE.USE_COATTENTION,
            num_coattention_iter=self.model_cfg.PARE.NUM_COATTENTION_ITER,
            coattention_conv=self.model_cfg.PARE.COATTENTION_CONV,
            use_upsampling=self.model_cfg.PARE.USE_UPSAMPLING,
            deconv_conv_kernel_size=self.model_cfg.PARE.DECONV_CONV_KERNEL_SIZE,
            use_soft_attention=self.model_cfg.PARE.USE_SOFT_ATTENTION,
            num_branch_iteration=self.model_cfg.PARE.NUM_BRANCH_ITERATION,
            branch_deeper=self.model_cfg.PARE.BRANCH_DEEPER,
            num_deconv_layers=self.model_cfg.PARE.NUM_DECONV_LAYERS,
            num_deconv_filters=self.model_cfg.PARE.NUM_DECONV_FILTERS,
            use_resnet_conv_hrnet=self.model_cfg.PARE.USE_RESNET_CONV_HRNET,
            use_position_encodings=self.model_cfg.PARE.USE_POS_ENC,
            use_mean_camshape=self.model_cfg.PARE.USE_MEAN_CAMSHAPE,
            use_mean_pose=self.model_cfg.PARE.USE_MEAN_POSE,
            init_xavier=self.model_cfg.PARE.INIT_XAVIER,
        ).to(self.device)

        self.load_pretrained_model(self.model)
        self.model.eval()
    
    def process(self, image, bbox = None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not bbox:
            height, width = image.shape[:2]
            center_x = width // 2
            center_y = height // 2
            bbox = [center_x, center_y, width, height]

        image = self.prepare_image(image, bbox)

        pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d = [], [], [], [], []

        with torch.no_grad():

            batch = image.unsqueeze(0)
            batch = batch.to(self.device)

            output = self.model(batch)

            pred_verts.append(output['smpl_vertices'])
            pred_pose.append(output['pred_pose'])
            pred_betas.append(output['pred_shape'])
            pred_joints3d.append(output['smpl_joints3d'])
            smpl_joints2d.append(output['smpl_joints2d'])
            
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            smpl_joints2d = torch.cat(smpl_joints2d, dim=0)

            del batch

            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()
            smpl_joints2d = smpl_joints2d.cpu().numpy()

            pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
                                                                   min_cutoff=0.004, beta=1.5)

            output_dict = {
                        'verts': pred_verts,
                        'pose': pred_pose,
                        'betas': pred_betas,
                        'joints3d': pred_joints3d,
                        'smpl_joints2d': smpl_joints2d
                    }
            
            return output_dict
        
    def convert_crop_coords_to_orig_img(self, bbox, keypoints, crop_size):
        cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]
        keypoints = 0.5 * crop_size * (keypoints + 1.0)
        keypoints *= h[..., None, None] / crop_size
        keypoints[:,:,0] = (cx - h/2)[..., None] + keypoints[:,:,0]
        keypoints[:,:,1] = (cy - h/2)[..., None] + keypoints[:,:,1]
        return keypoints
    
    def convert_crop_cam_to_orig_img(self, cam, bbox, img_width, img_height):
        cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
        hw, hh = img_width / 2., img_height / 2.
        sx = cam[:,0] * (1. / (img_width / h))
        sy = cam[:,0] * (1. / (img_height / h))
        tx = ((cx - hw) / hw / sx) + cam[:,1]
        ty = ((cy - hh) / hh / sy) + cam[:,2]
        orig_cam = np.stack([sx, sy, tx, ty]).T
        return orig_cam
    
    def prepare_image(self, image, bbox):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        norm_img, _, _ = get_single_image_crop_demo(
                img,
                bbox,
                kp_2d=None,
                scale=1,
                crop_size=224)

        return norm_img
    
    def load_pretrained_model(self, model):
        ckpt = torch.load(self.pare_ckpt, map_location=self.device)['state_dict']
        pretrained_keys = ckpt.keys()
        new_state_dict = OrderedDict()
        for pk in pretrained_keys:
            if pk.startswith('model.'):
                new_state_dict[pk.replace('model.', '')] = ckpt[pk]
            else:
                new_state_dict[pk] = ckpt[pk]

        model.load_state_dict(new_state_dict, strict=False)

        try:
            model.load_state_dict(ckpt, strict=False)
        except RuntimeError:
            model_state_dict = model.ckpt()
            pretrained_keys = ckpt.keys()
            model_keys = model_state_dict.keys()

            updated_pretrained_state_dict = ckpt.copy()

            for pk in pretrained_keys:
                if pk in model_keys:
                    if model_state_dict[pk].shape != ckpt[pk].shape:
                        if pk == 'model.head.fc1.weight':
                            updated_pretrained_state_dict[pk] = torch.cat(
                                [ckpt[pk], ckpt[pk][:,-7:]], dim=-1
                            )
                            continue
                        else:
                            del updated_pretrained_state_dict[pk]

            model.load_state_dict(updated_pretrained_state_dict, strict=False)