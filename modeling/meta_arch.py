import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from modeling.clip.make_model_clipreid import load_clip_to_cpu
from modeling.clip.LoRA import mark_only_lora_as_trainable as lora_train
from modeling.backbones.vit_pytorch import Mlp


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory, feat_dim):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        self.in_planes = feat_dim
        self.cv_embed_sign = cfg.MODEL.SIE_CAMERA
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_name = cfg.MODEL.TRANSFORMER_TYPE
        self.direct = cfg.MODEL.DIRECT
        self.fowrard_type = cfg.MODEL.FORWARD

        # 多尺度特征配置
        self.multi_scale = cfg.MODEL.MULTI_SCALE
        

        

        if self.fowrard_type == 'new':
            print('using new forward')
            self.pooling = nn.AdaptiveAvgPool1d(1)
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            self.camera_num = camera_num
        else:
            self.camera_num = 0
        # No view
        self.view_num = 0
        if cfg.MODEL.TRANSFORMER_TYPE == 'vit_base_patch16_224':
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                            num_classes=num_classes,
                                                            camera=self.camera_num, view=self.view_num,
                                                            stride_size=cfg.MODEL.STRIDE_SIZE,
                                                            drop_path_rate=cfg.MODEL.DROP_PATH,
                                                            drop_rate=cfg.MODEL.DROP_OUT,
                                                            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
            self.clip = 0
            self.base.load_param(model_path)
            print('Loading pretrained model from ImageNet')
            if cfg.MODEL.FROZEN:
                lora_train(self.base)
        elif cfg.MODEL.TRANSFORMER_TYPE == 'ViT-B-16':
            self.clip = 1
            self.sie_xishu = cfg.MODEL.SIE_COE
            clip_model = load_clip_to_cpu(cfg, self.model_name, cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0],
                                          cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1],
                                          cfg.MODEL.STRIDE_SIZE)
            print('Loading pretrained model from CLIP')
            clip_model.to("cuda")
            self.base = clip_model
            if cfg.MODEL.FROZEN:
                lora_train(self.base)
            if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            elif cfg.MODEL.SIE_CAMERA:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            elif cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(view_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(view_num))
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.inverse = cfg.MODEL.INVERSE
        self.prompt_num = cfg.MODEL.TEXT_PROMPT
        if self.prompt_num <= 0:
            self.prompt_num = None
        if self.inverse:
            self.inverseNet = Mlp(in_features=512, hidden_features=512 * 4, out_features=768, drop=0.1)

    def forward(self, image, text=None, label=None, cam_label=None, view_label=None, modality=None):
        # 计算可见光特征嵌入
        cv_embed = self.sie_xishu * self.cv_embed[cam_label] if self.cv_embed_sign else None

        # 处理文本特征
        if text is not None:
            text_features = self.base.encode_text(text, modality)
            global_feat_text = text_features[torch.arange(text_features.shape[0]), text.argmax(dim=-1)]

            if self.prompt_num is not None:
                learned_token = text_features[:, 5:5 + self.prompt_num]
                global_feat_text = torch.mean(torch.cat([global_feat_text.unsqueeze(1), learned_token], dim=1), dim=1)

            text_inverse = self.inverseNet(global_feat_text) if self.inverse else None
        else:
            text_features, text_inverse, global_feat_text = None, None, None

        # 计算图像特征 
        image_result = self.base.encode_image(image, cv_embed, modality, text_inverse=text_inverse) #(64，130，512)

        # 处理多尺度特征
        if self.multi_scale and isinstance(image_result, tuple):
            image_features, intermediate_features = image_result
        else:
            image_features = image_result if not isinstance(image_result, tuple) else image_result[0]
            intermediate_features = None
        
        global_feat_img = image_features[:, 0]

        # 处理文本特征的全局表示
        if text_features is not None:
            global_feat_txt = image_features[:, -1] if self.inverse else global_feat_text
        else:
            global_feat_txt = image_features[:, 0]
            text_features = image_features

        # 返回特征
        return_values = image_features[:, 1:-1] if self.inverse else image_features[:, 1:]
        if intermediate_features is not None:
            return return_values, global_feat_img, text_features, global_feat_txt, intermediate_features
        else:
            return return_values, global_feat_img, text_features, global_feat_txt
    #             (64，128，512)      (64,512)        (64,77,512)     (64,512)
    def forward_image(self, image, label=None, cam_label=None, view_label=None, modality=None):
        # 计算可见光特征嵌入
        cv_embed = self.sie_xishu * self.cv_embed[cam_label] if self.cv_embed_sign else None

        # 计算图像特征 
        image_result = self.base.encode_image(image, cv_embed, modality, text_inverse=None) #(64，130，512)
         # 处理多尺度特征
        if self.multi_scale and isinstance(image_result, tuple):
            image_features, intermediate_features = image_result
            global_feat_img = image_features[:, 0]
            return image_features, global_feat_img, intermediate_features
        else:
            image_features = image_result if not isinstance(image_result, tuple) else image_result[0]
            global_feat_img = image_features[:, 0]
            return image_features, global_feat_img
            #        (64，128，512)      (64,512)        (64,77,512)     (64,512)
        # 返回特征
        # return_values = image_features[:, 1:] 
        

    def forward_text(self, text=None, label=None, cam_label=None, view_label=None, modality=None):
        text_features = self.base.encode_text(text, modality)
        global_feat_text = text_features[torch.arange(text_features.shape[0]), text.argmax(dim=-1)]

        if self.prompt_num is not None:
            learned_token = text_features[:, 5:5 + self.prompt_num]
            global_feat_text = torch.mean(torch.cat([global_feat_text.unsqueeze(1), learned_token], dim=1), dim=1)
        global_feat_txt =  global_feat_text
        return  text_features, global_feat_txt
    #        (64，128，512)      (64,512)        (64,77,512)     (64,512)
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))
