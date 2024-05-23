import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict
from utils.init_func import init_weight
# from utils.load_utils import load_pretrain
from functools import partial
from models.mobileVit.model_config import get_config
from models.mobileVit.model import ConvLayer
from models.mobileVit.model import InvertedResidual
from models.mobileVit.model import MobileViTBlock
from utils.engine.logger import get_logger
import warnings

# from mmcv.cnn import MODELS as MMCV_MODELS
# from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
# from mmcv.utils import Registry
#
# MODELS = Registry('models', parent=MMCV_MODELS)
# ATTENTION = Registry('attention', parent=MMCV_ATTENTION)
#
# BACKBONES = MODELS
# NECKS = MODELS
# HEADS = MODELS
# LOSSES = MODELS
# SEGMENTORS = MODELS

#
# def build_backbone(cfg):
#     """Build backbone."""
#     return BACKBONES.build(cfg)
#
#
# def build_neck(cfg):
#     """Build neck."""
#     return NECKS.build(cfg)
#
#
# def build_head(cfg):
#     """Build head."""
#     return HEADS.build(cfg)
#
#
# def build_loss(cfg):
#     """Build loss."""
#     return LOSSES.build(cfg)
#
#
# def build_segmentor(cfg, train_cfg=None, test_cfg=None):
#     """Build segmentor."""
#     if train_cfg is not None or test_cfg is not None:
#         warnings.warn(
#             'train_cfg and test_cfg is deprecated, '
#             'please specify them in model', UserWarning)
#     assert cfg.get('train_cfg') is None or train_cfg is None, \
#         'train_cfg specified in both outer field and model field '
#     assert cfg.get('test_cfg') is None or test_cfg is None, \
#         'test_cfg specified in both outer field and model field '
#     return SEGMENTORS.build(
#         cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


logger = get_logger()


class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255),
                 norm_layer=nn.BatchNorm2d, single_GPU=False):
        super(EncoderDecoder, self).__init__()
        self.norm_layer = norm_layer

        if cfg.backbone == 'DFormer-Large':
            from .encoders.DFormer import DFormer_Large as backbone
            self.channels = [96, 192, 288, 576]
        elif cfg.backbone == 'DFormer-Base':
            from .encoders.DFormer import DFormer_Base as backbone
            self.channels = [64, 128, 256, 512]
        elif cfg.backbone == 'DFormer-Small':
            from .encoders.DFormer import DFormer_Small as backbone
            self.channels = [64, 128, 256, 512]
        elif cfg.backbone == 'DFormer-Tiny':
            from .encoders.DFormer import DFormer_Tiny as backbone
            self.channels = [32, 64, 128, 256]

        if single_GPU:
            print('single GPU')
            norm_cfg = dict(type='BN', requires_grad=True)
        else:
            norm_cfg = dict(type='SyncBN', requires_grad=True)

        if cfg.drop_path_rate is not None:
            self.backbone = backbone(drop_path_rate=cfg.drop_path_rate, norm_cfg=norm_cfg)
        else:
            self.backbone = backbone(drop_path_rate=0.1, norm_cfg=norm_cfg)

        self.aux_head = None

        if cfg.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes,
                                           norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)

        elif cfg.decoder == 'ham':
            logger.info('Using Ham Decoder')
            print(cfg.num_classes)
            from .decoders.ham_head import LightHamHead as DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels[1:], num_classes=cfg.num_classes,
                                           in_index=[1, 2, 3], norm_cfg=norm_cfg, channels=cfg.decoder_embed_dim)
            from .decoders.fcnhead import FCNHead
            if cfg.aux_rate != 0:
                self.aux_index = 2
                self.aux_rate = cfg.aux_rate
                print('aux rate is set to', str(self.aux_rate))
                self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            logger.info('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes,
                                       norm_layer=norm_layer)

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)
        model_vit_cfg = get_config("xx_small")
        out_channels = 40
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_vit_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_vit_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_vit_cfg["layer5"])
        exp_channels = min(model_vit_cfg["last_layer_exp_factor"] * out_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            in_channels=out_channels,
            out_channels=exp_channels,
            kernel_size=1
        )
        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module(name="flatten", module=nn.Flatten())
        if 0.0 < model_vit_cfg["cls_dropout"] < 1.0:
            self.classifier.add_module(name="dropout", module=nn.Dropout(p=model_vit_cfg["cls_dropout"]))
        self.classifier.add_module(name="fc", module=nn.Linear(in_features=exp_channels, out_features=cfg.num_classes))

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                        self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                        mode='fan_in', nonlinearity='relu')

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> [nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.Sequential(*block), input_channel

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        out = self.decode_head.forward(x)
        # out = F.interpolate(out, size=orisize[-2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out

    def forward(self, rgb, modal_x=None, label=None):
        # print('builder',rgb.shape,modal_x.shape)
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.conv_1x1_exp(out)
        out = self.classifier(out)
        # if label is not None:
        #     loss = self.criterion(out, label.long())
        #     if self.aux_head:
        #         loss += self.aux_rate * self.criterion(aux_fm, label.long())
        #     return loss
        return out
