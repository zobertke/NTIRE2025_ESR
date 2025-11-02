"""
Inference module for NTIRE2025 Efficient Super-Resolution
Contains model selection and forward inference logic
"""
import os
import torch
import base64
from io import BytesIO
from PIL import Image
import cv2
from utils import utils_image as util
from utils import utils_logger
import numpy as np

def select_model(model_id, device):
    """
    Load and return the specified model for inference
    
    Args:
        args: Arguments containing model_id
        device: torch device (cuda/mps/cpu)
    
    Returns:
        model: Loaded model ready for inference
        name: Model name string
        data_range: Data range for normalization (1.0 or 255.0)
        tile: Tile size for tiled inference (None for whole image)
    """
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    
    if model_id == 0:
        # Baseline: The 1st Place of the `Overall Performance`` of the NTIRE 2023 Efficient SR Challenge 
        # Edge-enhanced Feature Distillation Network for Efficient Super-Resolution
        # arXiv: https://arxiv.org/pdf/2204.08759
        # Original Code: https://github.com/icandle/EFDN
        # Ckpts: EFDN_gv.pth
        from models.team00_EFDN import EFDN
        name, data_range = f"{model_id:02}_EFDN_baseline", 1.0
        model_path = os.path.join('model_zoo', 'team00_EFDN.pth')
        model = EFDN()
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 1:
        pass
    elif model_id == 7:
        from models.team07_NanoSR import NanoSR_inference
        name, data_range = f"{model_id:02}_NanoSR_inference", 1.0
        model_path = os.path.join('model_zoo', 'team07_NanoSR.pth')
        model = NanoSR_inference(3, 3)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 8:
        from models.team08_FRnet import FRnet
        name, data_range = f"{model_id:02}_FRnet", 1.0
        model_path = os.path.join('model_zoo', 'team08_FRnet.pth')
        model = FRnet()
        model.load_state_dict(torch.load(model_path, map_location=device)['params_ema'], strict=False)
    elif model_id == 10:
        from models.team10_MoeASR import MixtureofAttention_Multiply
        name, data_range = f"{model_id:02}_MoeASR", 1.0
        model_path = os.path.join('model_zoo', 'team10_MoASR.pth')
        model = MixtureofAttention_Multiply(dim=36, kernel_size=7, num_experts=3, topk=1, scale=4, num_blocks=9)
        model.load_state_dict(torch.load(model_path, map_location=device)["params_ema"], strict=True)
    elif model_id == 13:
        from models.team13_HannahSR import HannahSR
        name, data_range = f"{model_id:02}_HannahSR", 1.0
        model_path = os.path.join('model_zoo', 'team13_HannahSR.pth')
        model = HannahSR()
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 15:
        from models.team15_DIPNet_slim_v2 import DIPNet_slim_v2
        name, data_range = f"{model_id:02}_DIPNet_slim_v2", 1.0
        model_path = os.path.join('model_zoo', 'team15_DIPNet_slim_v2.pt')
        model = DIPNet_slim_v2(3, 3, upscale=4, feature_channels=32)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 16:
        from models.team16_SCMSR import SCMSR
        name, data_range = f"{model_id:02}_SCMSR", 1.0
        model_path = os.path.join('model_zoo', 'team16_SCMSR.pth')
        model = SCMSR()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict['params_ema'], strict=True)
    elif model_id == 17:
        from models.team17_FSANet_arch import FSANet
        name, data_range = f"{model_id:02}_FSANet", 1.0
        model_path = os.path.join('model_zoo', 'team17_FSANet.pth')
        model = FSANet(fea_ch=56, conv='EPartialBSConvU', rgb_mean=[0.4488, 0.4371, 0.4040], channel_reduction_rate=2,
                       bias=False)
        model.load_state_dict(torch.load(model_path, map_location=device)["params"], strict=True)
        for module in model.modules():
            if hasattr(module, 'switch_deploy'):
                module.switch_deploy()
    elif model_id == 18:
        from models.team18_SGSDN import SGSDN
        name, data_range = f"{model_id:02}_SGSDN", 1.0
        model_path = os.path.join('model_zoo', 'team18_SGSDN.pth')
        model = SGSDN()
        model.load_state_dict(torch.load(model_path, map_location=device)['params_ema'], strict=True)
    elif model_id == 19:
        from models.team19_SAFMNv3 import SAFMN_NTIRE25
        name, data_range = f"{model_id:02}_SAFMNv3", 1.0
        model = SAFMN_NTIRE25(dim=40, num_blocks=6, ffn_scale=1.5, upscaling_factor=4)
        model_path = os.path.join('model_zoo', f'team19_safmnv3.pth')
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 20:
        from models.team20_DAN import DAN
        name, data_range = f"{model_id:02}_DAN", 1.0
        model_path = os.path.join('model_zoo', 'team20_DAN.pth')
        model = DAN()
        model.load_state_dict(torch.load(model_path, map_location=device)['params_ema'], strict=True)
    elif model_id == 21:
        from models.team21_IESRNet import IESRNet
        name, data_range = f"{model_id:02}_IESRNet", 1.0
        model = IESRNet(3, 3, upscale=4, feature_channels=32)
        model_path = os.path.join('model_zoo', 'team21_IESRNet.pt')
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 22:
        from models.team22_XL import ParticalSRFormer3
        name, data_range = f"{model_id:02}_ParticalSRFormer3_baseline", 1.0
        model_path = os.path.join('model_zoo', 'Team22-pretrain.pth')
        model = ParticalSRFormer3()
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 23:
        from models.team23_DSCF import DSCF
        name, data_range = f"{model_id:02}_DSCF", 1.0
        model_path = os.path.join('model_zoo', 'team23_DSCF.pth')
        model = DSCF(3, 3, feature_channels=26, upscale=4)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    elif model_id == 24:
        import importlib
        model_module = importlib.import_module(f'models.team{model_id:02}_SPANF')
        name, data_range = f"{model_id:02}_SPANF", 1.0
        model = getattr(model_module, f'SPANF')(3, 3, upscale=4, feature_channels=32).eval().to(device)
        model_path = os.path.join('model_zoo', f'team24_spanf.pth')
        stat_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(stat_dict, strict=True)
    elif model_id == 25:
        from models.team25_RepRLFN import RepRLFN
        name, data_range = f"{model_id:02}_RepRLFN", 1.0
        model_path = os.path.join('model_zoo', 'team25_RepRLFN.pth')
        model = RepRLFN()
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 26:
        from models.team26_FMDN import FMDN
        name, data_range = f"{model_id:02}_FMDN_baseline", 1.0
        model_path = os.path.join('model_zoo', 'team26_FMDN.pth')
        model = FMDN()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict['params_ema'], strict=True)
    elif model_id == 27:
        from models.team27_MVFMNet import MVFMNet
        name, data_range = f"{model_id:02}_MVFMNet", 1.0
        model_path = os.path.join('model_zoo', 'team27_MVFMNet.pth')
        model = MVFMNet(dim=26, n_blocks=6, upscaling_factor=4)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 28:
        from models.team28_bvi_srf_arch import BVI_SRF
        name, data_range = f"{model_id:02}_BVI_SRF", 1.0
        model_path = os.path.join('model_zoo', 'team28_bvi_srfnet_g.pth')
        model = BVI_SRF()
        model.load_state_dict(torch.load(model_path, map_location=device)['params'], strict=True)
        model = model.to(device)
        model.eval()
    elif model_id == 29:
        from models.team29_MAANRep import MAAN_rep
        name, data_range = f"{model_id:02}_MAAN",  1.0
        model_path = os.path.join('model_zoo', 'team29_MAANRep.pth')
        model = MAAN_rep()
        model.load_state_dict(torch.load(model_path, map_location=device)['params'], strict=True)
    elif model_id == 30:
        from models.team30_ARRLFN import ARRLFN
        name, data_range = f"{model_id:02}_ARRLFN", 1.0
        model_path = os.path.join('model_zoo', 'team30_ARRLFN.pth')
        model = ARRLFN()
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 31:
        from models.team31_TSR import TSR
        name, data_range = f"{model_id:02}_TSR", 1.0
        model_path = os.path.join('model_zoo', 'team31_TSR.pth')
        model = TSR()
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 33:
        from models.team33_EagleSR import BSRN
        name, data_range = f"{model_id:02}_EagleSR", 1.0
        model_path = os.path.join('model_zoo', 'team33_EagleSR.pth')
        model = BSRN()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict['params'])
    elif model_id == 34:
        from models.team34_PFVM import ParameterFreeVisionMamba
        name, data_range = f"{model_id:02}_PFVM", 1.0
        model_path = os.path.join('model_zoo', 'team34_PFVM.pth')
        model = ParameterFreeVisionMamba()
        model.load_state_dict(torch.load(model_path, map_location=device)['params'], strict=True)
    elif model_id == 35:
        from models.team35_SFNet import SFNet
        name, data_range = f"{model_id:02}_SFNet_baseline", 1.0
        model_path = os.path.join('model_zoo', 'team35_SFNet.pth')
        model = SFNet(3, 3, upscale=4, feature_channels=48)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 36:
        from models.team36_espan import ESPAN
        name, data_range = f"{model_id:02}_espan", 1.0
        model_path = os.path.join('model_zoo', 'team36_espan.pth')
        model = ESPAN(num_in_ch=3,
                      num_out_ch=3,
                      feature_channels=32,
                      mid_channels=32,
                      upscale=4,
                      bias=True,
                      teacher_feature_channels=32,
                      teacher_extra_depth=1,
                      use_fast_op=False)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 361:
        # the same submission as team 36, with the use_fast_op different
        from models.team36_espan import ESPAN
        name, data_range = f"{model_id:02}_espan", 1.0
        model_path = os.path.join('model_zoo', 'team36_espan.pth')
        model = ESPAN(num_in_ch=3,
                      num_out_ch=3,
                      feature_channels=32,
                      mid_channels=32,
                      upscale=4,
                      bias=True,
                      teacher_feature_channels=32,
                      teacher_extra_depth=1,
                      use_fast_op=True)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 37:
        from models.team37_RCUNet import RCUNet
        name, data_range = f"{model_id:02}_RCUNet", 1.0
        model_path = os.path.join('model_zoo', 'team37_RCUNet.pth')
        model = RCUNet()
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 38:
        from models.team38_ESRNet import ESRNet
        name, data_range = f"{model_id:02}_ESRNet", 1.0
        model_path = os.path.join('model_zoo', 'team38_ESRNet.pth')
        model = ESRNet(3, 3, upscale=4, feature_channels=28)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 39:
        from models.team39_ExpandRepNet import ExpandRepNet
        name, data_range = f"{model_id:02}_ExpandRepNet", 1.0
        model_path = os.path.join('model_zoo', 'team39_ExpandRepNet.pth')
        model = ExpandRepNet(dim=36, n_blocks=6, ffn_scale=1.5, upscaling_factor=4)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 40:
        from models.team40_mambairv2light import MambaIRv2LightModel
        name, data_range = f"{model_id:02}_MambaIRv2LightModel", 1.0
        model_path = os.path.join('model_zoo', 'team40_mambairv2light.pth')
        model = MambaIRv2LightModel()
        model.load_weights(model_path)
    elif model_id == 41:
        from models.team41_DepthIBN import IBMDN
        name, data_range = f"{model_id:02}_EFDN_baseline", 1.0
        model_path = os.path.join('model_zoo', 'team41_DepthIBN.pth')
        model = IBMDN()
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 42:
        from models.team42_FEAN import SWAVE
        name, data_range = f"{model_id:02}_[FEAN]", 1.0
        model_path = os.path.join('model_zoo', 'team42_FEAN.pt')
        model = SWAVE(num_in_ch=3, num_out_ch=3, upscale=4, feature_channels=64)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    elif model_id == 43:
        from models.team43_SepSRNet import ESA_CCA
        name, data_range = f"{model_id:02}_SepSRNet", 1.0
        model_path = os.path.join('model_zoo', 'team43_SepSRNet.pth')
        model = ESA_CCA()
        model.load_state_dict(torch.load(model_path, map_location=device)['params_ema'])
    elif model_id == 44:
        from models.team44_SPAN import SPAN
        name, data_range = f"{model_id:02}_SPAN", 1.0
        model_path = os.path.join("model_zoo", "team44_SPANx4.pth")
        model = SPAN(feature_channels=48, img_range=data_range)
        model.load_state_dict(torch.load(model_path, map_location=device)["params"], strict=True)
        model.repa()
    elif model_id == 45:
        from models.team45_TDESR import TDESR
        name, data_range = f"{model_id:02}_TDESR", 1.0
        model_path = os.path.join('model_zoo', 'team45_TDESR.pth')
        model = TDESR()
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 46:
        import importlib
        model_module = importlib.import_module(f'models.team{model_id:02}_SPAN')
        name, data_range = f"{model_id:02}_SPAN", 1.0
        model = getattr(model_module, f'SPAN')(3, 3, upscale=4, feature_channels=28).eval().to(device)
        model_path = os.path.join('model_zoo', f'team46_span.pth')
        stat_dict = torch.load(model_path, map_location=device)['params_ema']
        model.load_state_dict(stat_dict, strict=True)
    elif model_id == 47:
        from models.team47_EECNet import EECNet
        name, data_range = f"{model_id:02}_EECNet", 1.0
        model_path = os.path.join('model_zoo', 'team47_EECNet.pth')
        model = EECNet(dim=32, n_blocks=8)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 48:
        from models.team48_GLoReNet import CASRv016_hybrid_deploy
        name, data_range = f"{model_id:02}_GLoReNet_casrv016_hybrid", 1.0
        model_path = os.path.join('model_zoo', 'GLoReNet_m8c48_26_9205.pt')
        model = CASRv016_hybrid_deploy(module_nums=8, channel_nums=48, down_scale=1, act_type='gelu', scale=4, colors=3)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 50:
        from models.team50_TenInOneSR import TenInOneSR
        name, data_range = f"{model_id:02}_TenInOneSR_baseline", 1.0
        model_path = os.path.join('model_zoo', 'team50_TenInOneSR.pth')
        model = TenInOneSR()
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['params_ema'], strict=False)
    elif model_id == 51:
        import importlib
        model_module = importlib.import_module(f'models.team{model_id:02}_SPAN')
        name, data_range = f"{model_id:02}_SPAN", 1.0
        model = getattr(model_module, f'SPAN')(3, 3, upscale=4, feature_channels=28).eval().to(device)
        model_path = os.path.join('model_zoo', f'team51_span.pth')
        stat_dict = torch.load(model_path, map_location=device)['params_ema']
        model.load_state_dict(stat_dict, strict=True)
    elif model_id == 52:
        from models.team52_ECAS import ECAS
        name, data_range = f"{model_id:02}_ECAS", 1.0
        model_path = os.path.join('model_zoo', 'team52_ECAS.pth')
        model = ECAS(3, 3)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 54:
        from models.team54_HITSR import HiT_SRF
        name, data_range = f"{model_id:02}_HITSR", 1.0
        model_path = os.path.join('model_zoo', 'team54_HITSR.pth')
        model = HiT_SRF(
            upscale=4,
            in_chans=3,
            img_size=64,
            base_win_size=[8, 8],
            img_range=1.0,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            expansion_factor=2,
            resi_connection='1conv',
            hier_win_ratios=[0.5, 1, 2, 4, 6, 8],
            upsampler='pixelshuffledirect'
        )
        model.load_state_dict(torch.load(model_path, map_location=device)['params'], strict=True)
    elif model_id == 56:
        from models.team56_PAEDN import PAEDN
        name, data_range = f"{(model_id):02}_PAEDN", 1.0
        model_path = './model_zoo/team56_PAEDN.pth'
        model = PAEDN()
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 58:
        from models.team58_TSSR import TSSR
        name, data_range = f"{model_id:02}_TSR", 1.0
        model_path = os.path.join('model_zoo', 'team58_TSSR.pth')
        model = TSSR()
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    # Prepare model for inference
    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    return model, name, data_range, tile

def enhance(img, device='cuda', scale = 4, model_id=31, outscale=None):

    # --------------------------------
    # load model
    # --------------------------------
    model, model_name, data_range, tile = select_model(model_id, device)

    # --------------------------------
    # read image
    # --------------------------------
    h_input, w_input = img.shape[0:2]
    # img: numpy
    img = img.astype(np.float32)
    if np.max(img) > 256:  # 16-bit image
        max_range = 65535
        print('\tInput is a 16-bit image')
    else:
        max_range = 255
    img = img / max_range
    if len(img.shape) == 2:  # gray image
        img_mode = 'L'
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA image with alpha channel
        img_mode = 'RGBA'
        alpha = img[:, :, 3]
        img = img[:, :, 0:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_mode = 'RGB'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ------------------- process image (without the alpha channel) ------------------- #
    
    # Convert to tensor using the existing utility function
    img_tensor = util.uint2tensor4(img, data_range)
    
    # Move to device
    img_tensor = img_tensor.to(device)
    
    # --------------------------------
    # (2) img_sr
    # --------------------------------
    if device.type == 'cuda':  
        output_img = forward(img, model, tile, scale = scale)
        torch.cuda.synchronize()
    else:
        output_img = forward(img, model, tile, scale = scale)
        if device.type == 'mps':
            torch.mps.synchronize()
    output_img = util.tensor2uint(output_img, data_range)

    #util.imsave(output_img, os.path.join(save_path, img_name+ext))

    if img_mode == 'L':
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    # ------------------- process the alpha channel if necessary ------------------- #
    if img_mode == 'RGBA':
        h, w = alpha.shape[0:2]
        output_alpha = cv2.resize(alpha, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)

        # merge the alpha channel
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
        output_img[:, :, 3] = output_alpha

    # ------------------------------ return ------------------------------ #
    if max_range == 65535:  # 16-bit image
        output = (output_img / 255.0 * 65535.0).round().astype(np.uint16)
    # else:
    #     output = (output_img * 255.0).round().astype(np.uint8)

    if outscale is not None and outscale != scale:
        output = cv2.resize(
            output, (
                int(w_input * outscale),
                int(h_input * outscale),
            ), interpolation=cv2.INTER_LANCZOS4)

    return output, img_mode


def forward(img_lq, model, tile=None, tile_overlap=32, scale=4):
    """
    Perform forward inference on an image
    
    Args:
        img_lq: Low-quality input image tensor (B, C, H, W)
        model: Loaded model
        tile: Tile size for tiled inference (None for whole image)
        tile_overlap: Overlap between tiles in pixels
        scale: Upscaling factor
    
    Returns:
        output: Super-resolved image tensor
    """
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output
