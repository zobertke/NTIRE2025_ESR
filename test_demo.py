import os.path
import logging
import torch
import argparse
import json
import glob

from pprint import pprint
from fvcore.nn import FlopCountAnalysis
from utils.model_summary import get_model_activation, get_model_flops
from utils import utils_logger
from utils import utils_image as util
from inference import select_model, forward


def select_dataset(data_dir, mode):
    # inference on the DIV2K_LSDIR_test set
    if mode == "test":
        path = [
            (
                p.replace("_HR", "_LR").replace(".png", "x4.png"),
                p
            ) for p in sorted(glob.glob(os.path.join(data_dir, "DIV2K_LSDIR_test_HR/*.png")))
        ]

    # inference on the DIV2K_LSDIR_valid set
    elif mode == "valid":
        path = [
            (
                p.replace("_HR", "_LR").replace(".png", "x4.png"),
                p
            ) for p in sorted(glob.glob(os.path.join(data_dir, "DIV2K_LSDIR_valid_HR/*.png")))
        ]
    else:
        raise NotImplementedError(f"{mode} is not implemented in select_dataset")
    
    return path


def run(model, model_name, data_range, tile, logger, device, args, mode="test"):

    sf = 4
    border = sf
    results = dict()
    results[f"{mode}_runtime"] = []
    results[f"{mode}_psnr"] = []
    if args.ssim:
        results[f"{mode}_ssim"] = []
    # results[f"{mode}_psnr_y"] = []
    # results[f"{mode}_ssim_y"] = []

    # --------------------------------
    # dataset path
    # --------------------------------
    data_path = select_dataset(args.data_dir, mode)
    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    # Use CUDA events for timing on CUDA, or time.time() for MPS/CPU
    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    else:
        import time

    for i, (img_lr, img_hr) in enumerate(data_path):

        # --------------------------------
        # (1) img_lr
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_hr))
        img_lr = util.imread_uint(img_lr, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range)
        img_lr = img_lr.to(device)

        # --------------------------------
        # (2) img_sr
        # --------------------------------
        if device.type == 'cuda':
            start.record()
            img_sr = forward(img_lr, model, tile)
            end.record()
            torch.cuda.synchronize()
            results[f"{mode}_runtime"].append(start.elapsed_time(end))  # milliseconds
        else:
            start_time = time.time()
            img_sr = forward(img_lr, model, tile)
            if device.type == 'mps':
                torch.mps.synchronize()
            end_time = time.time()
            results[f"{mode}_runtime"].append((end_time - start_time) * 1000)  # convert to milliseconds
        img_sr = util.tensor2uint(img_sr, data_range)

        # --------------------------------
        # (3) img_hr
        # --------------------------------
        img_hr = util.imread_uint(img_hr, n_channels=3)
        img_hr = img_hr.squeeze()
        img_hr = util.modcrop(img_hr, sf)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        # print(img_sr.shape, img_hr.shape)
        psnr = util.calculate_psnr(img_sr, img_hr, border=border)
        results[f"{mode}_psnr"].append(psnr)

        if args.ssim:
            ssim = util.calculate_ssim(img_sr, img_hr, border=border)
            results[f"{mode}_ssim"].append(ssim)
            logger.info("{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.".format(img_name + ext, psnr, ssim))
        else:
            logger.info("{:s} - PSNR: {:.2f} dB".format(img_name + ext, psnr))


        # if np.ndim(img_hr) == 3:  # RGB image
        #     img_sr_y = util.rgb2ycbcr(img_sr, only_y=True)
        #     img_hr_y = util.rgb2ycbcr(img_hr, only_y=True)
        #     psnr_y = util.calculate_psnr(img_sr_y, img_hr_y, border=border)
        #     ssim_y = util.calculate_ssim(img_sr_y, img_hr_y, border=border)
        #     results[f"{mode}_psnr_y"].append(psnr_y)
        #     results[f"{mode}_ssim_y"].append(ssim_y)
        # print(os.path.join(save_path, img_name+ext))
            
        # --- Save Restored Images ---
        # util.imsave(img_sr, os.path.join(save_path, img_name+ext))

    # Get memory usage based on device type
    if device.type == 'cuda':
        results[f"{mode}_memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
    elif device.type == 'mps':
        results[f"{mode}_memory"] = torch.mps.current_allocated_memory() / 1024 ** 2
    else:
        results[f"{mode}_memory"] = 0  # CPU doesn't track memory this way
    
    results[f"{mode}_ave_runtime"] = sum(results[f"{mode}_runtime"]) / len(results[f"{mode}_runtime"]) #/ 1000.0
    results[f"{mode}_ave_psnr"] = sum(results[f"{mode}_psnr"]) / len(results[f"{mode}_psnr"])
    if args.ssim:
        results[f"{mode}_ave_ssim"] = sum(results[f"{mode}_ssim"]) / len(results[f"{mode}_ssim"])
    # results[f"{mode}_ave_psnr_y"] = sum(results[f"{mode}_psnr_y"]) / len(results[f"{mode}_psnr_y"])
    # results[f"{mode}_ave_ssim_y"] = sum(results[f"{mode}_ssim_y"]) / len(results[f"{mode}_ssim_y"])
    logger.info("{:>16s} : {:<.3f} [M]".format("Max Memory", results[f"{mode}_memory"]))  # Memery
    logger.info("------> Average runtime of ({}) is : {:.6f} milliseconds".format("test" if mode == "test" else "valid", results[f"{mode}_ave_runtime"]))
    logger.info("------> Average PSNR of ({}) is : {:.6f} dB".format("test" if mode == "test" else "valid", results[f"{mode}_ave_psnr"]))

    return results


def main(args):

    utils_logger.logger_info("NTIRE2025-EfficientSR", log_path="NTIRE2025-EfficientSR.log")
    logger = logging.getLogger("NTIRE2025-EfficientSR")

    # --------------------------------
    # basic settings
    # --------------------------------
    # Select best available device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    model, model_name, data_range, tile = select_model(args.model_id, device)
    logger.info(model_name)

    # if model not in results:
    if True:
        # --------------------------------
        # restore image
        # --------------------------------

        # inference on the DIV2K_LSDIR_valid set
        valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="valid")
        # record PSNR, runtime
        results[model_name] = valid_results

        # inference conducted by the Organizer on DIV2K_LSDIR_test set
        if args.include_test:
            test_results = run(model, model_name, data_range, tile, logger, device, args, mode="test")
            results[model_name].update(test_results)

        input_dim = (3, 256, 256)  # set the input dimension
        activations, num_conv = get_model_activation(model, input_dim)
        activations = activations/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
        logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

        # The FLOPs calculation in previous NTIRE_ESR Challenge
        # flops = get_model_flops(model, input_dim, False)
        # flops = flops/10**9
        # logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        # fvcore is used in NTIRE2025_ESR for FLOPs calculation
        input_fake = torch.rand(1, 3, 256, 256).to(device)
        flops = FlopCountAnalysis(model, input_fake).total()
        flops = flops/10**9
        logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        num_parameters = num_parameters/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
        results[model_name].update({"activations": activations, "num_conv": num_conv, "flops": flops, "num_parameters": num_parameters})

        with open(json_dir, "w") as f:
            json.dump(results, f)
    if args.include_test:
        fmt = "{:20s}\t{:10s}\t{:10s}\t{:14s}\t{:14s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Test PSNR", "Val Time [ms]", "Test Time [ms]", "Ave Time [ms]",
                       "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    else:
        fmt = "{:20s}\t{:10s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Val Time [ms]", "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    for k, v in results.items():
        val_psnr = f"{v['valid_ave_psnr']:2.2f}"
        val_time = f"{v['valid_ave_runtime']:3.2f}"
        mem = f"{v['valid_memory']:2.2f}"
        
        num_param = f"{v['num_parameters']:2.3f}"
        flops = f"{v['flops']:2.2f}"
        acts = f"{v['activations']:2.2f}"
        conv = f"{v['num_conv']:4d}"
        if args.include_test:
            # from IPython import embed; embed()
            test_psnr = f"{v['test_ave_psnr']:2.2f}"
            test_time = f"{v['test_ave_runtime']:3.2f}"
            ave_time = f"{(v['valid_ave_runtime'] + v['test_ave_runtime']) / 2:3.2f}"
            s += fmt.format(k, val_psnr, test_psnr, val_time, test_time, ave_time, num_param, flops, acts, mem, conv)
        else:
            s += fmt.format(k, val_psnr, val_time, num_param, flops, acts, mem, conv)
    with open(os.path.join(os.getcwd(), 'results.txt'), "w") as f:
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2025-EfficientSR")
    parser.add_argument("--data_dir", default="../", type=str)
    parser.add_argument("--save_dir", default="../results", type=str)
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--include_test", action="store_true", help="Inference on the `DIV2K_LSDIR_test` set")
    parser.add_argument("--ssim", action="store_true", help="Calculate SSIM")

    args = parser.parse_args()
    pprint(args)

    main(args)
