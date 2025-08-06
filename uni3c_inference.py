import argparse
import os
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from src.models.uni3c import RealisDanceDiT
from src.pipelines.pipeline_uni3c import RealisDanceDiTPipeline
from src.utils import set_seed, create_logger, is_main_process
from src.fsdp import hook_for_multi_gpu_inference
from diffusers.schedulers import UniPCMultistepScheduler
from src.dataset import load_dataset



def main():
    torch.set_grad_enabled(False)
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_image', type=str, default=None, required=True, help='path to reference image.')
    parser.add_argument('--render_path', type=str, default=None, required=True, help='the path of render folder.')
    parser.add_argument('--output_path', type=str, default="result.mp4", help='Path to output.')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt for video (from reference image).')
    parser.add_argument("--max_area", default=768 * 768, type=int, help="Total pixel area of height * width")
    parser.add_argument("--nframe", default=81, type=int, help="Total number of frames")
    parser.add_argument('--seed', type=int, default=1024, help='The generation seed.')
    parser.add_argument('--save-gpu-memory', action='store_true', help='Save GPU memory, but will be super slow.')
    parser.add_argument("--enable_sp", action="store_true", help="whether to use SP inference")
    args = parser.parse_args()

    # check args
    if args.save_gpu_memory and args.enable_sp:
        raise ValueError("`--enable_sp` and `--save-gpu-memory` cannot be set at the same time.")

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )

    # init dist and set seed
    if args.enable_sp:
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(sequence_parallel_degree=world_size, ulysses_degree=world_size)

    cfg = OmegaConf.load(hf_hub_download(repo_id="ewrfcas/Uni3C", filename="config.json"))
    logger = create_logger(None)
    logger.info(f"World size: {world_size}")

    set_seed(args.seed)

    # load model
    model_id = "theFoxofSky/RealisDance-DiT"
    base_model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    logger.info("loading transformer...")
    transformer = RealisDanceDiT.from_pretrained(model_id, subfolder="transformer", controlnet_cfg=cfg.controlnet_cfg, torch_dtype=torch.bfloat16)
    logger.info("loading controlnet...")
    transformer.build_controlnet(model_path="controlnet.pth", logger=logger)
    image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = RealisDanceDiTPipeline(
        tokenizer=AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer"),
        text_encoder=UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16),
        image_encoder=image_encoder,
        image_processor=CLIPImageProcessor.from_pretrained(base_model_id, subfolder="image_processor"),
        transformer=transformer,
        vae=vae,
        scheduler=UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    )
    if args.save_gpu_memory:
        logger.warning("Enable sequential cpu offload which will be super slow.")
        pipe.enable_sequential_cpu_offload()
    elif args.enable_sp:
        pipe = hook_for_multi_gpu_inference(pipe)
    else:
        pipe.enable_model_cpu_offload()

    image, render_video, render_mask, camera_embedding, smpl_video, hand_video, height, width = load_dataset(
        reference_image=args.reference_image,
        render_path=args.render_path,
        nframe=args.nframe,
        max_area=args.max_area,
        pipe=pipe,
        use_camera_embedding=cfg.get("camera_embedding", False),
        device=device,
        sp_degree=world_size if args.enable_sp else 1,
        logger=logger,
        load_human_info=True,
    )

    output = pipe(
        image=image,
        smpl=smpl_video.to(device),
        hamer=hand_video.to(device),
        render_video=render_video.to(device),
        render_mask=render_mask.to(device),
        camera_embedding=camera_embedding.to(device),
        prompt=(args.prompt),
        negative_prompt="",
        height=height,
        width=width,
        num_frames=args.nframe,
        guidance_scale=5.0,
    ).frames[0]

    if is_main_process():
        export_to_video(output, args.output_path, fps=16)


if __name__ == "__main__":
    main()
