#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API v2: audio (wav) -> talking portrait.

"""
import os
import shutil
import pickle
import logging
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from time import time
from datetime import datetime

import cv2
import uvicorn
import numpy as np
import subprocess
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from omegaconf import OmegaConf
from zipfile import ZipFile

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline
from src.utils import utils


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("faster_liveportrait_api_v2")

app = FastAPI()


def run_ffmpeg(params: List[str]) -> None:
    """Run ffmpeg with a list of parameters."""
    cmd = ["ffmpeg"] + [str(p) for p in params]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"ffmpeg failed: {' '.join(cmd)} - {e}")


@contextmanager
def video_writer_context(file_path: str, fps: int, frame_size: Tuple[int, int]):
    """Context manager for video writing to ensure proper resource cleanup - OPTIMIZED."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use original codec
    writer = cv2.VideoWriter(file_path, fourcc, fps, frame_size)
    try:
        yield writer
    finally:
        writer.release()


def process_motion_frames(pipe, motion_infos: Dict[str, Any], source_img, source_info) -> List:
    """Process motion frames and return cropped and original frames with robust fallbacks."""
    start_time = time()
    print(f"[PROFILE] process_motion_frames started")
    motion_list = motion_infos["motion"]
    n = len(motion_list)
    print(f"[PROFILE] Processing {n} frames")

    def ensure_list(key_main: str, key_fallback: str) -> List[Any]:
        if key_main in motion_infos and isinstance(motion_infos[key_main], list):
            lst = motion_infos[key_main]
        elif key_fallback in motion_infos and isinstance(motion_infos[key_fallback], list):
            lst = motion_infos[key_fallback]
        else:
            lst = [None] * n

        if len(lst) < n:
            last = lst[-1] if lst else None
            lst = lst + [last] * (n - len(lst))
        elif len(lst) > n:
            lst = lst[:n]
        return lst

    c_eyes_list = ensure_list("c_eyes_lst", "c_d_eyes_lst")
    c_lip_list  = ensure_list("c_lip_lst",  "c_d_lip_lst")

    frames: List[np.ndarray] = []
    frame_processing_start = time()

    frame_skip = 2
    pipeline_frames = 0
    last_frame_bgr: Optional[np.ndarray] = None

    for frame_idx in range(0, n, frame_skip):
        frame_start = time()
        motion_info = [motion_list[frame_idx], c_eyes_list[frame_idx], c_lip_list[frame_idx]]
        _, out_org = pipe.run_with_pkl(
            motion_info, source_img, source_info, first_frame=(frame_idx == 0)
        )[:2]

        if out_org is not None:
            frame_bgr = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
            last_frame_bgr = frame_bgr
            pipeline_frames += 1
        elif last_frame_bgr is not None:
            frame_bgr = last_frame_bgr
        else:
            continue

        if not frames:
            frames.append(frame_bgr)
        else:
            prev_frame_bgr = frames[-1]
            gap = min(frame_skip, n - len(frames))
            if gap <= 0:
                gap = frame_skip
            for offset in range(1, gap):
                alpha = offset / gap
                interp = cv2.addWeighted(prev_frame_bgr, 1 - alpha, frame_bgr, alpha, 0)
                frames.append(interp)
            frames.append(frame_bgr)

        if frame_idx < 10 or frame_idx >= n - 10:
            print(f"[PROFILE] Frame {frame_idx} processing took {time() - frame_start:.3f}s")

    if len(frames) < n and last_frame_bgr is not None:
        gap = n - len(frames)
        for offset in range(1, gap + 1):
            frames.append(last_frame_bgr.copy())
    elif len(frames) > n:
        frames = frames[:n]

    total_time = time() - start_time
    avg_frame_time = (time() - frame_processing_start) / pipeline_frames if pipeline_frames > 0 else 0
    print(f"[PROFILE] process_motion_frames total time: {total_time:.3f}s, processed {pipeline_frames}/{n} keyframes, avg per keyframe: {avg_frame_time:.3f}s, output frames: {len(frames)}")
    return frames


def run_with_pkl(pipe, source_image_path: str, driving_pickle_path: str, save_dir: str,
                 use_cached_source: bool = False) -> str:
    """Generate video from source image and driving pickle file."""
    start_time = time()
    print(f"[PROFILE] run_with_pkl started")

    if not use_cached_source or not getattr(pipe, "src_imgs", None):
        if not pipe.prepare_source(source_image_path, realtime=False):
            logger.warning(f"No face detected in {source_image_path}")
            return None

    load_start = time()
    with open(driving_pickle_path, "rb") as f:
        motion_infos = pickle.load(f)
    print(f"[PROFILE] Loading pickle took {time() - load_start:.3f}s")

    fps = int(motion_infos["output_fps"])
    if not pipe.src_imgs:
        raise RuntimeError("Source images not prepared")
    h, w = pipe.src_imgs[0].shape[:2]

    # Generate output file paths
    base_name = f"{Path(source_image_path).stem}-{Path(driving_pickle_path).stem}"
    save_dir_path = Path(save_dir)
    result_path = str(save_dir_path / f"{base_name}-res.mp4")

    # Process frames
    frames_start = time()
    frames = process_motion_frames(pipe, motion_infos, pipe.src_imgs[0], pipe.src_infos[0])
    print(f"[PROFILE] Frame processing took {time() - frames_start:.3f}s for {len(frames)} frames")

    adjusted_fps = fps
    print(f"[PROFILE] Adjusted FPS from {fps} to {adjusted_fps}")

    write_start = time()
    with video_writer_context(result_path, adjusted_fps, (w, h)) as org_writer:
        for frame in frames:
            org_writer.write(frame)
    print(f"[PROFILE] Video writing took {time() - write_start:.3f}s")

    total_time = time() - start_time
    print(f"[PROFILE] run_with_pkl total time: {total_time:.3f}s")
    return result_path


def mux_audio_with_video_ultrafast(video_path: str, audio_path: str, output_path: str) -> None:
    """ffmpeg: ultrafast + высокий CRF."""
    run_ffmpeg([
        '-i', video_path, '-i', audio_path,
        '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency', '-crf', '28',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '96k',
        '-shortest',
        '-movflags', '+faststart',
        '-threads', '0',
        '-y', output_path
    ])


def run_with_audio(pipe, joy_pipe, source_image_path: str, audio_path: str, save_dir: str,
                flag_do_crop=True, flag_pasteback=True, driving_multiplier=1.0,
                scale=2.3, vx_ratio=0.0, vy_ratio=-0.125, animation_region="all",
                cfg_scale=1.2) -> Optional[str]:
    """Generate talking portrait video from source image and audio."""
    start_time = time()
    print(f"[PROFILE] run_with_audio started")

    pipe.init_vars()

    # Update configuration - OPTIMIZED FOR PERFORMANCE
    cfg_start = time()
    pipe.update_cfg({
        'flag_relative_motion': False,
        'flag_do_crop': flag_do_crop,
        'flag_pasteback': flag_pasteback,
        'driving_multiplier': driving_multiplier,
        'src_scale': scale,
        'src_vx_ratio': vx_ratio,
        'src_vy_ratio': vy_ratio,
        'animation_region': animation_region,
        'cfg_scale': cfg_scale,
        'flag_stitching': True,
        'flag_crop_driving_video': False,
        'flag_video_editing_head_rotation': False,
        'dri_scale': 2.2,
        'dri_vx_ratio': 0.0,
        'dri_vy_ratio': -0.1,
        'driving_smooth_observation_variance': 1e-7,
    })
    print(f"[PROFILE] Config update took {time() - cfg_start:.3f}s")

    # Prepare source
    source_start = time()
    if not pipe.prepare_source(source_image_path, realtime=False):
        raise RuntimeError("No face detected in source image")
    print(f"[PROFILE] Source preparation took {time() - source_start:.3f}s")

    # Generate motion sequence
    motion_start = time()
    motion_infos = joy_pipe.gen_motion_sequence(audio_path)
    motion_pickle_path = Path(save_dir) / f"{Path(audio_path).stem}_motion.pkl"

    with open(motion_pickle_path, 'wb') as f:
        pickle.dump(motion_infos, f)
    print(f"[PROFILE] Motion generation took {time() - motion_start:.3f}s")

    # Generate videos based on motion sequence
    video_start = time()
    result_path = run_with_pkl(pipe, source_image_path, motion_pickle_path, save_dir, use_cached_source=True)
    print(f"[PROFILE] Video generation from pickle took {time() - video_start:.3f}s")

    if result_path:
        # Create output paths with audio
        result_with_audio_path = str(Path(result_path).with_name(f"{Path(result_path).stem}-audio.mp4"))

        # Mux audio with videos
        mux_start = time()
        mux_audio_with_video_ultrafast(result_path, audio_path, result_with_audio_path)
        print(f"[PROFILE] Audio muxing took {time() - mux_start:.3f}s")

        total_time = time() - start_time
        print(f"[PROFILE] run_with_audio total time: {total_time:.3f}s")
        return result_with_audio_path

    return None


@contextmanager
def temporary_directories(base_dir: str):
    """Context manager for creating and cleaning up temporary directories."""
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    base_path = Path(base_dir)
    save_dir = base_path / timestamp
    temp_dir = base_path / f"temp-{timestamp}"

    save_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)

    try:
        yield str(save_dir), str(temp_dir)
    finally:
        # Only remove the temporary working directory here.
        # save_dir contains the final outputs and is removed by a BackgroundTasks job
        # scheduled by the caller after the response is sent.
        shutil.rmtree(temp_dir, ignore_errors=True)


class PredictAudio:
    def __init__(self, app: FastAPI):
        self.project_dir = Path(__file__).parent
        self.checkpoints_dir = Path(os.environ.get("CHECKPOINT_DIR", self.project_dir / "checkpoints"))
        self.results_dir = Path(os.environ.get("RESULTS_DIR", self.project_dir / "results"))
        
        self._init_pipelines()
        self.results_dir.mkdir(exist_ok=True)
        app.post("/predict_audio/")(self.predict_audio_handler)
        app.get("/ping")(self.ping_handler)

    def _init_pipelines(self):
        """Initialize the processing pipelines."""
        cfg_file = self.project_dir / "configs" / "trt_infer.yaml"
        infer_cfg = OmegaConf.load(cfg_file)
        infer_cfg.infer_params.flag_pasteback = True
        
        self.flp_pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=False)
        self.joy_pipe = JoyVASAAudio2MotionPipeline(
            motion_model_path=self.flp_pipe.cfg.joyvasa_models.motion_model_path,
            audio_model_path=self.flp_pipe.cfg.joyvasa_models.audio_model_path,
            motion_template_path=self.flp_pipe.cfg.joyvasa_models.motion_template_path,
            cfg_mode=self.flp_pipe.cfg.infer_params.cfg_mode,
            cfg_scale=self.flp_pipe.cfg.infer_params.cfg_scale
        )

    def _save_uploaded_file_to_tempdir(self, upload: UploadFile, temp_dir: str) -> str:
        """Save an UploadFile into temp_dir and return the saved file path."""
        if upload is None:
            raise HTTPException(status_code=400, detail="Missing file (upload required)")
        safe_upload_filename = Path(upload.filename).name    
        dest_path = Path(temp_dir) / safe_upload_filename
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        return str(dest_path)

    async def ping_handler(self):
        return {"status": "ok"}

    async def predict_audio_handler(
            self,
            source_image: UploadFile = File(...),
            driving_audio: UploadFile = File(...),
            flag_do_crop: bool = Form(True),
            flag_pasteback: bool = Form(True),
            driving_multiplier: float = Form(1.0),
            scale: float = Form(2.3),
            vx_ratio: float = Form(0.0),
            vy_ratio: float = Form(-0.125),
            animation_region: str = Form("all"),
            cfg_scale: float = Form(1.2),
            background_tasks: BackgroundTasks = None
    ):
        start_time = time()
        print(f"[PROFILE] predict_audio_handler started at {start_time}")

        with temporary_directories(self.results_dir) as (save_dir, temp_dir):
            background_tasks.add_task(shutil.rmtree, save_dir, True)

            # Resolve file paths
            file_save_start = time()
            source_path = self._save_uploaded_file_to_tempdir(source_image, temp_dir)
            audio_path = self._save_uploaded_file_to_tempdir(driving_audio, temp_dir)
            print(f"[PROFILE] File saving took {time() - file_save_start:.3f}s")

            # Generate videos
            generation_start = time()
            org_mp4  = run_with_audio(
                self.flp_pipe, self.joy_pipe, source_path, audio_path, save_dir,
                flag_do_crop=flag_do_crop,
                flag_pasteback=flag_pasteback,
                driving_multiplier=driving_multiplier,
                scale=scale,
                vx_ratio=vx_ratio,
                vy_ratio=vy_ratio,
                animation_region=animation_region,
                cfg_scale=cfg_scale
            )
            print(f"[PROFILE] Video generation took {time() - generation_start:.3f}s")

            total_time = time() - start_time
            logger.info(f"predict_audio completed in {total_time:.2f}s")
            print(f"[PROFILE] predict_audio_handler total time: {total_time:.3f}s")

            if org_mp4 is None:
                raise HTTPException(status_code=400, detail="Video generation failed (no face detected?)")

            return FileResponse(org_mp4, media_type="video/mp4", filename="output.mp4")


# Register endpoint
predict_audio_api = PredictAudio(app)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*"
    )
    warnings.filterwarnings(
        "ignore",
        message="Default grid_sample and affine_grid behavior has changed.*"
    )
    
    uvicorn.run(
        app,
        host=os.environ.get("BIND_HOST", "0.0.0.0"),
        port=int(os.environ.get("HTTP_PORT", 8081))
    )
