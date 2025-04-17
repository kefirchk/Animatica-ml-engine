import logging
import os
import shutil
from tempfile import NamedTemporaryFile
from typing import Iterator

from fastapi import APIRouter, File, HTTPException, UploadFile
from src.ml.services.video_animation import VideoAnimationService
from starlette.responses import StreamingResponse

router = APIRouter(prefix="/fomm", tags=["First Order Motion Model"])
log = logging.getLogger(__name__)


def stream_and_cleanup(filepath: str) -> Iterator[bytes]:
    try:
        with open(filepath, "rb") as f:
            yield from iter(lambda: f.read(8192), b"")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@router.post("/video")
async def animate_image_by_video(
    source_image: UploadFile = File(...),
    driving_video: UploadFile = File(...),
):
    try:
        log.info("Save temporary files")
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_image_file, NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp_video_file:

            shutil.copyfileobj(source_image.file, temp_image_file)
            shutil.copyfileobj(driving_video.file, temp_video_file)
            temp_image_path = temp_image_file.name
            temp_video_path = temp_video_file.name

        output_video_path = temp_video_path.replace(".mp4", "_result.mp4")
        log.info(f"Path to final output video: {output_video_path}")

        service = VideoAnimationService(
            config_path="./data/configs/vox-256.yaml",
            checkpoint_path="./data/checkpoints/vox-cpk.pth.tar",
            source_image_path=temp_image_path,
            driving_video_path=temp_video_path,
            result_video_path=output_video_path,
            relative=False,
            adapt_scale=False,
            find_best=False,
            best_frame=None,
            cpu=True,
        )
        log.info("Run image animation service")
        service.run()

        return StreamingResponse(stream_and_cleanup(output_video_path), media_type="video/mp4")

    except Exception as e:
        log.error(f"Failed to process video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process video: {e}")

    finally:
        log.info("Remove temp files")
        for path in [locals().get("temp_image_path"), locals().get("temp_video_path")]:
            if path and os.path.exists(path):
                os.remove(path)
