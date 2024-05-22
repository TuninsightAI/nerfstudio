import moviepy.video.io.ImageSequenceClip
import tyro
from dataclasses import dataclass
from pathlib import Path


@dataclass(kw_only=True)
class ImageToVideoConfig:
    base_dir: Path
    fps: int = 24
    extension: str = ".jpg"
    output_dir: Path
    video_name: str = "out-big-right"
    output_format: str = ".mp4"

    def main(self):

        images = [
            str(x) for x in sorted(Path(self.base_dir).rglob(f"*{self.extension}"))
        ]

        movie_clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
            images, self.fps
        )
        self.output_dir.mkdir(exist_ok=True, parents=True)
        movie_clip.write_videofile(
            str(self.output_dir / (self.video_name + self.output_format)), codec="h264"
        )


def entrypoint():
    tyro.cli(ImageToVideoConfig).main()


if __name__ == "__main__":
    entrypoint()
