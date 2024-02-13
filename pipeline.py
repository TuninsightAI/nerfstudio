import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from uuid import uuid4

import tyro

from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.scripts import run_command

run_command = partial(run_command, verbose=True)


@dataclass
class RunColMAP:
    video_path: Path
    output_dir: Path
    num_frames_target: int = 100

    def __post_init__(self):
        """make sure that video path exists and if output_dir is not there, create it."""
        assert self.video_path.exists(), f"Video path {self.video_path} does not exist."
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def main(self):
        cmd = (f" ns-process-data video "
               f" --data {self.video_path} "
               f" --output-dir {self.output_dir}  "
               f" --num-frames-target {self.num_frames_target}")
        CONSOLE.print(f"Running colmap on {self.video_path} to {self.output_dir}")
        run_command(cmd)


@dataclass
class RunNerFacto:
    data: Path
    model_dir: Path

    def __post_init__(self):
        """make sure that data path exists and if model_dir is not there, create it."""
        assert self.data.exists(), f"Data path {self.data} does not exist."

    def main(self):
        cmd = (f"ns-train nerfacto "
               f"--data {self.data}  "
               f"--output-dir {self.model_dir} "
               f"--timestamp '' "
               f"--method-name 'exp' "
               f" --vis tensorboard ")
        CONSOLE.print(f"Running nerfacto on {self.data} to {self.model_dir}")
        CONSOLE.print(cmd)

        run_command(cmd)


@dataclass
class ExtractPCD:
    model_dir: Path
    output_dir: Path

    def main(self):
        cmd = (f"ns-export pointcloud "
               f"--load-config {self.find_config_file()} "
               f"--output-dir {self.output_dir}")
        run_command(cmd)

    def find_config_file(self):
        for config_file in self.model_dir.rglob("**/config.yml"):
            return config_file.as_posix()

        raise FileNotFoundError()


def main(video_path: Path, output_dir: Path, remove_tmp: bool = False):
    pseudo_id = str(uuid4())[:6]
    data_output_dir = Path(f"/tmp/data/{pseudo_id}")
    model_output_dir = Path(f"/tmp/model/{pseudo_id}")

    run_colmap = RunColMAP(video_path, Path(data_output_dir))
    run_colmap.main()

    run_nerfacto = RunNerFacto(data_output_dir, model_output_dir)
    run_nerfacto.main()

    extract_pcd = ExtractPCD(model_output_dir, output_dir)
    extract_pcd.main()
    if remove_tmp:
        CONSOLE.print(f"remove temporal data folder {data_output_dir}")
        shutil.rmtree(data_output_dir, ignore_errors=True)

        CONSOLE.print(f"remove temporal model folder {model_output_dir}")
        shutil.rmtree(model_output_dir, ignore_errors=True)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
