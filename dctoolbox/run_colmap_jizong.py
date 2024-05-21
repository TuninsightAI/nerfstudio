import appdirs
import os
import requests
import rich
import shutil
import typing
import tyro
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
from rich.console import Console
from rich.progress import track
from rich.prompt import Confirm

from dctoolbox.colmap_priors import ColmapPriorConfig
from dctoolbox.inject_pose_priors_jizong import (
    ColmapPriorInjectionConfig,
)
from nerfstudio.utils.rich_utils import status
from nerfstudio.utils.scripts import run_command

CONSOLE = Console(width=120)

colmap_command = "colmap"
magick_command = "magick"


def get_vocab_tree() -> Path:
    """Return path to vocab tree. Downloads vocab tree if it doesn't exist.

    Returns:
        The path to the vocab tree.
    """
    vocab_tree_filename = Path(appdirs.user_data_dir("nerfstudio")) / "vocab_tree.fbow"

    if not vocab_tree_filename.exists():
        r = requests.get(
            "https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin", stream=True
        )
        vocab_tree_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_tree_filename, "wb") as f:
            total_length = r.headers.get("content-length")
            assert total_length is not None
            for chunk in track(
                r.iter_content(chunk_size=1024),
                total=int(total_length) / 1024 + 1,
                description="Downloading vocab tree...",
            ):
                if chunk:
                    f.write(chunk)
                    f.flush()
    return vocab_tree_filename


def assert_dataset_path(database_path: Path | str) -> None:
    """
    Assert that the database path is valid.
    """
    database_path = Path(database_path)
    assert (
        database_path.exists() and database_path.is_file()
    ), f"Expected a file, got {database_path}"
    assert database_path.suffix == ".db", f"Expected a .db file, got {database_path}"
    assert (
        database_path.stem == "database"
    ), f"Expected a database.db file, got {database_path}"


def create_empty_database(database_path: Path | str, verbose: bool = False):
    if Path(database_path).exists():
        if Confirm("Exisiting the database, overriding?"):
            os.remove(database_path)
        else:
            raise FileExistsError(f"{database_path} already exists.")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = f"{colmap_command} database_creator --database_path {database_path}"
    rich.print(cmd)
    with status(
        msg="[bold yellow]Running COLMAP dataset creator...",
        spinner="moon",
        verbose=verbose,
    ):
        run_command(cmd, verbose=verbose)


def injection_to_empty_database(
    meta_json_path: Path,
    output_dir: Path,
    database_path: Path,
    image_dir: Path | None = None,
    verbose: bool = False,
):
    assert_dataset_path(database_path)

    priorConfig = ColmapPriorConfig(
        meta_json=meta_json_path, output_folder=output_dir, image_dir=image_dir
    )
    priorConfig.main()

    inject_prior = ColmapPriorInjectionConfig(output_dir, database_path)

    inject_prior.main()


def feature_extraction(
    database_path: Path | str,
    image_folder: str | Path,
    camera_mask_folder: Path | str = None,
    verbose: bool = True,
):
    database_path = Path(database_path)
    assert_dataset_path(database_path=database_path)

    shutil.copy(
        database_path, database_path.parent / "database.db_before_feature_extraction"
    )

    feature_extractor_cmd = [
        f"{colmap_command} feature_extractor",
        f"--database_path {database_path.as_posix()}",
        f"--image_path {image_folder}",
        f"--ImageReader.camera_model  PINHOLE",
        f"--SiftExtraction.use_gpu 1",
        # f"--SiftExtraction.domain_size_pooling 1 ",
        # f"--ImageReader.single_camera_per_folder 1",
        # "--SiftExtraction.estimate_affine_shape 1"
    ]
    if camera_mask_folder is not None:
        feature_extractor_cmd.append(
            f"--ImageReader.mask_path {str(camera_mask_folder)}"
        )
    feature_extractor_cmd = " ".join(feature_extractor_cmd)
    rich.print(feature_extractor_cmd)
    with status(msg="[bold yellow]Extracting features...", spinner="moon"):
        run_command(feature_extractor_cmd, verbose=verbose)


def feature_matching(
    matching_method: typing.Literal[
        "vocab_tree", "exhaustive", "sequential", "spatial"
    ],
    database_path: Path,
    verbose: bool = False,
):
    shutil.copy(
        database_path, database_path.parent / "database.db_before_feature_matching"
    )

    # Feature matching
    feature_matcher_cmd = [
        f"{colmap_command} {matching_method}_matcher",
        f"--database_path {database_path}",
        f"--SiftMatching.use_gpu 1",
        # f"--SiftMatching.guided_matching 1"
    ]
    if matching_method in ["vocab_tree", "sequential"]:
        vocab_tree_filename = get_vocab_tree()
        if matching_method == "vocab_tree":
            feature_matcher_cmd.append(
                f"--VocabTreeMatching.vocab_tree_path {vocab_tree_filename}"
            )
        elif matching_method == "sequential":
            feature_matcher_cmd.append(
                f"--SequentialMatching.vocab_tree_path {vocab_tree_filename}"
            )
    elif matching_method == "spatial":
        feature_matcher_cmd.append(f"--SpatialMatching.is_gps 0")

    feature_matcher_cmd = " ".join(feature_matcher_cmd)
    rich.print(feature_matcher_cmd)
    with status(
        msg="[bold yellow]Running COLMAP feature matcher...",
        spinner="runner",
        verbose=verbose,
    ):
        run_command(feature_matcher_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done matching COLMAP features.")


def mapper(
    *, database_path: Path, image_dir: Path, verbose: bool = False, sparse_dir: Path
):
    # Bundle adjustment
    shutil.copy(database_path, database_path.parent / "database.db_before_mapper")

    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd = [
        f"{colmap_command} mapper",
        f"--database_path {database_path}",
        f"--image_path {image_dir}",
        f"--output_path {sparse_dir}",
        "--Mapper.ba_global_function_tolerance 1e-6",
    ]

    mapper_cmd = " ".join(mapper_cmd)

    rich.print(mapper_cmd)

    with status(
        msg="[bold yellow]Running COLMAP bundle adjustment... (This may take a while)",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(mapper_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done COLMAP bundle adjustment.")


def point_triangulation(
    *,
    database_path: Path,
    image_dir: Path,
    prior_dir: Path,
    sparse_dir: Path,
    verbose: bool = True,
):
    # shutil.copy(database_path, database_path.parent / "database.db_before_point_triangulation")

    sparse_dir.mkdir(parents=True, exist_ok=True)
    point_triangulation_cmd = [
        f"{colmap_command} point_triangulator",
        f"--database_path {database_path}",
        f"--image_path {image_dir}",
        f"--input_path {prior_dir}",
        f"--output_path {sparse_dir}",
        "--Mapper.ba_global_function_tolerance 1e-6",
    ]
    rich.print(" ".join(point_triangulation_cmd))

    with status(
        msg="[bold yellow]Running COLMAP point triangulation... (This may take a while)",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(" ".join(point_triangulation_cmd), verbose=verbose)


def bundle_adjustment(
    *, input_path: str | Path, output_path: str | Path, verbose: bool = True
):
    # shutil.copy(database_path, database_path.parent / "database.db_before_point_triangulation")
    bundle_adjuster_cmd = [
        f"{colmap_command} bundle_adjuster",
        f"--input_path {input_path}",
        f"--output_path {output_path}",
        "--BundleAdjustment.refine_principal_point 1",
    ]
    rich.print(" ".join(bundle_adjuster_cmd))
    with status(
        msg="[bold yellow]Running COLMAP bundle adjustment... (This may take a while)",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(" ".join(bundle_adjuster_cmd), verbose=verbose)


def model_alignment(database_path: Path, sparse_dir: Path, verbose: bool = False):
    assert_dataset_path(database_path)
    model_alignment_cmd = [
        "colmap model_alignment",
        f"--input_path {str(sparse_dir)}",
        f"--output_path {str(sparse_dir)}",
        f"--dataset_path {str(database_path)}",
        f"--ref_is_gps 0",
        f"----alignment_type custom",
        "--robust_alignment 1",
        "--robust_alignment_max_error 3.0",
    ]
    rich.print(" ".join(model_alignment_cmd))
    with status(
        msg="[bold yellow]Running COLMAP model alignment",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(" ".join(model_alignment_cmd), verbose=verbose)


@dataclass
class ColmapRunner:

    data_dir: Path
    image_folder_name: str = "images"
    mask_folder_name: str | None = None
    experiment_name: str = "colmap"
    matching_type: typing.Literal[
        "vocab_tree", "exhaustive", "sequential", "spatial"
    ] = "exhaustive"

    prior_injection: bool = False
    meta_file: Path | None = None

    def __post_init__(self):
        if self.prior_injection:
            assert (
                self.meta_file is not None
            ), "meta_file is required for prior injection."

    def main(self):
        data_dir = self.data_dir
        image_dir = data_dir / self.image_folder_name
        mask_dir = data_dir / self.mask_folder_name
        exp_dir = data_dir / self.experiment_name
        prior_dir = exp_dir / "priors"

        assert data_dir.exists(), f"{data_dir} does not exist."
        assert image_dir.exists(), f"{image_dir} does not exist."

        database_path = data_dir / self.experiment_name / "database.db"

        create_empty_database(database_path, verbose=True)

        if self.prior_injection:
            meta_file = Path(self.meta_file)
            assert meta_file.exists(), f"{meta_file} does not exist."
            injection_to_empty_database(
                meta_json_path=meta_file,
                database_path=database_path,
                verbose=True,
                output_dir=prior_dir,
                image_dir=image_dir,
            )

        feature_extraction(database_path, image_dir, mask_dir, verbose=False)

        feature_matching(self.matching_type, database_path, verbose=False)


@dataclass
class ColmapRunnerFromScratch(ColmapRunner):
    model_alignment: bool = False

    def __post_init__(self):
        if self.model_alignment:
            if self.prior_injection is False:
                logger.warning("Model alignment requires prior injection.. Ignoring...")
                self.model_alignment = False

    def main(self):
        data_dir = self.data_dir
        image_dir = data_dir / self.image_folder_name
        exp_dir = data_dir / self.experiment_name
        database_path = data_dir / self.experiment_name / "database.db"

        super().main()

        mapper(
            database_path=database_path,
            image_dir=image_dir,
            verbose=True,
            sparse_dir=exp_dir / "sparse",
        )
        bundle_adjustment(
            input_path=exp_dir / "sparse" / "0",
            output_path=exp_dir / "sparse" / "0",
            verbose=True,
        )
        if self.prior_injection:
            model_alignment(database_path, exp_dir / "sparse" / "0", verbose=False)


@dataclass
class ColmapRunnerWithPointTriangulation(ColmapRunner):
    refinement_time: int = 5

    def __post_init__(self):
        assert self.prior_injection is True
        assert self.meta_file is not None

    def main(self):
        data_dir = self.data_dir
        image_dir = data_dir / self.image_folder_name
        exp_dir = data_dir / self.experiment_name
        database_path = data_dir / self.experiment_name / "database.db"
        prior_dir = exp_dir / "priors"
        super().main()

        for _ in range(self.refinement_time):
            point_triangulation(
                database_path=database_path,
                image_dir=image_dir,
                prior_dir=prior_dir,
                sparse_dir=exp_dir / "prior_sparse",
                verbose=False,
            )
            bundle_adjustment(
                input_path=exp_dir / "prior_sparse",
                output_path=exp_dir / "prior_sparse",
                verbose=True,
            )
            model_alignment(database_path, exp_dir / "prior_sparse", verbose=False)


if __name__ == "__main__":
    config = tyro.extras.subcommand_cli_from_dict(
        {
            "colmap": ColmapRunnerFromScratch,
            "triangulation": ColmapRunnerWithPointTriangulation,
        }
    )
    config.main()
