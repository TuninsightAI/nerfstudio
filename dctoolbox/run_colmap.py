from __future__ import annotations

import os
import shutil
import sqlite3
import typing
import typing as t
from dataclasses import dataclass
from pathlib import Path

import appdirs
import requests
import rich
import tyro
from loguru import logger
from rich.console import Console
from rich.progress import track
from rich.prompt import Confirm

from dctoolbox.colmap_priors import ColmapPriorConfig
from dctoolbox.inject_pose_priors import (
    ColmapPriorInjectionConfig,
)
from nerfstudio.utils.rich_utils import status
from nerfstudio.utils.scripts import run_command

CONSOLE = Console(width=120)

colmap_command = "colmap"
glomap_command = "glomap"
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
    image_extension: t.Literal["png", "jpg", "jpeg"] = "png",
    verbose: bool = False,
):
    assert_dataset_path(database_path)

    priorConfig = ColmapPriorConfig(
        meta_json=meta_json_path,
        output_folder=output_dir,
        image_dir=image_dir,
        image_extension=image_extension,
    )
    priorConfig.main()

    inject_prior = ColmapPriorInjectionConfig(output_dir, database_path)

    inject_prior.main()


def feature_extraction(
    database_path: Path | str,
    image_folder: str | Path,
    camera_mask_folder: Path | str = None,
    verbose: bool = True,
    img_extension: str = "png",
):
    database_path = Path(database_path)
    assert_dataset_path(database_path=database_path)

    # check consistency of image and mask
    if camera_mask_folder is not None:
        mask_names = sorted(
            set(
                [
                    x.stem
                    for x in camera_mask_folder.rglob("*.png")
                    if len(Path(x.stem).suffix) > 0
                ]
            )
        )
        image_names = sorted(
            set(
                [
                    x.name
                    for x in image_folder.rglob(f"*.{img_extension}")
                    if x.is_file()
                ]
            )
        )
        assert set(image_names).issubset(set(mask_names)), (
            f"Mask and image names are not consistent. "
            f"mask: {mask_names[:5]}, image: {image_names[:5]}"
        )

    feature_extractor_cmd = [
        f"{colmap_command} feature_extractor",
        f"--database_path {database_path.as_posix()}",
        f"--image_path {image_folder}",
        f"--ImageReader.camera_model  PINHOLE",
        f"--SiftExtraction.use_gpu 0",
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
        f"--SiftMatching.use_gpu 0",
        # "--SiftMatching.min_num_inliers 50",
        # f"--SiftMatching.guided_matching 1",
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

def glomap_mapper(
    *, database_path: Path, image_dir: Path, verbose: bool = False, sparse_dir: Path
):
    # Bundle adjustment
    shutil.copy(database_path, database_path.parent / "database.db_before_mapper")

    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd = [
        f"{glomap_command} mapper",
        f"--database_path {database_path}",
        f"--image_path {image_dir}",
        f"--output_path {sparse_dir}",
    ]

    mapper_cmd = " ".join(mapper_cmd)

    rich.print(mapper_cmd)

    with status(
        msg="[bold yellow]Running GLOMAP bundle adjustment... (This may take a while)",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(mapper_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done GLOMAP bundle adjustment.")

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
    CONSOLE.log("[bold green]:tada: Done COLMAP point triangulation.")


def bundle_adjustment(
    *,
    input_path: str | Path,
    output_path: str | Path,
    verbose: bool = True,
    max_num_iterations: int = 100,
):
    # shutil.copy(database_path, database_path.parent / "database.db_before_point_triangulation")
    bundle_adjuster_cmd = [
        f"{colmap_command} bundle_adjuster",
        f"--input_path {input_path}",
        f"--output_path {output_path}",
        "--BundleAdjustment.refine_principal_point 1",
        "--BundleAdjustment.function_tolerance 1e-6",
        f"--BundleAdjustment.max_num_iterations {max_num_iterations}",
    ]
    rich.print(" ".join(bundle_adjuster_cmd))
    with status(
        msg="[bold yellow]Running COLMAP bundle adjustment... (This may take a while)",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(" ".join(bundle_adjuster_cmd), verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done COLMAP bundle adjustment.")


def rig_bundle_adjustment(
    *,
    input_path: str | Path,
    output_path: str | Path,
    verbose: bool = True,
    max_num_iterations: int = 100,
    rig_camera_json: str | Path,
):

    bundle_adjuster_cmd = [
        f"{colmap_command} rig_bundle_adjuster",
        f"--input_path {input_path}",
        f"--output_path {output_path}",
        f"--rig_config_path {rig_camera_json}",
        "--BundleAdjustment.refine_principal_point 1",
        "--BundleAdjustment.function_tolerance 1e-6",
        f"--BundleAdjustment.max_num_iterations {max_num_iterations}",
    ]
    rich.print(" ".join(bundle_adjuster_cmd))
    with status(
        msg="[bold yellow]Running COLMAP bundle adjustment... (This may take a while)",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(" ".join(bundle_adjuster_cmd), verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done COLMAP bundle adjustment.")


def model_alignment(database_path: Path, sparse_dir: Path, verbose: bool = False):
    assert_dataset_path(database_path)
    model_alignment_cmd = [
        "colmap model_aligner",
        f"--input_path {str(sparse_dir)}",
        f"--output_path {str(sparse_dir)}",
        f"--database_path {str(database_path)}",
        f"--ref_is_gps 0",
        f"--alignment_type custom",
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
    CONSOLE.log("[bold green]:tada: Done COLMAP model alignment.")


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
    image_extension: t.Literal["png", "jpg", "jpeg"] = "png"

    rig_bundle_adjustment: bool = False

    def __post_init__(self):
        if self.prior_injection:
            assert (
                self.meta_file is not None
            ), "meta_file is required for prior injection."
        if self.rig_bundle_adjustment:
            assert (
                self.prior_injection
            ), "Rig bundle adjustment requires prior injection."
            assert self.meta_file is not None

    def main(self):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        data_dir = self.data_dir
        image_dir = data_dir / self.image_folder_name
        assert image_dir.exists(), f"{image_dir} does not exist."
        if self.mask_folder_name is not None:
            mask_dir = data_dir / self.mask_folder_name
            assert mask_dir.exists(), f"{mask_dir} does not exist."
        else:
            mask_dir = None
        exp_dir = data_dir / self.experiment_name
        prior_dir = exp_dir / "priors"

        assert data_dir.exists(), f"{data_dir} does not exist."
        assert image_dir.exists(), f"{image_dir} does not exist."

        database_path = data_dir / self.experiment_name / "database.db"

        if not database_path.exists():
            create_empty_database(database_path, verbose=True)
        else:
            logger.info(f"{database_path} already exists.")

        if self.prior_injection:
            meta_file = Path(self.meta_file)
            assert meta_file.exists(), f"{meta_file} does not exist."
            try:
                injection_to_empty_database(
                    meta_json_path=meta_file,
                    database_path=database_path,
                    verbose=True,
                    output_dir=prior_dir,
                    image_dir=image_dir,
                    image_extension=self.image_extension,
                )
            except sqlite3.IntegrityError:
                logger.info("Prior already injected. Skipping...")

        feature_extraction(
            database_path,
            image_dir,
            mask_dir,
            verbose=False,
            img_extension=self.image_extension,
        )

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
        if self.rig_bundle_adjustment:
            rig_bundle_adjustment(
                input_path=exp_dir / "sparse" / "0",
                output_path=exp_dir / "sparse" / "0",
                verbose=True,
                rig_camera_json=exp_dir / "priors" / "rig_cameras.json",
            )
        else:
            bundle_adjustment(
                input_path=exp_dir / "sparse" / "0",
                output_path=exp_dir / "sparse" / "0",
                verbose=True,
            )
        if self.prior_injection:
            model_alignment(database_path, exp_dir / "sparse" / "0", verbose=False)


@dataclass
class ColmapRunnerWithPointTriangulation(ColmapRunner):
    refinement_time: int = 1
    prior_injection: tyro.conf.Suppress[bool] = True
    meta_file: Path

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
        max_num_iterations = 100 // self.refinement_time
        for i in range(self.refinement_time):
            if i == 0:
                sparse_dir = prior_dir
            else:
                sparse_dir = exp_dir / "prior_sparse"
            point_triangulation(
                database_path=database_path,
                image_dir=image_dir,
                prior_dir=sparse_dir,
                sparse_dir=exp_dir / "prior_sparse",
                verbose=False,
            )
            if self.rig_bundle_adjustment:
                rig_bundle_adjustment(
                    input_path=exp_dir / "prior_sparse",
                    output_path=exp_dir / "prior_sparse",
                    verbose=True,
                    max_num_iterations=max_num_iterations,
                    rig_camera_json=exp_dir / "priors" / "rig_cameras.json",
                )
            else:
                bundle_adjustment(
                    input_path=exp_dir / "prior_sparse",
                    output_path=exp_dir / "prior_sparse",
                    verbose=True,
                    max_num_iterations=max_num_iterations,
                )
            model_alignment(database_path, exp_dir / "prior_sparse", verbose=False)

@dataclass
class ColmapRunnerWithGlomap(ColmapRunner):
    refinement_time: int = 1
    prior_injection: tyro.conf.Suppress[bool] = True
    meta_file: Path

    def __post_init__(self):
        assert self.prior_injection is True
        assert self.meta_file is not None

    def main(self):
        data_dir = self.data_dir
        image_dir = data_dir / self.image_folder_name
        exp_dir = data_dir / self.experiment_name
        database_path = data_dir / self.experiment_name / "database.db"
        super().main()
        max_num_iterations = 100 // self.refinement_time
        for i in range(self.refinement_time):
            glomap_mapper(
                database_path=database_path,
                image_dir=image_dir,
                verbose=False,
                sparse_dir=exp_dir / "prior_sparse",
            )

            if self.rig_bundle_adjustment:
                rig_bundle_adjustment(
                    input_path=exp_dir / "prior_sparse/0",
                    output_path=exp_dir / "prior_sparse/0",
                    verbose=True,
                    max_num_iterations=max_num_iterations,
                    rig_camera_json=exp_dir / "priors" / "rig_cameras.json",
                )
            else:
                bundle_adjustment(
                    input_path=exp_dir / "prior_sparse/0",
                    output_path=exp_dir / "prior_sparse/0",
                    verbose=True,
                    max_num_iterations=max_num_iterations,
                )
            model_alignment(database_path, exp_dir / "prior_sparse/0", verbose=False)


if __name__ == "__main__":
    config = tyro.extras.subcommand_cli_from_dict(
        {
            "colmap": ColmapRunnerFromScratch,
            "triangulation": ColmapRunnerWithPointTriangulation,
        }
    )
    config.main()
