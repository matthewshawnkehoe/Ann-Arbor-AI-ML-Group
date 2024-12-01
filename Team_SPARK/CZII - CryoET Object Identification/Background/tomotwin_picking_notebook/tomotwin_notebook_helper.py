import subprocess
import numpy as np
import pandas as pd

from pathlib import Path


def save_mrc(
    mic: np.ndarray,
    filename: str | Path,
    *,
    overwrite: bool = True,
    voxel_size: list | tuple = None,
):
    """
    Save a numpy array to MRC format.

    Parameters:
        mic (np.ndarray): The numpy array to be saved.
        filename (str): The name of the output MRC file.
        overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to True.
        voxel_size (list | tuple, optional): The voxel size of the data. Defaults to None.

    Returns:
        None
    """
    import mrcfile

    if mic.dtype != np.dtype("float32"):
        mic = mic.astype(np.float32)
    with mrcfile.new(str(filename), overwrite=overwrite) as mrc:
        mrc.set_data(mic)
        if voxel_size:
            mrc.voxel_size = voxel_size
    return filename


def convert_single_zarr_to_mrc(
    *,
    zarr_fname: str | Path,
    mrc_fname: str | Path,
    pyramid_level: int = 0,
    voxel_size: list | float = None,
):
    """
    Convert a single zarr file to MRC format.

    Args:
        zarr_fname (str): The path to the zarr file.
        mrc_fname (str): The path to save the MRC file.
        pyramid_level (int, optional): The pyramid level to extract from the zarr file. Defaults to 0.
        voxel_size (list or float, optional): The voxel size of the output MRC file. If a float is provided, it will be
            used for all dimensions. If a list of 3 floats is provided, each float will correspond to the voxel size
            along the x, y, and z dimensions, respectively. Defaults to None.

    Returns:
        bool: True if the conversion is successful.

    Raises:
        ValueError: If voxel_size is not a float or a list of 3 floats.

    Note:
        We can use this function within a python API to convert a single zarr file to MRC format. For bulk processing,
        we can use the `convert_copick_zarrs_to_mrc` function or the command line interface.
    """
    import zarr

    t1 = zarr.open(
        str(zarr_fname),
        mode="r",
    )

    if np.size(voxel_size) == 1:
        voxel_size = [voxel_size, voxel_size, voxel_size]
    elif np.size(voxel_size) != 3:
        raise ValueError("Voxel_size must be a float or a list of 3 floats")

    save_mrc(np.array(t1[pyramid_level]), mrc_fname, voxel_size=voxel_size)
    return mrc_fname


def run_tomotwin_subolume_extraction(
    *,
    coords_fname: str | Path,
    mrc_fname: str | Path,
    out_dir_path: str | Path,
    protein_name: str,
):
    """
    Runs the TomoTwin subvolume extraction tool.

    Args:
        coords_fname (str | Path): The filename of the coordinates file in ".coords" format. This is just a csv file with x, y, z coordinates, separated by a space.
        mrc_fname (str | Path): The filename of the tomogram MRC file.
        out_dir_path (str | Path): The output directory path.
        protein_name (str): The name of the protein. This will be used to name the output files.

    Raises:
        CalledProcessError: If the subprocess command fails.

    Returns:
        None

    """
    command = [
        "tomotwin_tools.py",
        "extractref",
        "--tomo",
        str(mrc_fname),
        "--coords",
        str(coords_fname),
        "--out",
        str(out_dir_path),
        "--filename",
        protein_name,
    ]
    command = " ".join(command)

    print("Executing command:")
    print(command)
    subprocess.run(command, check=True, text=True, shell=True)


def run_tomotwin_embedding_calculation(
    *,
    input_mrc_path: str | Path,
    model_path: str | Path,
    output_dir_path: str | Path,
    stride: int = 2,
    batchsize: int = 64,
):
    """
    Runs TomoTwin script to calculate embeddings for a given MRC file.

    Args:
        input_mrc_path (str): The path to the input MRC file.
        output_dir_path (str): The directory where the output files will be saved.
        model_path (str, optional): The path to the model file. Defaults to "/hpc/projects/group.czii/saugat.kandel/tomotwin_album/tomotwin_latest.pth".
        stride (int, optional): The stride value. Defaults to 2.
        batchsize (int, optional): The batch size. Defaults to 64.
    """
    print("input_mrc_path:", input_mrc_path)
    # Construct the command to run the tomoslice.py script
    command = [
        "tomotwin_embed.py",
        "tomogram",
        "-m",
        str(model_path),
        "-v",
        str(input_mrc_path),
        "-o",
        str(output_dir_path),
        "-s",
        str(stride),
        "-b",
        str(batchsize),
    ]
    command = " ".join(command)

    print("Executing command:")
    print(command)
    # Execute the command
    result = subprocess.run(command, check=True, text=True, shell=True)
    print("Script executed successfully.")
    print("Output:\n", result.stdout)


def run_tomotwin_subvolume_embedding_calculation(
    *, model_path: str | Path, subvolume_mrc_paths: list, out_dir_path: str | Path
):
    """
    Runs the TomoTwin subvolume embedding calculation.

    Args:
        model_path (str): The path to the model file.
        subvolume_mrc_paths (list): A list of paths to the subvolume MRC files.
        out_dir_path (str): The output directory path.

    Raises:
        CalledProcessError: If the command execution fails.

    """
    command = [
        "tomotwin_embed.py",
        "subvolumes",
        "-m",
        str(model_path),
        "-v",
        " ".join([str(p) for p in subvolume_mrc_paths]),
        "-o",
        str(out_dir_path),
    ]

    command = " ".join(command)

    print("Executing command:")
    print(command)
    subprocess.run(command, check=True, text=True, shell=True)


def run_tomotwin_similarity_calculation(
    *,
    ref_embedding_fname: str | Path,
    tomo_embedding_fname: str | Path,
    out_dir_path: str | Path,
):
    """
    Runs TomoTwin script to calculate the per-embedding pairwise similiarity between the reference embeddings and the tomogram volume embeddings.

    Args:
        ref_embedding_fname (str | Path): File name or path to the reference embedding.
        tomo_embedding_fname (str | Path): File name or path to the tomogram embedding.
        out_dir_path (str | Path): Directory path to save the output.

    Raises:
        subprocess.CalledProcessError: If the command execution fails.

    """
    command = [
        "tomotwin_map.py",
        "distance",
        "-r",
        str(ref_embedding_fname),
        "-v",
        str(tomo_embedding_fname),
        "--out",
        str(out_dir_path),
    ]
    command = " ".join(command)

    print("Executing command:")
    print(command)
    subprocess.run(command, text=True, check=True, shell=True)


def run_tomotwin_particle_locator(*, map_fname: str | Path, out_dir_path: str | Path):
    """

    Runs TomoTwin script lo locate potential particle locations in the similarity map and output the coordinates.

    Args:
        map_fname (str or Path): The path to the map file.
        out_dir_path (str or Path): The output directory path where the results will be saved.

    Raises:
        subprocess.CalledProcessError: If the command execution fails.

    """
    command = [
        "tomotwin_locate.py",
        "findmax",
        "-m",
        str(map_fname),
        "-o",
        str(out_dir_path),
    ]
    command = " ".join(command)

    print("Executing command:")
    print(command)
    subprocess.run(command, text=True, check=True, shell=True)


def get_particle_slab_projection(
    tomo_array_xyz: np.ndarray,
    particle_coord_xyz: tuple | list,
    *,
    x_width: int = 50,
    y_width: int = 50,
    z_width: int = 30,
    projection_axis: str = "z",
):
    """
    Retrieves a slab of particles from an MRC file based on the given coordinates and dimensions.

    Args:
        tomo_array_xyz (np.ndarray): The tomogram array, with the coordinates arranged in xyz format.
        coordinate_xyz (tuple | list): The coordinates (x, y, z) of the particle center.
        x_width (int, optional): The width of the slab in the x-direction in voxels. Defaults to 50.
        y_width (int, optional): The width of the slab in the y-direction in voxels. Defaults to 50.
        z_width (int, optional): The width of the slab in the z-direction in voxels. Defaults to 30.
        projection_axis (str, optional): The axis along which to project the slab. Defaults to "z".

    Returns:
        numpy.ndarray: The slab of particles from the MRC file.

    """
    x, y, z = particle_coord_xyz
    x0, x1 = max(x - x_width // 2, 0), x + x_width // 2
    y0, y1 = max(y - y_width // 2, 0), y + y_width // 2
    z0, z1 = max(z - z_width // 2, 0), z + z_width // 2
    slab = tomo_array_xyz[x0:x1, y0:y1, z0:z1]

    projection_axis = projection_axis.lower()
    proj_axis_num = {"x": 0, "y": 1, "z": 2}[projection_axis]
    projection = np.sum(slab, axis=proj_axis_num)
    return projection


def remove_repeated_picks(coordinates: np.ndarray, distance_threshold: float):
    """
    Use agglomerative clustering with a distance threshold to identify particle locations that would be considered the same pick.

    Args:
        coordinates (np.ndarray): Array of coordinates with shape (n, 3), where n is the number of points.
        distance_threshold (float): The maximum distance allowed between two points for them to be considered as repeated picks.

    Returns:
        np.ndarray: Array of mean coordinates for each cluster after removing repeated picks.
    """
    from sklearn.cluster import AgglomerativeClustering

    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="complete",
        metric="euclidean",
    )
    clusters = agg.fit_predict(coordinates)

    unique_clusters = np.unique(clusters)

    # Initialize an array to store the average of each group
    mean_coordinates = np.zeros((unique_clusters.size, coordinates.shape[1]))
    # medoid_coordinates = np.zeros((unique_clusters.size, coordinates.shape[1]))
    # Calculate the mean and medoid for each cluster

    for i, c in enumerate(unique_clusters):
        mean_coordinates[i] = np.mean(coordinates[clusters == c], axis=0)
        # medoid_coordinates[i] = find_cluster_medoid(coordinates[clusters == c])
    return mean_coordinates  # , medoid_coordinates


def precision(*, true_positives: int, false_positives: int):
    out = np.divide(
        true_positives, (true_positives + false_positives + 1e-10), casting="unsafe"
    )
    return out


def recall(*, true_positives: int, false_negatives: int):
    out = np.divide(
        true_positives,
        (true_positives + false_negatives + 1e-10),
        casting="unsafe",
    )
    return out


def fbeta_score(
    *, true_positives: int, false_positives: int, false_negatives: int, beta: float = 1
):
    num = (1 + beta**2) * true_positives
    denom = (1 + beta**2) * true_positives + beta**2 * false_negatives + false_positives
    out = np.divide(num, denom + 1e-10, casting="unsafe")
    return out


def calculate_metrics(
    *,
    gt_picks_df: pd.DataFrame,
    tomotwin_picks_df: pd.DataFrame,
    class_prediction_threshold: float,
    connected_size_threshold: int,
    gt_threshold_distance: float,
    fp_repeat_threshold_distance: float,
):
    """
    Calculate various metrics for evaluating the performance of a prediction algorithm.

    Parameters:
    - gt_picks_df (pd.DataFrame): DataFrame containing coordinates for the ground truth picks.
    - tomotwin_picks_df (pd.DataFrame): DataFrame containing the predicted picks and their metrics.
    - class_prediction_threshold (float): Threshold for class prediction.
    - connected_size_threshold (int): Threshold for connected size.
    - gt_threshold_distance (float): Threshold distance for ground truth picks.
    - fp_repeat_threshold_distance (float): Threshold distance for removing repeated false positives.

    Returns:
    - output (dict): Dictionary containing the calculated metrics:
        - "true_positives" (pd.DataFrame): DataFrame of true positive picks.
        - "false_negatives" (pd.DataFrame): DataFrame of false negative picks.
        - "false_positives" (pd.DataFrame): DataFrame of false positive picks.
        - "precision" (float): Precision score.
        - "recall" (float): Recall score.
        - "f10" (float): F-beta score with beta=10.
    """
    from cupyx.scipy import spatial

    # This is the default output if there are no predicted positives
    output = {
        "true_positives": None,
        "false_negatives": gt_picks_df,
        "false_positives": None,
        "precision": 0,
        "recall": 0,
        "f10": 0,
    }

    predicted_positives = tomotwin_picks_df[
        (tomotwin_picks_df["metric_best"] > class_prediction_threshold)
        & (tomotwin_picks_df["size"] > connected_size_threshold)
    ]
    if predicted_positives.shape[0] < 1:
        return output

    # running this on the gpu gives a significant speedup
    dist_matrix = spatial.distance.cdist(
        predicted_positives[["X", "Y", "Z"]],
        gt_picks_df[["X", "Y", "Z"]],
        "euclidean",
    ).get()

    false_negatives = gt_picks_df[np.min(dist_matrix, axis=0) >= gt_threshold_distance]

    true_positives = gt_picks_df[np.min(dist_matrix, axis=0) < gt_threshold_distance]

    false_positives_all = predicted_positives[
        np.min(dist_matrix, axis=1) >= gt_threshold_distance
    ]

    if false_positives_all.shape[0] < 3:
        false_positives_means = false_positives_all
    else:
        false_positives_means = remove_repeated_picks(
            false_positives_all[["X", "Y", "Z"]].values, fp_repeat_threshold_distance
        )

    # Precision, Recall, F1 Score

    tp = len(true_positives)
    fp = len(false_positives_means)
    fn = len(false_negatives)

    output = {
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives_means,
        "precision": precision(true_positives=tp, false_positives=fp),
        "recall": recall(true_positives=tp, false_negatives=fn),
        "f10": fbeta_score(
            true_positives=tp, false_positives=fp, false_negatives=fn, beta=10
        ),
    }

    return output
