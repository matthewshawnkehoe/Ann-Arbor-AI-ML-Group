import json
import os
from typing import Any, Dict, List

import copick
import numpy as np
import ome_zarr.writer
import starfile
import zarr
from scipy.spatial.transform import Rotation as R


def get_copick_project_tomo_ids(copickRoot):
    copickRoot = copick.from_file(copickRoot)
    tomoIDs = [run.name for run in copickRoot.runs]

    return tomoIDs


def read_copick_tomogram_group(copickRoot, voxelSize, tomoAlgorithm, tomoID=None):
    """Find the Zarr Group Relating to a Copick Tomogram.

    Args:
        copickRoot: Target Copick Run to Extract Tomogram Zarrr Group.
        voxelSize: Name of the Tomogram.
        tomoAlgorithm: Session ID for the Tomogram.
        tomoID: Tomogram ID for that Dataset.

    Returns:
        ZarrGroup for the Tomogram object.
    """

    # Get First Run and Pull out Tomgram
    if tomoID == None:
        run = copickRoot.get_run(copickRoot.runs[0].name)
    else:
        run = copickRoot.get_run(tomoID)

    tomogram = run.get_voxel_spacing(voxelSize).get_tomogram(tomoAlgorithm)

    # Convert Zarr into Vol and Extract Shape
    group = zarr.open(tomogram.zarr())

    return group


def get_copick_tomogram(copickRoot, voxelSize=10, tomoAlgorithm="denoised", tomoID=None):
    """Return a Tomogram from a Copick Run.

    Args:
        copickRoot: Target Copick Root to Extract Tomogram.
        voxelSize: Name of the Tomogram.
        tomoAlgorithm: Session ID for the Tomogram.
        tomoID: Tomogram ID for that Dataset.

    Returns:
        ZarrGroup for the Tomogram object.
    """

    group = read_copick_tomogram_group(copickRoot, voxelSize, tomoAlgorithm, tomoID)

    # Return Volume
    return list(group.arrays())[0][1]


def get_copick_tomogram_shape(copickRoot, voxelSize=10, tomoAlgorithm="denoised"):
    """Return a Tomogram Dimensions (nx, ny, nz) from a Copick Run.

    Args:
        copickRoot: Target Copick Run to Extract Tomogram Dimensions.
        voxelSize: Name of the Tomogram.
        tomoAlgorithm: Session ID for the Tomogram.
        tomoID: Tomogram ID for that Dataset.

    Returns:
        TomogrameShape
    """

    # Return Volume Shape
    return get_copick_tomogram(copickRoot, voxelSize, tomoAlgorithm).shape


def get_target_empty_tomogram(copickRoot, voxelSize=10, tomoAlgorithm="denoised"):
    """Return an Empty Tomogram with Equivalent Dimensions (nx, ny, nz) from a Copick Run.
    Args:
        copickRoot: Target Copick Run to Extract Empty Tomogram.
        voxelSize: Name of the Tomogram.
        tomoAlgorithm: Session ID for the Tomogram.

    Returns:
        A Tomogram Composed of Zeros with the Same Shape as in Copick Project.
    """

    return np.zeros(get_copick_tomogram_shape(copickRoot, voxelSize, tomoAlgorithm), dtype=np.int8)


def get_copick_segmentation(copickRun, segmentationName="test-segmentation7", userID="deepfinder"):
    """Return a Specified Copick Segmentation.
    Args:
        copickRun: Target Copick Run to Extract Tomogram.
        segmentationName: Name of the Segmentation.
        userID: User who Created the Segmentation.

    Returns:
        The Segmentation within that Copick-Run.
    """

    # Get the Segmentation from the Following Copick Run
    seg = copickRun.get_segmentations(name=segmentationName, user_id=userID)[0]

    # Return the Corresponding Segmentation Volume
    store = seg.zarr()

    return zarr.open(store, mode="r")[0]


def get_ground_truth_coordinates(copickRun, voxelSize, proteinIndex):
    """Get the Ground Truth Coordinates From Copick and Return as a Numpy Array.

    Args:
        copickRun: Voxel size for the segmentation.
        voxelSize: Name of the segmentation.
        proteinIndex: Session ID for the segmentation.

    Returns:
        coords: The newly created segmentation object.
    """

    picks = copickRun.picks[proteinIndex]

    coords = []
    for ii in range(len(picks.points)):
        coords.append((
            picks.points[ii].location.x / voxelSize,
            picks.points[ii].location.y / voxelSize,
            picks.points[ii].location.z / voxelSize,
        ))

    return np.array(coords)


def custom_get_copick_protein_coords(
    copick_root,
    tomo_id,
    protein_name,
    session_id=None,
    user_id="deepfinder",
    voxel_spacing=10,
):
    """
    Get the coordinates of a specific protein in a tomogram.

    Parameters:
        copick_root (CopickRoot): The root object of the Copick data structure.
        tomo_id (str): The ID of the tomogram.
        protein_name (str): The name of the protein.
        session_id (str, optional): The ID of the session. Defaults to None.
        user_id (str, optional): The ID of the user. Defaults to "deepfinder".
        voxel_spacing (int, optional): The voxel spacing. Defaults to 10.

    Returns:
        numpy.ndarray: An array of coordinates (x, y, z) of the protein. If `session_id` is None, the coordinates of the pick with the highest session ID are returned.

    Raises:
        ValueError: If no picks match the parameter combination specified.
    """
    copick_run = copick_root.get_run(tomo_id)
    _pick_idx = None
    for _ip, _pick in enumerate(copick_run.picks):
        if _pick.pickable_object_name != protein_name or _pick.user_id != user_id:
            continue
        if session_id is None:
            _pick_idx = max(_pick_idx, _ip) if _pick_idx is not None else _ip
        elif _pick.session_id == session_id:
            _pick_idx = _ip
            break

    coords = []
    if _pick_idx is None:
        picked_objects = [_pick.pickable_object_name for _pick in copick_run.picks]
        for p in copick_run.picks:
            print(p)
        if protein_name not in picked_objects:
            print(
                f"No picks for protein {protein_name} in tomogram {tomo_id} with session_id {session_id}"
            )

            return np.array(coords).astype("int")
        else:
            print(
                f"Specified parameters: user_id: {user_id}, tomo_id: {tomo_id}, protein_name: {protein_name}, session_id: {session_id}"
            )
            raise ValueError("No picks match the parameter combination specified.")
    _pick = copick_run.picks[_pick_idx]
    coords = []
    for ii in range(len(_pick.points)):
        coords.append((
            _pick.points[ii].location.x / voxel_spacing,
            _pick.points[ii].location.y / voxel_spacing,
            _pick.points[ii].location.z / voxel_spacing,
        ))
    return np.floor(coords).astype("int")


# I need to Figure Out if I want Option 1 or Option 2..
# def get_pickable_object_label(copickRun, objectName):
#     for ii in range(len(copickRun.picks)):
#         if copickRun.picks[ii].pickable_object_name == objectName:
#             return ii


def get_pickable_object_label(copickRoot, objectName):
    for ii in range(len(copickRoot.pickable_objects)):
        if copickRoot.pickable_objects[ii].name == objectName:
            return copickRoot.pickable_objects[ii].label


def read_copick_json(filePath):
    """
    Read and processes a copick JSON coordinate file and returns as NumPy array.

    Args:
    - filePath (str): The path to the JSON file to be read.

    Returns:
    - np.ndarray: A NumPy array where first three columns are the coordinates (x, y, z), and the next three columns are the Euler angles (in degrees).
                  Note: X/Y/Z coordinates are returned in Angstroms.
    """

    # Load JSON data from a file
    with open(os.path.join(filePath), "r") as jFile:
        data = json.load(jFile)

    # Initialize lists to hold the converted data
    coordinates = []
    eulerAngles = []

    # Loop through each point in the JSON data
    for point in data["points"]:
        rotationMatrix = []

        # Extract the location and convert it to a NumPy array
        currLocation = np.array([
            point["location"]["x"],
            point["location"]["y"],
            point["location"]["z"],
        ])

        # Extract the transformation matrix and convert it to a NumPy array
        rotationMatrix = R.from_matrix(np.array(point["transformation_"])[:3, :3])
        currEulerAngles = rotationMatrix.as_euler("ZYZ", degrees=True)

        coordinates.append(currLocation)
        eulerAngles.append(currEulerAngles)

    return np.hstack((np.array(coordinates), np.array(eulerAngles)))


def convert_copick_coordinates_to_xml(copickRun, xml_objects, pixelSize=10):
    picks = copickRun.picks
    for proteinLabel in range(len(picks)):
        xml_objects.append()

    return xml_objects


def write_relion_output(
    specimen,
    tomoID,
    coords,
    outputDirectory="refinedCoPicks/ExperimentRuns",
    pixelSize=10,
):
    """
    Read and processes a copick JSON coordinate file and returns as NumPy array.

    Args:
    - filePath (str): The path to the JSON file to be read.

    Returns:
    - np.ndarray: A NumPy array where first three columns are the coordinates (x, y, z), and the next three columns are the Euler angles (in degrees).
                  Note: X/Y/Z coordinates are returned in Angstroms.
    """

    outputStarFile = {}

    # Coordinates
    if coords.shape[0] > 0:
        outputStarFile["rlnCoordinateX"] = coords[:, 0] / pixelSize
        outputStarFile["rlnCoordinateY"] = coords[:, 1] / pixelSize
        outputStarFile["rlnCoordinateZ"] = coords[:, 2] / pixelSize

        # Angles
        outputStarFile["rlnAngleRot"] = coords[:, 3]
        outputStarFile["rlnAngleTilt"] = coords[:, 4]
        outputStarFile["rlnAnglePsi"] = coords[:, 5]
    else:
        outputStarFile["rlnCoordinateX"] = []
        outputStarFile["rlnCoordinateY"] = []
        outputStarFile["rlnCoordinateZ"] = []

        # Angles
        outputStarFile["rlnAngleRot"] = []
        outputStarFile["rlnAngleTilt"] = []
        outputStarFile["rlnAnglePsi"] = []

    # Write
    if tomoID == None:
        savePath = os.path.join(outputDirectory, f"{specimen}.star")
    else:
        savePath = os.path.join(outputDirectory, tomoID, f"{tomoID}_{specimen}.star")
    starfile.write({"particles": pd.DataFrame(outputStarFile)}, savePath)


def write_copick_output(
    specimen,
    tomoID,
    finalPicks,
    outputDirectory="refinedCoPicks/ExperimentRuns",
    pickMethod="deepfinder",
    sessionID="0",
    knownTemplate=False,
):
    """
    Writes the output data from 3D protein picking algorithm into a JSON file.

    Args:
    - specimen (str): The name of the specimen.
    - tomoID (str): The ID of the tomogram.
    - finalPicks (np.ndarray): An array of final picks, where each row contains coordinates (x, y, z) and Euler angles (rotation, tilt, psi).
    - outputDirectory (str): The directory where the output JSON file will be saved.
    - pickMethod (str): The method used for picking. Default is 'deepfinder'.
    - sessionID (str): The ID of the session. Default is '0'.
    - knownTemplate (bool): A flag indicating whether the template is known. Default is False.
    """

    # Define the JSON structure
    json_data = {
        "pickable_object_name": specimen,
        "user_id": pickMethod,
        "session_id": sessionID,
        "run_name": tomoID,
        "voxel_spacing": None,
        "unit": "angstrom",
    }
    if not knownTemplate:
        json_data["trust_orientation"] = "false"

    json_data["points"] = []
    for ii in range(finalPicks.shape[0]):
        rotationMatrix = convert_euler_to_rotation_matrix(
            finalPicks[ii, 3], finalPicks[ii, 4], finalPicks[ii, 5]
        )

        # Append to points data
        json_data["points"].append({
            "location": {
                "x": finalPicks[ii, 0],
                "y": finalPicks[ii, 1],
                "z": finalPicks[ii, 2],
            },
            "transformation_": rotationMatrix,  # Convert matrix to list for JSON serialization
            "instance_id": 0,
            "score": 1.0,
        })

    # Generate custom formatted JSON
    formatted_json = custom_format_json(json_data)

    # Save to file
    os.makedirs(os.path.join(outputDirectory, tomoID, "Picks"), exist_ok=True)
    savePath = os.path.join(
        outputDirectory, tomoID, "Picks", f"{pickMethod}_{sessionID}_{specimen}.json"
    )
    with open(savePath, "w") as json_file:
        json_file.write(formatted_json)


def custom_format_json(data):
    result = "{\n"
    for key, value in data.items():
        if key == "points":
            result += '   "{}": [\n'.format(key)
            for point in value:
                result += "      {\n"
                for p_key, p_value in point.items():
                    if p_key in ["location", "transformation_"]:
                        if p_key == "location":
                            loc_str = ", ".join([
                                '"{}": {}'.format(k, v) for k, v in p_value.items()
                            ])
                            result += '         "{}": {{ {} }},\n'.format(p_key, loc_str)
                        if p_key == "transformation_":
                            trans_str = ",\n            ".join([
                                "[{}]".format(", ".join(map(str, row))) for row in p_value
                            ])
                            result += '         "{}": [\n            {}\n         ],\n'.format(
                                p_key, trans_str
                            )
                    else:
                        result += '         "{}": {},\n'.format(p_key, json.dumps(p_value))
                result = result.rstrip(",\n") + "\n      },\n"
            result = result.rstrip(",\n") + "\n   ]\n"
        else:
            result += '   "{}": {},\n'.format(key, json.dumps(value))
    result = result.rstrip(",\n") + "\n}"
    return result


def convert_euler_to_rotation_matrix(angleRot, angleTilt, anglePsi):
    """
    Convert Euler angles (rotation, tilt, and psi) in the Relion 'ZYZ' convention into a 4x4 rotation matrix.

    Parameters:
    - angleRot (float): The rotation angle (in degrees) about the Z-axis.
    - angleTilt (float): The tilt angle (in degrees) about the Y-axis.
    - anglePsi (float): The psi angle (in degrees) about the Z-axis.

    Returns:
    - np.ndarray: Rotation matrix.
    """

    rotation = R.from_euler("zyz", [angleRot, angleTilt, anglePsi], degrees=True)
    rotation_matrix = rotation.as_matrix()

    # Append a zero column to the right and zero row at the bottom
    new_column = np.zeros((3, 1))
    new_row = np.zeros((1, 4))
    rotation_matrix = np.vstack((np.hstack((rotation_matrix, new_column)), new_row))

    # Set the last element to 1
    rotation_matrix[-1, -1] = 1
    rotation_matrix = np.round(rotation_matrix, 3)

    return rotation_matrix


def ome_zarr_axes() -> List[Dict[str, str]]:
    """
    Returns a list of dictionaries defining the axes information for an OME-Zarr dataset.

    Returns:
    - List[Dict[str, str]]: A list of dictionaries, each specifying the name, type, and unit of an axis.
      The axes are 'z', 'y', and 'x', all of type 'space' and unit 'angstrom'.
    """
    return [
        {
            "name": "z",
            "type": "space",
            "unit": "angstrom",
        },
        {
            "name": "y",
            "type": "space",
            "unit": "angstrom",
        },
        {
            "name": "x",
            "type": "space",
            "unit": "angstrom",
        },
    ]


def ome_zarr_transforms(voxel_size: float) -> List[Dict[str, Any]]:
    """
    Return a list of dictionaries defining the coordinate transformations of OME-Zarr dataset.

    Parameters:
    - voxel_size (float): The size of a voxel.

    Returns:
    - List[Dict[str, Any]]: A list containing a single dictionary with the 'scale' transformation,
      specifying the voxel size for each axis and the transformation type as 'scale'.
    """
    return [{"scale": [voxel_size, voxel_size, voxel_size], "type": "scale"}]


def write_ome_zarr_segmentation(
    run,
    inputSegmentVol,
    voxelSize=10,
    segmentationName="segmentation",
    userID="deepfinder",
    sessionID="0",
):
    """
    Write a OME-Zarr segmentation into a Copick Directory.

    Parameters:
    - run: The run object, which provides a method to create a new segmentation.
    - segmentation: The segmentation data to be written.
    - voxelsize (float): The size of the voxels. Default is 10.
    """

    # Create a new segmentation or Read Previous Segmentation
    try:
        seg = run.new_segmentation(
            voxel_size=voxelSize,
            name=segmentationName,
            session_id=sessionID,
            is_multilabel=True,
            user_id=userID,
        )
    except:
        seg = run.get_segmentations(name=segmentationName, user_id=userID, session_id=sessionID)[0]

    # Write the zarr file
    loc = seg.zarr()
    root_group = zarr.group(loc, overwrite=True)

    ome_zarr.writer.write_multiscale(
        [inputSegmentVol],
        group=root_group,
        axes=ome_zarr_axes(),
        coordinate_transformations=[ome_zarr_transforms(voxelSize)],
        storage_options=dict(chunks=(256, 256, 256), overwrite=True),
        compute=True,
    )
