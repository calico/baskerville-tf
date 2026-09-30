# taken and modified from calico-ukbb-mri-ml repo
# https://github.com/calico/calicolabs-ukbb-mri-ml/tree/main/src/ukbb_mri_ml/helpers
# =========================================================================

import os
import logging
from base64 import b64decode
from json import loads
from os.path import exists, join, isfile
from re import match
from typing import List

from google.cloud.storage import Client
from google.auth.exceptions import DefaultCredentialsError

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def _get_storage_client() -> Client:
    """
    Returns: Google Cloud Storage Client
    """
    try:
        # Attempt to infer credentials from environment
        storage_client = Client()
        logger.info("Inferred credentials from environment")
    except DefaultCredentialsError:
        try:
            # Attempt to load JSON credentials from GOOGLE_APPLICATION_CREDENTIALS
            storage_client = Client.from_service_account_info(
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            )
            logger.info("Loaded credentials from GOOGLE_APPLICATION_CREDENTIALS")
        except AttributeError:
            # Attempt to load JSON credentials from base64 encoded string
            storage_client = Client.from_service_account_info(
                loads(
                    b64decode(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).decode(
                        "utf-8"
                    )
                )
            )
            logger.info("Loaded credentials from base64 encoded string")
    return storage_client


def download_from_gcs(gcs_path: str, local_path: str, bytes=True) -> None:
    """
    Downloads a file from GCS
    Args:
        gcs_path: string path to GCS file to download
        local_path: string path to download to
        bytes: boolean flag indicating if gcs file contains bytes

    Returns: None

    """
    storage_client = _get_storage_client()
    write_mode = "wb" if bytes else "w"
    with open(local_path, write_mode) as o:
        storage_client.download_blob_to_file(gcs_path, o)


def download_folder_from_gcs(gcs_dir: str, local_dir: str, bytes=True) -> None:
    """
    Downloads a whole folder from GCS
    Args:
        gcs_dir: string path to GCS folder to download
        local_dir: string path to download to
        bytes: boolean flag indicating if gcs file contains bytes

    Returns: None

    """
    storage_client = _get_storage_client()
    if not is_gcs_path(gcs_dir):
        raise ValueError(f"gcs_dir is not a valid GCS path: {gcs_dir}")
    bucket_name, gcs_object_prefix = split_gcs_uri(gcs_dir)
    # Get the bucket from the client.
    bucket = storage_client.bucket(bucket_name)

    # Ensure local folder exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    # List all blobs with the given prefix (i.e., folder path).
    blobs = bucket.list_blobs(prefix=gcs_object_prefix)
    # Download each blob.
    for blob in blobs:
        # Compute the full path to which we'll download the blob.
        blob_rel_path = os.path.relpath(blob.name, gcs_object_prefix)
        local_blob_path = os.path.join(local_dir, blob_rel_path)

        # Ensure the local directory structure exists
        local_blob_dir = os.path.dirname(local_blob_path)
        if not os.path.exists(local_blob_dir):
            os.makedirs(local_blob_dir)
        download_from_gcs(join(gcs_dir, blob_rel_path), local_blob_path, bytes=bytes)


def sync_dir_to_gcs(
    local_dir: str, gcs_dir: str, verbose=False, recursive=False
) -> None:
    """
    Copies all files in a local directory to the gcs directory
    Args:
        local_dir: string local directory path to upload from
        gcs_dir: string GCS destination path. Will create folders that do not exist.
        verbose: boolean flag to print logging statements
        recursive: boolean flag to recursively upload files in subdirectories

    Returns: None

    """
    storage_client = _get_storage_client()
    if not is_gcs_path(gcs_dir):
        raise ValueError(f"gcs_dir is not a valid GCS path: {gcs_dir}")

    if not exists(local_dir):
        raise FileNotFoundError(f"local_dir does not exist: {local_dir}")

    local_files = os.listdir(local_dir)
    bucket_name, gcs_object_prefix = split_gcs_uri(gcs_dir)
    bucket = storage_client.bucket(bucket_name)

    for filename in local_files:
        gcs_object_name = join(gcs_object_prefix, filename)
        local_file = join(local_dir, filename)
        if recursive and not isfile(local_file):
            sync_dir_to_gcs(
                local_file,
                f"gs://{join(bucket_name, gcs_object_name)}",
                verbose=verbose,
                recursive=recursive,
            )
        elif not isfile(local_file):
            pass
        else:
            blob = bucket.blob(gcs_object_name)
            if verbose:
                print(
                    f"Uploading {local_file} to gs://{join(bucket_name, gcs_object_name)}"
                )
            blob.upload_from_filename(local_file)


def upload_folder_gcs(local_dir: str, gcs_dir: str) -> None:
    """
    Copies all files in a local directory to the gcs directory
    Args:
        local_dir: string local directory path to upload from
        gcs_dir: string GCS destination path. Will create folders that do not exist.
    Returns: None
    """
    storage_client = _get_storage_client()
    bucket_name = gcs_dir.split("//")[1].split("/")[0]
    gcs_object_prefix = "/".join(gcs_dir.split("//")[1].split("/")[1:])
    local_prefix = local_dir.split("/")[-1]
    bucket = storage_client.bucket(bucket_name)
    for filename in os.listdir(local_dir):
        gcs_object_name = f"{gcs_object_prefix}/{local_prefix}/{filename}"
        local_file = join(local_dir, filename)
        blob = bucket.blob(gcs_object_name)
        blob.upload_from_filename(local_file)


def upload_file_gcs(local_path: str, gcs_path: str, bytes=True) -> None:
    """
    Upload a file to gcs
    Args:
        local_path: local path to file
        gcs_path: string GCS Uri follows the format gs://$BUCKET_NAME/OBJECT_NAME

    Returns: None
    """
    storage_client = _get_storage_client()
    bucket_name = gcs_path.split("//")[1].split("/")[0]
    bucket = storage_client.bucket(bucket_name)
    gcs_object_prefix = "/".join(gcs_path.split("//")[1].split("/")[1:])
    filename = local_path.split("/")[-1]
    blob = bucket.blob(f"{gcs_object_prefix}/{filename}")
    blob.upload_from_filename(local_path)


def gcs_join(*args):
    args = [arg.replace("gs://", "").strip("/") for arg in args]
    return "gs://" + join(*args)


def split_gcs_uri(gcs_uri: str) -> tuple:
    """
    Splits a GCS bucket and object_name from a GCS URI
    Args:
        gcs_uri: string GCS Uri follows the format gs://$BUCKET_NAME/OBJECT_NAME

    Returns: bucket_name, object_name
    """
    matches = match("gs://(.*?)/(.*)", gcs_uri)
    if matches:
        return matches.groups()
    else:
        raise ValueError(
            f"{gcs_uri} does not match expected format: gs://BUCKET_NAME/OBJECT_NAME"
        )


def is_gcs_path(gcs_path: str) -> bool:
    """
    Returns True if the string passed starts with gs://
    Args:
        gcs_path: string path to check

    Returns: Boolean flag indicating the gcs_path starts with gs://

    """
    return gcs_path.startswith("gs://")


def get_filename_in_dir(files_dir: str, recursive: bool = False) -> List[str]:
    """
    Returns list of filenames inside a directory.
    """
    # currently only Niftidataset receives gs bucket paths, so this isn't necessary
    # commenting out for now even though it is functional (lots of files)
    storage_client = _get_storage_client()
    if is_gcs_path(files_dir):
        bucket_name, object_name = split_gcs_uri(files_dir)
        blob_iterator = storage_client.list_blobs(bucket_name, prefix=object_name)
        return [str(blob) for blob in blob_iterator]
    dir_contents = os.listdir(files_dir)
    files = []
    for entry in dir_contents:
        entry_path = join(files_dir, entry)
        if isfile(entry_path):
            files.append(entry_path)
        elif recursive:
            files.extend(get_filename_in_dir(entry_path, recursive=recursive))
        else:
            print("Nothing happened here")
            pass
    return files


def download_rename_inputs(filepath: str, temp_dir: str, is_dir: bool = False) -> str:
    """
    Download file from gcs to local dir
    Args:
        filepath: GCS Uri follows the format gs://$BUCKET_NAME/OBJECT_NAME
        temp_dir: local dir to download to
        is_dir: boolean flag indicating if the filepath is a directory
    Returns: new filepath in the local machine
    """
    if is_dir:
        dir_name = filepath.split("/")[-1]
        download_folder_from_gcs(filepath, f"{temp_dir}/{dir_name}")
        return f"{temp_dir}/{dir_name}"
    else:
        _, filename = split_gcs_uri(filepath)
        if "/" in filename:
            filename = filename.split("/")[-1]
        download_from_gcs(filepath, f"{temp_dir}/{filename}")
        return f"{temp_dir}/{filename}"


def gcs_file_exist(gcs_path: str) -> bool:
    """
    check if a file exist in gcs
    params: gcs_path
    returns: true/false
    """
    storage_client = _get_storage_client()
    bucket, filename = split_gcs_uri(gcs_path)
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(filename)
    return blob.exists()
