#!/usr/bin/env python3
"""
Upload Model to Firebase Storage

Creates a .zip archive of the agent_states_multiprocess folder and uploads
it to Firebase Storage in the models directory.

Usage:
    python scripts/upload_model_to_storage.py [path_to_folder] [--name MODEL_NAME]

Examples:
    python scripts/upload_model_to_storage.py /path/to/agent_states_multiprocess
    python scripts/upload_model_to_storage.py ./agent_states_multiprocess --name my_model_v1
"""

import sys
import os
import argparse
import logging
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import firebase_admin
from firebase_admin import credentials, storage

from config.firebase_config import get_service_account_path, PROJECT_ID

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_zip_archive(source_folder: Path, output_path: Path) -> Path:
    """
    Create a zip archive of the source folder.

    Args:
        source_folder: Path to the folder to zip
        output_path: Path for the output zip file

    Returns:
        Path to the created zip file
    """
    logger.info(f"Creating zip archive of {source_folder}...")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(source_folder.parent)
                zipf.write(file_path, arcname)
                logger.debug(f"  Added: {arcname}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Created zip archive: {output_path} ({size_mb:.2f} MB)")

    return output_path


def initialize_firebase():
    """Initialize Firebase Admin SDK."""
    if not firebase_admin._apps:
        service_account_path = get_service_account_path()
        logger.info(f"Initializing Firebase with service account: {service_account_path}")

        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': f'{PROJECT_ID}.appspot.com'
        })

    return storage.bucket()


def upload_to_storage(bucket, local_path: Path, remote_path: str) -> str:
    """
    Upload a file to Firebase Storage.

    Args:
        bucket: Firebase Storage bucket
        local_path: Path to the local file
        remote_path: Destination path in Firebase Storage

    Returns:
        Public URL of the uploaded file
    """
    logger.info(f"Uploading {local_path} to gs://{bucket.name}/{remote_path}...")

    blob = bucket.blob(remote_path)

    # Upload with progress indication for large files
    file_size = local_path.stat().st_size
    logger.info(f"File size: {file_size / (1024 * 1024):.2f} MB")

    blob.upload_from_filename(str(local_path))

    # Make the blob publicly accessible (optional)
    # blob.make_public()

    logger.info(f"Upload complete!")
    logger.info(f"Storage path: gs://{bucket.name}/{remote_path}")

    # Generate a signed URL valid for 7 days
    from datetime import timedelta
    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(days=7),
        method="GET"
    )

    return signed_url


def main():
    parser = argparse.ArgumentParser(
        description="Upload agent_states folder to Firebase Storage"
    )
    parser.add_argument(
        "folder",
        help="Path to the agent_states_multiprocess folder to upload"
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Custom name for the uploaded model (default: folder name with timestamp)"
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the local zip file after upload"
    )
    args = parser.parse_args()

    # Validate source folder
    source_folder = Path(args.folder).resolve()
    if not source_folder.exists():
        logger.error(f"Folder not found: {source_folder}")
        return 1

    if not source_folder.is_dir():
        logger.error(f"Not a directory: {source_folder}")
        return 1

    logger.info(f"Source folder: {source_folder}")

    # Generate model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.name:
        model_name = args.name
    else:
        model_name = f"{source_folder.name}_{timestamp}"

    zip_filename = f"{model_name}.zip"
    remote_path = f"models/{zip_filename}"

    logger.info(f"Model name: {model_name}")
    logger.info(f"Remote path: {remote_path}")

    try:
        # Initialize Firebase
        bucket = initialize_firebase()

        # Create temporary zip file
        if args.keep_zip:
            zip_path = Path.cwd() / zip_filename
        else:
            temp_dir = tempfile.mkdtemp()
            zip_path = Path(temp_dir) / zip_filename

        # Create zip archive
        create_zip_archive(source_folder, zip_path)

        # Upload to Firebase Storage
        signed_url = upload_to_storage(bucket, zip_path, remote_path)

        # Cleanup
        if not args.keep_zip:
            os.remove(zip_path)
            os.rmdir(temp_dir)
            logger.info("Cleaned up temporary files")
        else:
            logger.info(f"Kept local zip file: {zip_path}")

        logger.info("\n" + "=" * 60)
        logger.info("UPLOAD SUCCESSFUL")
        logger.info("=" * 60)
        logger.info(f"Storage path: gs://{bucket.name}/{remote_path}")
        logger.info(f"Signed URL (valid 7 days):\n{signed_url}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
