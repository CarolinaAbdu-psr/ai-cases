"""
Common RAG utilities shared across different agent types.
This module contains shared functionality for vectorstore management.

Important: ChromaDB keeps SQLite database files locked. Before downloading new vectorstores,
call release_chromadb_handles() or ensure all Chroma objects are deleted and gc.collect()
is called to release file locks.
"""

import datetime as dt
import os
import logging
import shutil
import zipfile
from typing import List, Tuple
import gc
import time

import api_s3

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def release_chromadb_handles():
    """Force release of ChromaDB file handles.

    Call this before attempting to download/update vectorstores to ensure
    SQLite database files are not locked by ChromaDB clients.

    Example:
        >>> from helper import rag_common
        >>> rag_common.release_chromadb_handles()
        >>> rag_common.download_latest_rag(...)
    """
    logger.info("Forcing garbage collection to release ChromaDB file handles...")
    gc.collect()
    time.sleep(1.0)  # Give OS time to release file handles
    logger.info("File handles released")


def extract_rag_to_folder(rag_name: str, folder_path: str) -> bool:
    """Extract vectorstore from S3 to vectorstores/source_type folder

    This function provides a robust download mechanism with:
    1. Automatic backup of existing vectorstore if present
    2. Cleanup of old files before downloading to prevent conflicts
    3. Download and extraction of new vectorstore from S3
    4. Automatic restore from backup if download/extraction fails

    Important: This function automatically calls release_chromadb_handles() to ensure
    ChromaDB files are not locked before attempting backup/cleanup operations.

    Args:
        rag_name: Name of the RAG file in S3 (e.g., 'rag_factory_2026-02-01_10-30-00.zip')
        folder_path: Local path where vectorstore should be extracted

    Returns:
        bool: True if download/extraction succeeded, False otherwise

    Example:
        >>> extract_rag_to_folder('rag_factory_2026-02-01_10-30-00.zip', 'vectorstores/factory_vectorstore')
        True
    """
    backup_path = None
    zip_path = None

    try:
        # CRITICAL: Release any ChromaDB file handles before backup/cleanup
        logger.info("Preparing for vectorstore download...")
        release_chromadb_handles()

        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Check if there's an existing vectorstore to backup
        has_existing_data = False
        if os.path.exists(folder_path):
            existing_files = [f for f in os.listdir(folder_path) if not f.endswith('.zip')]
            has_existing_data = len(existing_files) > 0

        # Backup existing vectorstore before cleanup
        if has_existing_data:
            backup_path = folder_path + '_backup'
            logger.info(f"Backing up existing vectorstore to {backup_path}")

            # Remove old backup if exists
            if os.path.exists(backup_path):
                try:
                    shutil.rmtree(backup_path)
                except Exception as e:
                    logger.warning(f"Could not remove old backup, trying to continue: {e}")

            # Create backup with retry logic for locked files
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.copytree(folder_path, backup_path, ignore=shutil.ignore_patterns('*.zip'))
                    logger.info(f"Backup created at {backup_path}")
                    break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Backup attempt {attempt + 1} failed (file locked), retrying in 2 seconds...")
                        gc.collect()
                        time.sleep(2)
                    else:
                        logger.error(f"Could not create backup after {max_retries} attempts: {e}")
                        logger.warning("Proceeding with download without backup (risky)")
                        backup_path = None
                        break

        # Cleanup: Remove old vectorstore files (except zip files)
        if has_existing_data:
            logger.info(f"Cleaning up old vectorstore files in {folder_path}")

            # Try to remove files with retry logic
            for item in os.listdir(folder_path):
                if item.endswith('.zip'):
                    continue

                item_path = os.path.join(folder_path, item)

                # Retry logic for locked files
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        break  # Success
                    except (PermissionError, OSError) as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Cleanup attempt {attempt + 1} failed for {item} (file locked), retrying...")
                            gc.collect()
                            time.sleep(2)
                        else:
                            logger.error(f"Could not remove {item} after {max_retries} attempts: {e}")
                            # If we can't cleanup, we should restore backup and abort
                            if backup_path and os.path.exists(backup_path):
                                logger.error("Cleanup failed, aborting download")
                                raise PermissionError(f"Cannot cleanup locked file: {item}")

            logger.info(f"Cleanup completed")

        # Download new vectorstore
        zip_path = os.path.join(folder_path, rag_name)
        logger.info(f"Downloading RAG {rag_name} from S3...")
        api_s3.download_file_from_s3(rag_name, zip_path)
        logger.info(f"RAG {rag_name} downloaded to {folder_path}")

        # Ensure zip extension
        if not zip_path.endswith('.zip'):
            zip_path += '.zip'

        # Extract new vectorstore
        logger.info(f"Extracting RAG {rag_name} from {zip_path} to {folder_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        logger.info(f"RAG {rag_name} extracted successfully to {folder_path}")

        # Remove the zip file after successful extraction
        os.remove(zip_path)

        # Remove backup after successful download
        if backup_path and os.path.exists(backup_path):
            logger.info(f"Removing backup at {backup_path}")
            shutil.rmtree(backup_path)

        return True

    except Exception as e:
        logger.error(f"Error extracting RAG {rag_name}: {str(e)}")

        # Restore from backup if download/extraction failed
        if backup_path and os.path.exists(backup_path):
            logger.warning(f"Download failed, restoring previous vectorstore from {backup_path}")
            try:
                # Clean up failed download
                if os.path.exists(folder_path):
                    for item in os.listdir(folder_path):
                        item_path = os.path.join(folder_path, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)

                # Restore backup
                for item in os.listdir(backup_path):
                    src = os.path.join(backup_path, item)
                    dst = os.path.join(folder_path, item)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src):
                        shutil.copytree(src, dst)

                logger.info(f"Previous vectorstore restored successfully")

                # Remove backup after restoration
                shutil.rmtree(backup_path)
            except Exception as restore_error:
                logger.error(f"Error restoring backup: {str(restore_error)}")
        else:
            # Clean up partial download if no backup exists
            if zip_path and os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except:
                    pass

        return False


def get_rag_list() -> List[str]:
    """Get list of all vectorstores from S3

    Returns:
        List[str]: List of RAG filenames in S3 bucket
    """
    try:
        rag_list = api_s3.list_files_in_s3()
        return rag_list
    except Exception as e:
        logger.error(f"Error getting RAG list from S3: {str(e)}")
        return []


def get_rag_list_with_dates(rag_list: List[str], source_type: str = None) -> List[Tuple[dt.datetime, str]]:
    """Get a list sorted by date of the available RAGs of a source type

    Supports both old and new naming formats:
    - Old format: rag_YYYY-MM-DD_HH-MM-SS.zip
    - New format: rag_{source_type}_YYYY-MM-DD_HH-MM-SS.zip

    Args:
        rag_list: List of RAG filenames from S3
        source_type: Filter by source type (knowledge_hub, factory, psrio, etc.)

    Returns:
        List[Tuple[datetime, str]]: Sorted list of (date, filename) tuples, newest first
    """
    rag_with_dates = []
    for rag in rag_list:
        try:
            if not rag.startswith('rag_') or not rag.endswith('.zip'):
                continue
            rag_basename = rag.replace('.zip', '')

            # Check if this is old format first (rag_YYYY-MM-DD_HH-MM-SS)
            splitted_name = rag_basename.split('_')
            if len(splitted_name) == 3 and len(splitted_name[1]) == 10 and '-' in splitted_name[1]:
                # Old format: rag_YYYY-MM-DD_HH-MM-SS
                if source_type is not None:
                    continue  # Skip old format files when filtering by source_type
                date_str = splitted_name[1] + ' ' + splitted_name[2].replace('-', ':')
                date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                rag_with_dates.append((date_obj, rag))
                continue

            # New format: rag_{source_type}_{YYYY-MM-DD}_{HH-MM-SS}
            # Find the last two parts that look like date and time
            date_part = None
            time_part = None
            extracted_source_type = None

            if len(splitted_name) >= 4:
                # Try to find date pattern (YYYY-MM-DD) and time pattern (HH-MM-SS)
                for i in range(len(splitted_name) - 1, 0, -1):
                    part = splitted_name[i]
                    if len(part) == 8 and part.count('-') == 2:  # HH-MM-SS
                        time_part = part
                        if i > 0 and len(splitted_name[i-1]) == 10 and splitted_name[i-1].count('-') == 2:  # YYYY-MM-DD
                            date_part = splitted_name[i-1]
                            # Everything between 'rag' and date_part is the source_type
                            extracted_source_type = '_'.join(splitted_name[1:i-1])
                            break

            if not (date_part and time_part and extracted_source_type):
                continue

            # If source_type is specified, filter by it
            if source_type and extracted_source_type != source_type:
                continue

            date_str = date_part + ' ' + time_part.replace('-', ':')
            date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            rag_with_dates.append((date_obj, rag))

        except Exception as e:
            logger.error(f"Error processing RAG {rag}: {str(e)}")
            continue

    return sorted(rag_with_dates, key=lambda x: x[0], reverse=True)


def get_latest_rag_date(source_type: str = None) -> dt.datetime:
    """Returns the date of the last vectorstore of a source type

    Args:
        source_type: Source type (knowledge_hub, factory, psrio, etc.)

    Returns:
        datetime: Date of the most recent RAG for the source type

    Raises:
        ValueError: If no RAG files found for the source type
    """
    rag_list = get_rag_list()
    rag_with_dates = get_rag_list_with_dates(rag_list, source_type)

    if rag_with_dates:
        date, _ = rag_with_dates[0]
        return date
    else:
        logger.warning(f"No RAG files found for source_type: {source_type}")
        raise ValueError(f"No RAG files found for source_type: {source_type}")


def download_rag(rag_name: str, chroma_dir_name: str, source_type: str = None) -> dt.datetime:
    """Download a specific RAG vectorstore to local vectorstore folder

    Args:
        rag_name: Name of the RAG file to download
        chroma_dir_name: Local directory path for extraction
        source_type: Source type for validation (optional)

    Returns:
        datetime: Date of the downloaded RAG

    Raises:
        ValueError: If RAG name not found in S3
    """
    rag_list = get_rag_list()
    if rag_name not in rag_list:
        logger.error(f"RAG {rag_name} not found in the available RAG list.")
        raise ValueError(f"RAG {rag_name} not found in the available RAG list.")

    rag_with_dates = get_rag_list_with_dates(rag_list, source_type)
    for date, rag in rag_with_dates:
        if rag == rag_name:
            extract_rag_to_folder(rag, chroma_dir_name)
            return date

    logger.error(f"RAG {rag_name} not found in the RAG list with dates.")
    raise ValueError(f"RAG {rag_name} not found in the RAG list with dates.")


def download_latest_rag(chroma_dir_name: str, source_type: str = "factory") -> dt.datetime:
    """Get the latest available RAG from a specific source type

    Supports backward compatibility with old naming format for factory type.

    Args:
        chroma_dir_name: Local directory path for extraction
        source_type: Source type (knowledge_hub, factory, psrio, etc.)

    Returns:
        datetime: Date of the downloaded RAG

    Raises:
        ValueError: If no RAG files found or download fails
    """
    try:
        latest_rag_date = get_latest_rag_date(source_type)

        # Generate RAG name based on source type
        if source_type == "factory":
            # For backward compatibility, try both old and new formats
            rag_name = f"rag_factory_{latest_rag_date.strftime('%Y-%m-%d_%H-%M-%S')}.zip"
            old_rag_name = f"rag_{latest_rag_date.strftime('%Y-%m-%d_%H-%M-%S')}.zip"

            # Check which format exists
            rag_list = get_rag_list()
            if rag_name in rag_list:
                logger.info(f"Downloading most recent Factory RAG (new format): {rag_name}")
                return download_rag(rag_name, chroma_dir_name, source_type)
            elif old_rag_name in rag_list:
                logger.info(f"Downloading most recent Factory RAG (old format): {old_rag_name}")
                return download_rag(old_rag_name, chroma_dir_name, source_type)
            else:
                raise ValueError(f"Neither {rag_name} nor {old_rag_name} found in S3")
        else:
            rag_name = f"rag_{source_type}_{latest_rag_date.strftime('%Y-%m-%d_%H-%M-%S')}.zip"
            logger.info(f"Downloading most recent {source_type} RAG: {rag_name}")
            return download_rag(rag_name, chroma_dir_name, source_type)

    except Exception as e:
        logger.error(f"Error downloading most recent RAG for {source_type}: {str(e)}")
        raise

