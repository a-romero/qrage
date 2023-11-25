import os
import requests
import zipfile
import tarfile
import datetime

def process_paths(path_array):
    result_paths = []

    def download_and_extract_http_file(http_url, target_dir):
        local_filename = http_url.split('/')[-1]
        local_path = os.path.join(target_dir, local_filename)

        # Download the file from the HTTP URL
        with requests.get(http_url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Decompress if necessary and remove the compressed file
        if zipfile.is_zipfile(local_path):
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            os.remove(local_path)
        elif tarfile.is_tarfile(local_path):
            with tarfile.open(local_path, 'r') as tar_ref:
                tar_ref.extractall(target_dir)
            os.remove(local_path)

    def process_path(path):
        # Check if the path is a file
        if os.path.isfile(path):
            result_paths.append(path)
        # Check if the path is a directory
        elif os.path.isdir(path):
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                # Recursively process each file/directory
                process_path(file_path)

    for path in path_array:
        if path.startswith("http://") or path.startswith("https://"):
            # Handle HTTP URL
            date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            http_target_dir = f"./data/http_{date_str}"
            os.makedirs(http_target_dir, exist_ok=True)
            download_and_extract_http_file(path, http_target_dir)
            process_path(http_target_dir)
        else:
            # Handle regular file or directory path
            process_path(path)

    return result_paths


def classify_files(path_array):
    txt_files = []
    pdf_files = []
    md_files = []
    docx_files = []

    for path in path_array:
        if path.endswith('.txt'):
            txt_files.append(path)
        elif path.endswith('.pdf'):
            pdf_files.append(path)
        elif path.endswith('.md'):
            md_files.append(path)
        elif path.endswith('.docx'):
            docx_files.append(path)

    return txt_files, pdf_files, md_files, docx_files