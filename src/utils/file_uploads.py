import os
import uuid
from werkzeug.utils import secure_filename
from typing import List, Optional, Dict, Any

def save_uploaded_file(file, directory: str, prefix: str = "") -> Optional[str]:
    """
    Save an uploaded file to the specified directory with a unique name.
    
    Args:
        file: The file object from request.files
        directory: The directory to save the file to
        prefix: Optional prefix for the filename
        
    Returns:
        The path to the saved file or None if unsuccessful
    """
    if not file or file.filename == '':
        return None
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Generate a unique filename
    filename = secure_filename(f"{prefix}_{uuid.uuid4()}_{file.filename}" if prefix else f"{uuid.uuid4()}_{file.filename}")
    file_path = os.path.join(directory, filename)
    
    # Save the file
    try:
        file.save(file_path)
        return file_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return None

def save_multiple_files(files: Dict[str, Any], directory: str) -> List[str]:
    """
    Save multiple uploaded files and return their paths.
    
    Args:
        files: Dictionary of files from request.files
        directory: Directory to save files to
        
    Returns:
        List of saved file paths
    """
    saved_paths = []
    
    for key, file in files.items():
        if file and file.filename != '':
            path = save_uploaded_file(file, directory, prefix=key)
            if path:
                saved_paths.append(path)
    
    return saved_paths

def allowed_file(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Check if a filename has an allowed extension.
    
    Args:
        filename: The filename to check
        allowed_extensions: List of allowed extensions (e.g., ['jpg', 'png'])
        
    Returns:
        True if the file has an allowed extension, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions 