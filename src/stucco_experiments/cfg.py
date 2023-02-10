def ensure_rviz_resource_path(filepath):
    """Sanitize some path to something in this package to be RVIZ resource loader compatible"""
    # get path after the first instance
    relative_path = filepath.partition("stucco")[2]
    return f"package://stucco/{relative_path.strip('/')}"
