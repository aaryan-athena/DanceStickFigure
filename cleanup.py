import os
import shutil

def cleanup_project():
    """Remove unnecessary files and directories to optimize the project"""
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Directories to remove (these are recreated as needed)
    dirs_to_remove = [
        "poses",           # No longer needed - we process in memory
        "output_frames",   # No longer needed - we generate video directly
    ]
    
    # Files to remove (optional - keep old pipeline as backup)
    files_to_remove = [
        "stick_figure_maker.py",  # Replaced by optimized pipeline
    ]
    
    print("🧹 Cleaning up project...")
    
    # Remove directories
    for dir_name in dirs_to_remove:
        dir_path = os.path.join(BASE_DIR, dir_name)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"✅ Removed directory: {dir_name}")
    
    # Remove files
    for file_name in files_to_remove:
        file_path = os.path.join(BASE_DIR, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ Removed file: {file_name}")
    
    print("🎉 Project cleanup complete!")
    print("\n📁 Optimized project structure:")
    print("- optimized_app.py (main app)")
    print("- optimized_pipeline.py (fast processing)")
    print("- colab_pose_pipeline.py (backup - can be removed)")
    print("- input_videos/ (user uploads)")
    print("- output_videos/ (generated results)")
    print("- pages/ (additional features)")

if __name__ == "__main__":
    cleanup_project()
