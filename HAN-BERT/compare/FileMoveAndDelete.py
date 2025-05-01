import os
import shutil

class FileManager:
    
    def __init__(self, folder_path):
        """
        初始化文件管理器，传入文件夹路径
        :param folder_path: 要操作的文件夹路径
        """
        self.folder_path = folder_path
    
    def delete_txt_files(self):
        """
        删除当前文件夹下所有 .txt 文件
        """
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.folder_path, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    
    def move_txt_files(self, target_folder):
        """
        将当前文件夹下的所有 .txt 文件移动到目标文件夹
        :param target_folder: 目标文件夹路径
        """
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.txt'):
                source_path = os.path.join(self.folder_path, filename)
                target_path = os.path.join(target_folder, filename)
                try:
                    shutil.move(source_path, target_path)
                    print(f"Moved: {source_path} -> {target_path}")
                except Exception as e:
                    print(f"Error moving {source_path}: {e}")

