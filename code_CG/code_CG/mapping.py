import os
import pandas as pd
import shutil
import re
# 读取Excel文件
file_path = '/home/lyf/code_cg/5分类名单.xlsx'
df = pd.read_excel(file_path)

# 用于记录未找到的文件名
not_found = {column: [] for column in df.columns}

# 设置基础路径，修改为实际的目标路径
destination_base_path = 'data_new'
source_base_path = 'data'
# 遍历每一列，创建分类文件夹并移动文件
for column in df.columns:
    category = column
    
    for index, name in df[column].dropna().items():
        file_found = False
        # 假设文件名包含扩展名，如 file_name.jpg
        for file_name in os.listdir(source_base_path):
            match = re.search(r'([\u4e00-\u9fa5]+)', file_name)
            if match and name in match.group(1):
                source_file_path = os.path.join(source_base_path, file_name)
        
                # 创建目标文件夹路径 
                destination_folder = os.path.join(destination_base_path, category)
            
                # 创建目标文件夹（如果不存在）
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                
                # 移动文件到目标文件夹
                if os.path.exists(source_file_path):
                    shutil.move(source_file_path, destination_folder)
                    file_found = True
                else:
                    print(f"File {source_file_path} does not exist and cannot be moved.")

        if not file_found:
            not_found[column].append(name)

not_found_df = pd.DataFrame({col: pd.Series(names) for col, names in not_found.items()})
not_found_file_path = '/home/lyf/code_cg/not_found_files.xlsx'  # 请修改为实际路径
not_found_df.to_excel(not_found_file_path, index=False)
print("File categorization and moving completed.")
