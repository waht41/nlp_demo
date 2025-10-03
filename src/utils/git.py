import os
import subprocess


def get_git_info(output_dir):
    """获取当前的 git commit hash 和未提交的修改，并保存到结果目录"""
    try:
        # 1. 获取 git commit hash
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')

        # 2. 将未提交的修改保存为一个 patch 文件
        # 这对于复现至关重要，因为它记录了在最后一次 commit 之后的所有代码改动
        diff_path = os.path.join(output_dir, "code_changes.patch")
        with open(diff_path, "w") as f:
            subprocess.run(['git', 'diff', 'HEAD'], stdout=f)

        print(f"✅ Git Hash ({git_hash}) 已记录。")
        if os.path.getsize(diff_path) > 0:
            print(f"⚠️ 发现未提交的代码修改，已保存至: {diff_path}")
        else:
            os.remove(diff_path)  # 如果没有改动，则删除空的 patch 文件

        return git_hash
    except subprocess.CalledProcessError:
        print("❓ 未能获取 Git 信息。可能不是一个 Git 仓库。")
        return "N/A"
