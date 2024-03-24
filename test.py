from rich.progress import Progress
import time
# 创建一个进度条对象
progress = Progress()

task = progress.add_task("[cyan]Working...", total=100)

# 开始进度条
progress.start()

for i in range(100):
    # 模拟任务的进行
    time.sleep(0.1)
    
    # 更新进度条的进度
    progress.update(task, advance=1, description=f"[cyan]Processing item {i}")

# 完成进度条
progress.stop()
