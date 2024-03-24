import time
from functools import wraps



def retry(max_attempts=3, delay=1, backoff=2):
    '''
    # retry装饰器
    ### 在函数发生异常的时候,受到retry装饰的函数会尝试直到被装饰的函数成功运行或者达到最大尝试次数
    - max_attempts: 指定最大尝试次数
    - delay: 尝试延迟
    - backoff : 不知道
    '''
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"尝试次数: {attempt}")
                    print(f"异常信息: {e}")
                    attempt += 1
                    time.sleep(delay * backoff ** (attempt - 2))
            raise Exception(f"函数 {func.__name__} 执行失败")
        return wrapper
    return decorator
