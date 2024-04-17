import asyncio,requests
from typing import Callable, Any, List
from rich.progress import Progress


class AsyncPoolExecutor:
    '''
    AsyncPoolExecutor类是为了在异步环境中执行同步代码而创建的,
    它可以允许你像collection.futures那样,在异步环境中执行同步代码,同时保证CPU占用不会很高
    '''
    def __init__(self, sync_task: Callable, max_workers: int = 5,task_num : int = 20) -> None:
        self.sync_task = sync_task
        self.task_num = task_num
        self.max_workers = max_workers
        self.queue = asyncio.Queue()
        self.argv_queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()

    async def init_tasks(self, *args:Any, **kwargs:Any) -> None:
        '''初始化任务'''
        for _ in range(self.task_num):
            await self.queue.put((self.sync_task, args, kwargs))

    async def add_difference_argv(self, argv: List[tuple[Any,Any]]) -> None:
        '''支持添加不同的参数'''
        for arg in argv:
            if not isinstance(arg, tuple):
                raise TypeError(f"arg参数必须是tuple,但是你传入了{type(arg).__name__}")
            args , kwargs = arg
            await self.argv_queue.put((args, kwargs))

    async def __call__(self, *args: Any, **kwds: Any) -> Any:
        '''把类变成可以Callable的类型'''
        return await self.run()

    async def run(self,timeout:float = 300) -> List[Any]:
        '''开始执行任务并返回结果'''
        #断言的前提是queue和argv_queue都必须有东西,避免误判
        if self.queue.qsize() != 0 and self.argv_queue.qsize() != 0:
           assert self.queue.qsize() == self.argv_queue.qsize(), "指定的参数总数量和任务数量不一致"
        result = []
        with Progress() as progress:
            q_task = progress.add_task("[cyan]运行中...", total=self.queue.qsize())
            while not self.queue.empty():
                tasks = []
                for _ in range(self.max_workers):
                    try:
                        if self.argv_queue.empty():
                            task, args, kwargs = await self.queue.get()
                            tasks.append(self.loop.run_in_executor(None, task, *args, **kwargs)) # type: ignore
                        else:
                            task, _, _ = await self.queue.get()
                            arg = await self.argv_queue.get()
                            tasks.append(self.loop.run_in_executor(None, task, *arg)) # type: ignore
                    except asyncio.QueueEmpty:
                        break
                if tasks:
                    done, _ = await asyncio.wait(tasks,timeout=timeout)
                    result.extend([t.result() for t in done])
                    progress.update(q_task, advance=len(tasks))
        return result

if __name__ == "__main__":
    '''
    测试结果: HTTP
    CPU: 50% - 60%
    time: 96s
    '''
