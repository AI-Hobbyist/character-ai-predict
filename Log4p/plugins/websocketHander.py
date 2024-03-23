import logging
import asyncio
import websockets

class WebsocketHandler(logging.Handler):
    def __init__(self, server_address):
        super().__init__()
        self.server_address = server_address
    
    async def send_log_async(self, message):
        async with websockets.connect(self.server_address) as websocket:
            await websocket.send(message)
    
    def send_log_sync(self, message):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.send_log_async(message))

    def emit(self, record):
        log_entry = self.format(record)
        
        if asyncio.get_event_loop().is_running():
            asyncio.create_task(self.send_log_async(log_entry))
        else:
            try:
                self.send_log_sync(log_entry)
            except Exception as e:
                logging.warning("Failed to send log synchronously: %s", e)