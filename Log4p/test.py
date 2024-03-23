from core import *

if __name__ == '__main__':
    logger = LogManager().GetLogger(log_name='example')
    logger.info('这是一个成功信息')
    logger.debug('这是一个调试信息')
    logger.critical('这是一个严重错误信息')
    logger.error('这是一个错误信息')
    logger.warning('这是一个警告信息')