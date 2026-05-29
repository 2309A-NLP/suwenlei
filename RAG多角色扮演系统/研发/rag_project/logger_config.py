# -*- coding: utf-8 -*-
# logger_config.py日志配置
import logging

def setup_logger(name=__name__):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)

logger = setup_logger()