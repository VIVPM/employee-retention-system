from datetime import datetime
import logging
import queue

class Logger:
    # A class-level queue to hold live log messages for SSE streaming
    log_queue = queue.Queue()
    """
    *****************************************************************************
    *
    * filename:       logger.py
    * version:        1.0
    * author:         VPM
    * creation date:  11-MAR-2026
    *
    * change history:
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * VIVEK           11-MAR-2026    1.0      initial creation
    *
    *
    * description:    Class to generate logs
    *
    ****************************************************************************
    """

    def __init__(self,run_id,log_module,log_file_name):
        self.logger = logging.getLogger(str(log_module)+'_' + str(run_id))
        self.logger.setLevel(logging.DEBUG)
        if log_file_name=='training':
            file_handler = logging.FileHandler('logs/training_logs/train_log_' + str(run_id) + '.log')
        else:
            file_handler = logging.FileHandler('logs/prediction_logs/predict_log_' + str(run_id) + '.log')

        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self,message):
        self.logger.info(message)
        # Format and push to queue for live streaming
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        log_str = f"{timestamp} : INFO : {message}"
        Logger.log_queue.put(log_str)

    def exception(self,message):
        self.logger.exception(message)
        # Format and push to queue for live streaming
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        log_str = f"{timestamp} : ERROR : {message}"
        Logger.log_queue.put(log_str)
