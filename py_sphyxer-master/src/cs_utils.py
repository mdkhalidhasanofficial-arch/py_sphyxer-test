
"""
utility functions to establish logging functionality
"""
import sys
import time
import yaml
import logging
from pathlib import Path


def setup_logger(sys_args: list, config: dict) -> logging.getLoggerClass():
    """
    initiate logger for main routine
    all hard coded options at the moment
    :return: logger object
    """

    log_format = '%(levelname)-10s | %(asctime)-25s | %(name)-40s | %(funcName)-20s | %(lineno)5d | %(message)s'

    log_filename = sys_args[0].strip('.py') + '.log'

    try:
        logging.basicConfig(
                            format=log_format,
                            level=logging.INFO,
                            filename=log_filename,
                            filemode='w')

        # ... add console log stream
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(log_format)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    except PermissionError as pe:
        print(' ***** PermissionError in instantiating logging file: ' + str(pe))

    logger = logging.getLogger()

    try:
        if 'logging_level' in config:
            log_level = logging.getLevelName(config['logging_level'])
            logger.setLevel(log_level)
    except ValueError as ve:
        print(' ***** ValueError in setting log level from config file : ' + str(ve))

    return logger


def log_machine(f):
    """
    decorator function to add logging capability
    usage : add @log_machine to functions for which logging is requested
    logs start time and completion time and time to execute each decorated function
    assumes : log file and logger previously initiated by setup_logger()
    :param f: wrapped (decorated) function
    :return: result of wrapper function
    """

    def wrapper(*args, **kwargs):
        """
        executes the logging mechanism at entry and exit of decorated function
        :param args: args supplied to wrapped function
        :param kwargs: kwargs supplied to wrapped function
        :return: returned objs from wrapped function
        """

        # start the timer
        t = Timer()
        t.start()

        # start message to log file
        logger = logging.getLogger(f.__module__ + '.' + f.__name__)
        log_message = 'start ...'
        logger.info(log_message)

        # execute the called function
        returned_objs = f(*args, **kwargs)

        # stop the timer, log exit
        log_message = ('... complete | %.2f' % (t.stop()))
        logger.info(log_message)

        # return control to calling function
        return returned_objs

    return wrapper


def read_config(file_path):
    """
    :param file_path:
    :return: cfg, loaded yaml file
    """
    with open(file_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def setup_this_run(sys_args):
    """
    initiate config dict which provides run control parameter settings
    some default setting are defined
    config file name 'config.yaml' from main directory is read (if available) and overwerites defaults
    finally, if sys.argv had command line definition of input directory, that overwrites any previously stored string

    :param sys_args: command line sys.argv list
    :return: config: dict - k,v pairs of run config parameters
    """

    # some default config settings :

    config = {}

    # if available, overwrite defaults with config.yaml file settings
    try:
        config_yml = read_config("config.yaml")

        for ykey in config_yml:
            config[ykey] = config_yml[ykey]

    except FileNotFoundError as fnfe:
        print(' ***** config.yaml file not found - aborting execution')
        sys.exit(-1)

    # retain base capability to define input path as command line argument
    if len(sys_args) > 1:
        config['input'] = sys_args[1]

    return config


# ... from : https://realpython.com/python-timer/
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self, kill: bool=True) -> time:

        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time

        if kill:
            self._start_time = None

        return elapsed_time


def replace_text_line(f_path: Path, current: str, revised: str) -> None:
    """
    replace string (line) in text file with new line
    :param f_path: path to text(like) file
    :param current: string, existing line to identify
    :param revised: replacement string, line in updated file
    :return: None
    """
    with open(f_path, "r") as ftxt:
        newline = []
        for word in ftxt.readlines():
            newline.append(word.replace(current, revised))

    with open(f_path, "w") as ftxt:
        for line in newline:
            ftxt.writelines(line)
