import os
import sys
import logging
import datetime
from typing import Union
from src.utils.variables import DEFAULT_LOG_FOLDER
from src.utils.configs import LogConfigs
        

def get_logger(name: str, log_folder: str, mode: str = 'smart'):
    if mode == 'smart':
        log_folder = os.path.join(log_folder, name)
        my_logger = SmartLogger(name, verbose=False, log_dir=log_folder)

        def handle_exception(exc_type, exc_value, exc_traceback):
            my_logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        sys.excepthook = handle_exception

        return my_logger
    elif mode == 'dumb':
        return DumbLogger(name)
    else:
        raise ValueError('Mode "{}" not supported!'.format(mode))


def get_logger_from_configs(configs: LogConfigs):
    """Create and return a logger instance from a `LogConfigs` object.

    Parameters
    ----------
    configs : LogConfigs
        Configuration dataclass containing `name`, `log_folder` and `log_mode`.

    Returns
    -------
    SmartLogger | DumbLogger
        Logger instance configured according to `configs`.
    """
    return get_logger(name=configs.name,
                    log_folder=configs.log_folder,
                    mode=configs.log_mode)


class DumbLogger:
    def __init__(self, name: str):
        """A minimal console-only logger used as default when no file logging is required.

        The `DumbLogger` implements a small subset of the `SmartLogger` API that
        is sufficient for libraries and simple scripts (methods like
        `print_it`, `print_error_to_console`, `error`, `critical`, and
        `set_logger_newline`). It is intentionally lightweight and does not
        create log files.
        """
        self.name = name

    @staticmethod
    def print_it(msg, *args, console_only=False, file_only=False, **kwargs):
        """Logging method for the MIA level. The `msg` gets logged both to stdout and to file
        (if a file handler is present), irrespective of verbosity settings.
        If console_only=True, prints only to console, never to log file.
        If file_only=True, logs only to file, never to console."""
        if console_only and file_only:
            raise ValueError("Cannot set both console_only and file_only")
        if console_only:
            return print(msg, *args, **kwargs)
        else:
            # For DumbLogger, file_only doesn't make sense since there's no file logging
            return print(msg, *args, **kwargs)

    @staticmethod
    def print_it_same_line(msg, *args, console_only=False, file_only=False, **kwargs):
        """Logging method for the MIA level. The `msg` gets logged both to stdout and to file
        (if a file handler is present), irrespective of verbosity settings.
        If console_only=True, prints only to console on same line, never to log file.
        If file_only=True, logs only to file on same line, never to console."""
        if console_only and file_only:
            raise ValueError("Cannot set both console_only and file_only")
        if console_only:
            return print(msg, end='\r', *args, **kwargs)
        else:
            # For DumbLogger, file_only doesn't make sense since there's no file logging
            return print(msg, end='\r', *args, **kwargs)

    def set_logger_newline(self, console_only: bool = False):
        print()

    @staticmethod
    def print_error_to_console(msg, *args, **kwargs):
        """Print error messages to stderr."""
        return print(msg, *args, file=sys.stderr, **kwargs)

    @staticmethod
    def error(msg, *args, **kwargs):
        """Print error messages to stderr (console only for DumbLogger)."""
        return print(msg, *args, file=sys.stderr, **kwargs)

    @staticmethod
    def critical(msg, *args, **kwargs):
        """Print critical error messages to stderr (console only for DumbLogger)."""
        return print(msg, *args, file=sys.stderr, **kwargs)
    
    def get_folder(self):
        return None
    
    def get_mode(self):
        return 'dumb'

class SmartLogger(logging.getLoggerClass()):
    def __init__(self, name, verbose, log_dir='logs'):
        """Create a custom logger with the specified `name`. When `log_dir` is None, a simple
        console logger is created. Otherwise, a file logger is created in addition to the console
        logger.

        This custom logger class adds an extra logging level FRAMEWORK (at INFO priority), with the
        aim of logging messages irrespective of any verbosity settings.

        By default, the five standard logging levels (DEBUG through CRITICAL) only display
        information in the log file if a file handler is added to the logger, but **not** to the
        console.

        :param name: name for the logger
        :param verbose: bool: whether the logging should be verbose; if True, then all messages get
            logged both to stdout and to the log file (if `log_dir` is specified); if False, then
            messages only get logged to the log file (if `log_dir` is specified), with the exception
            of FRAMEWORK level messages which get logged either way
        :param log_dir: str: (optional) the directory for the log file; if not present, no log file
            is created
        """
        # Create custom logger logging all five levels
        super().__init__(name)
        self.setLevel(logging.DEBUG)

        # Add new logging level
        logging.addLevelName(logging.INFO, 'MIA {}'.format(self.name))

        # Determine verbosity settings
        self.verbose = verbose

        # Create stream handler for logging to stdout (log all five levels)
        self.stdout_handler = logging.StreamHandler(sys.stdout)
        self.stdout_handler.setLevel(logging.INFO)
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)9s | %(message)s'))
        self.enable_console_output()

        self.file_handler = None
        if log_dir:
            self.log_dir = log_dir
            self.add_file_handler()

        self.name = name
    
    def get_name(self):
        return self.name
    
    def get_folder(self):
        return self.log_dir
    
    def get_mode(self):
        return 'smart'
    
    def get_verbosity(self):
        return self.verbose

    def add_file_handler(self):
        """Add a file handler for this logger with the specified `name` (and store the log file
        under `log_dir`)."""
        # Format for file log
        fmt = '%(asctime)s | %(levelname)9s | %(message)s'
        formatter = logging.Formatter(fmt)

        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir)
            except:
                print(f"{self.__class__.__name__}: Cannot create directory {self.log_dir}. ",
                        end='', file=sys.stderr)
                self.log_dir = '/tmp' if sys.platform.startswith('linux') else '.'
                print(f"Defaulting to {self.log_dir}.", file=sys.stderr)
        log_file = self.get_log_file()

        # Create file handler for logging to a file (log all five levels)
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(formatter)
        self.addHandler(self.file_handler)

    def get_log_dir(self):
        return self.log_dir
    
    def get_log_file(self):
        return "{}/{}.log".format(self.log_dir, self.name)

    def has_console_handler(self):
        return len([h for h in self.handlers if type(h) == logging.StreamHandler]) > 0

    def has_file_handler(self):
        return len([h for h in self.handlers if isinstance(h, logging.FileHandler)]) > 0

    def disable_console_output(self):
        if not self.has_console_handler():
            return
        self.removeHandler(self.stdout_handler)

    def enable_console_output(self):
        if self.has_console_handler():
            return
        self.addHandler(self.stdout_handler)

    def disable_file_output(self):
        if not self.has_file_handler():
            return
        self.removeHandler(self.file_handler)

    def enable_file_output(self):
        if self.has_file_handler():
            return
        self.addHandler(self.file_handler)

    def set_logger_inline(self):
        self.stdout_handler.terminator = '\r'

    def set_logger_newline(self, console_only: bool = False):
        self.print_it('', console_only=console_only)
        self.stdout_handler.terminator = '\n'

    def print_it_log_file_only(self, msg, *args, **kwargs):
        """Log only to file, never to console. Deprecated: use print_it(file_only=True) instead."""
        previous_verbosity = self.verbose
        self.verbose = False
        self._custom_log(super().info, msg, *args, **kwargs)
        self.verbose = previous_verbosity

    def print_it(self, msg, *args, console_only=False, file_only=False, **kwargs):
        """Logging method for the MIA level. The `msg` gets logged both to stdout and to file
        (if a file handler is present), irrespective of verbosity settings.
        If console_only=True, prints only to console, never to log file.
        If file_only=True, logs only to file, never to console."""
        if console_only and file_only:
            raise ValueError("Cannot set both console_only and file_only")
        elif console_only:
            # Temporarily disable file output if it exists
            had_file = self.has_file_handler()
            if had_file:
                self.disable_file_output()
            # Print to console only, format the message like a normal INFO log message but print only to console
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            level_name = 'MIA {}'.format(self.name)
            formatted_msg = f'{timestamp} | {level_name:9s} | {msg}'
            print(formatted_msg, *args, **kwargs)
            # Re-enable file output
            if had_file:
                self.enable_file_output()
        elif file_only:
            # Temporarily set verbose to False to ensure file-only logging
            previous_verbosity = self.verbose
            self.verbose = False
            result = self._custom_log(super().info, msg, *args, **kwargs)
            self.verbose = previous_verbosity
            return result
        else:
            return super().info(msg, *args, **kwargs)

    def print_it_same_line(self, msg, *args, console_only=False, file_only=False, **kwargs):
        """Logging method for the MIA level. The `msg` gets logged both to stdout and to file
        (if a file handler is present), irrespective of verbosity settings.
        If console_only=True, prints only to console on same line, never to log file.
        If file_only=True, logs only to file, never to console."""
        if console_only and file_only:
            raise ValueError("Cannot set both console_only and file_only")
        elif console_only:
            # Temporarily disable file output if it exists
            had_file = self.has_file_handler()
            if had_file:
                self.disable_file_output()
            # Print to console only
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            level_name = 'MIA {}'.format(self.name)
            formatted_msg = f'{timestamp} | {level_name:9s} | {msg}'
            print(formatted_msg, end='\r', *args, **kwargs)
            # Re-enable file output
            if had_file:
                self.enable_file_output()
        elif file_only:
            # Temporarily set verbose to False to ensure file-only logging
            previous_verbosity = self.verbose
            self.verbose = False
            result = self._custom_log(super().info, msg, *args, **kwargs)
            self.verbose = previous_verbosity
            return result
        else:
            # For file logging, we need to log normally since same line doesn't make sense in files
            return super().info(msg, *args, **kwargs)

    def print_error_to_console(self, msg, *args, **kwargs):
        """Print error messages to stderr (console) and also to log file.
        Note: This method is equivalent to logger.error() for SmartLogger."""
        # Print to stderr for immediate visibility
        print(msg, *args, file=sys.stderr, **kwargs)
        # Also log to file using standard logging mechanism
        if self.has_file_handler():
            # Temporarily set console handler level to prevent duplicate output
            if self.has_console_handler():
                old_level = self.stdout_handler.level
                self.stdout_handler.setLevel(logging.CRITICAL + 1)
            # Call parent's error method which will log to file
            super().error(msg, *args, **kwargs)
            # Restore console handler level
            if self.has_console_handler():
                self.stdout_handler.setLevel(old_level)

    def _custom_log(self, func, msg, *args, **kwargs):
        """Helper method for logging DEBUG through CRITICAL messages by calling the appropriate
        `func()` from the base class."""
        # Log normally if verbosity is on
        if self.verbose:
            return func(msg, *args, **kwargs)

        # If verbosity is off and there is no file handler, there is nothing left to do
        if not self.has_file_handler():
            return

        # If verbosity is off and a file handler is present, then disable stdout logging, log, and
        # finally reenable stdout logging
        self.disable_console_output()
        func(msg, *args, **kwargs)
        self.enable_console_output()

    def debug(self, msg, *args, **kwargs):
        self._custom_log(super().debug, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._custom_log(super().info, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._custom_log(super().warning, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Print error messages to stderr (console) and also log to file."""
        # Log to file using standard logging mechanism
        if self.has_file_handler():
            # Temporarily set console handler level to prevent duplicate output
            if self.has_console_handler():
                old_level = self.stdout_handler.level
                self.stdout_handler.setLevel(logging.CRITICAL + 1)
            # Call parent's error method which will log to file
            super().error(msg, *args, **kwargs)
            # Restore console handler level
            if self.has_console_handler():
                self.stdout_handler.setLevel(old_level)
        # Also print to stderr for immediate visibility
        print(msg, *args, file=sys.stderr, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Print critical error messages to stderr (console) and also log to file."""
        # Log to file using standard logging mechanism
        if self.has_file_handler():
            # Temporarily set console handler level to prevent duplicate output
            if self.has_console_handler():
                old_level = self.stdout_handler.level
                self.stdout_handler.setLevel(logging.CRITICAL + 1)
            # Call parent's critical method which will log to file
            super().critical(msg, *args, **kwargs)
            # Restore console handler level
            if self.has_console_handler():
                self.stdout_handler.setLevel(old_level)
        # Also print to stderr for immediate visibility
        print(msg, *args, file=sys.stderr, **kwargs)


class Loggable():
    def __init__(self, logger: Union[DumbLogger, SmartLogger] = None):
        if logger is None:
            self.logger = get_logger(name='log', log_folder=DEFAULT_LOG_FOLDER, mode='dumb')
        else:
            self.logger = logger

    def reset_logger(self):
        if self.logger.get_mode() == 'dumb':
            self.logger = get_logger(name='log', log_folder=DEFAULT_LOG_FOLDER, mode='dumb')
        elif self.logger.get_mode() == 'smart':
            self.logger = SmartLogger(name=self.logger.get_name(),
                                    verbose=self.logger.get_verbosity(),
                                    log_dir=self.logger.get_log_dir())
    