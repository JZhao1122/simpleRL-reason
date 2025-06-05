# injector.py

import os
import shutil
import subprocess
import logging
import datetime
import atexit
import tempfile
from typing import List, Tuple, Dict, Union # Added Union
import argparse
import sys

# --- ANSI Color Codes ---
class ANSIColors:
    RESET = "\033[0m"
    DEBUG = "\033[94m"
    INFO = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    CRITICAL = "\033[91m\033[1m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

# --- Custom Color Formatter ---
class ColorFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: ANSIColors.DEBUG + "%(asctime)s - %(levelname)s - %(message)s" + ANSIColors.RESET,
        logging.INFO: ANSIColors.INFO + "%(asctime)s - %(levelname)s - %(message)s" + ANSIColors.RESET,
        logging.WARNING: ANSIColors.WARNING + "%(asctime)s - %(levelname)s - %(message)s" + ANSIColors.RESET,
        logging.ERROR: ANSIColors.ERROR + "%(asctime)s - %(levelname)s - %(message)s" + ANSIColors.RESET,
        logging.CRITICAL: ANSIColors.CRITICAL + "%(asctime)s - %(levelname)s - %(message)s" + ANSIColors.RESET,
        "DEFAULT_NO_COLOR": "%(asctime)s - %(levelname)s - %(message)s" # For file logs
    }

    def __init__(self, use_color=True, fmt=None, datefmt=None, style='%', validate=True): # Added default style='%'
        # Ensure the initial format string is set based on use_color for the parent
        initial_fmt = fmt
        if initial_fmt is None: # if no specific format is passed to __init__
            if use_color:
                # Pick a default colored format, e.g., INFO, or just a generic colored one
                # This is mainly for the self._style object to be initialized correctly if super() uses it.
                # However, we override format() so it's less critical what initial_fmt is here.
                initial_fmt = self.FORMATS[logging.INFO]
            else:
                initial_fmt = self.FORMATS["DEFAULT_NO_COLOR"]
        super().__init__(initial_fmt, datefmt, style, validate=validate) # Pass style char
        self.use_color = use_color
        # Store the original datefmt to reuse
        self._orig_datefmt = datefmt # logging.Formatter stores it as self.datefmt

    def format(self, record):
        if self.use_color:
            log_fmt_str = self.FORMATS.get(record.levelno, self.FORMATS["DEFAULT_NO_COLOR"])
        else:
            log_fmt_str = self.FORMATS["DEFAULT_NO_COLOR"]
        
        # For Python 3.10+, logging.Formatter constructor's `style` parameter is correctly handled.
        # The `_style` attribute of the formatter instance holds the style object.
        # When creating a new Formatter, we need to pass the style *character*.
        # The parent class's `self.default_msec_format` might also be relevant if not using datefmt.
        
        # Simplest way: assume '%' style for all our formats.
        # If we wanted to support different styles dynamically, this would be more complex.
        current_style_char = '%' # Since all our FORMATS use %-style

        formatter = logging.Formatter(log_fmt_str, datefmt=self.datefmt, style=current_style_char)
        return formatter.format(record)

# Global logger setup
console_color_formatter = ColorFormatter(use_color=True)
file_log_formatter = ColorFormatter(use_color=False) # Or use a standard logging.Formatter for files

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_color_formatter)

logger = logging.getLogger("FileInjector")
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

# Globals for atexit cleanup
_originals_to_restore: Dict[str, str] = {}
_replaced_files: List[str] = []

def _cleanup_files():
    """Ensures files are restored on script exit."""
    if not _originals_to_restore and not _replaced_files:
        return

    logger.info(f"{ANSIColors.CYAN}--- Initiating cleanup via atexit hook ---{ANSIColors.RESET}")
    restored_during_cleanup = False
    for target_path, backup_path in list(_originals_to_restore.items()):
        if os.path.exists(backup_path):
            try:
                if target_path in _replaced_files and os.path.exists(target_path):
                    try:
                        os.remove(target_path)
                        logger.info(f"ATEIXT: Removed replaced file before restore: {target_path}")
                    except OSError as e:
                        logger.error(f"ATEIXT: Error removing replaced file '{target_path}' before restore: {e}. Will attempt move anyway.")
                shutil.move(backup_path, target_path)
                logger.info(f"ATEIXT: Successfully restored '{target_path}' from backup '{backup_path}'.")
                _originals_to_restore.pop(target_path, None)
                if target_path in _replaced_files:
                    _replaced_files.remove(target_path)
                restored_during_cleanup = True
            except Exception as e:
                logger.error(f"ATEIXT: Error restoring '{target_path}' from '{backup_path}': {e}")
        else:
            logger.warning(f"ATEIXT: Backup file '{backup_path}' for '{target_path}' not found. Cannot restore.")
            _originals_to_restore.pop(target_path, None)

    for target_path in list(_replaced_files):
        if target_path not in _originals_to_restore:
            logger.warning(f"ATEIXT: File '{target_path}' was marked as replaced but no backup was scheduled for restore. Manual check might be needed.")
            _replaced_files.remove(target_path)

    if restored_during_cleanup:
        logger.info(f"{ANSIColors.CYAN}--- Atexit cleanup completed ---{ANSIColors.RESET}")
    elif _originals_to_restore or _replaced_files:
        logger.warning(f"{ANSIColors.CYAN}--- Atexit cleanup finished, but some files might still need attention. Check logs. ---{ANSIColors.RESET}")
        logger.warning(f"Files pending restoration: {_originals_to_restore}")
        logger.warning(f"Files marked as replaced: {_replaced_files}")

atexit.register(_cleanup_files)

class FileInjector:
    def __init__(self, original_files: List[str], replacement_files: List[str], working_directory: str = None, log_dir_base: str = None):
        if len(original_files) != len(replacement_files):
            raise ValueError("The number of original files must match the number of replacement files.")
        if not original_files:
            logger.warning("No files specified for replacement.")

        self.original_files = [os.path.abspath(p) for p in original_files]
        self.replacement_files = [os.path.abspath(p) for p in replacement_files]
        self.working_directory = os.path.abspath(working_directory) if working_directory else os.getcwd()
        self.log_dir_base = os.path.abspath(log_dir_base) if log_dir_base else os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

        if not os.path.exists(self.log_dir_base):
            os.makedirs(self.log_dir_base, exist_ok=True)
        
        has_file_handler = any(isinstance(h, logging.FileHandler) and h.formatter == file_log_formatter for h in logger.handlers)
        if not has_file_handler:
            log_file_path = os.path.join(self.log_dir_base, f"injection_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(file_log_formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file_path}") # This will be colored in console, plain in file
        else:
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler) and handler.formatter == file_log_formatter:
                    logger.info(f"Continuing to log to existing file: {handler.baseFilename}")
                    break

        self.backup_dir = tempfile.mkdtemp(prefix="file_injector_backup_")
        logger.info(f"Created temporary backup directory: {self.backup_dir}")
        self.backup_paths: Dict[str, str] = {}

    def _inject_files(self) -> bool:
        global _originals_to_restore, _replaced_files
        logger.info(f"{ANSIColors.CYAN}--- Starting file injection process ---{ANSIColors.RESET}")
        all_successful = True
        for i, original_path in enumerate(self.original_files):
            replacement_path = self.replacement_files[i]

            if not os.path.exists(original_path):
                logger.error(f"Original file '{original_path}' does not exist. Skipping replacement.")
                all_successful = False
                continue
            if not os.path.exists(replacement_path):
                logger.error(f"Replacement file '{replacement_path}' does not exist. Skipping replacement for '{original_path}'.")
                all_successful = False
                continue

            try:
                backup_filename = os.path.basename(original_path) + f".backup_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                backup_path_in_tempdir = os.path.join(self.backup_dir, backup_filename)
                shutil.copy2(original_path, backup_path_in_tempdir)
                self.backup_paths[original_path] = backup_path_in_tempdir
                _originals_to_restore[original_path] = backup_path_in_tempdir
                logger.info(f"Backed up '{original_path}' to '{backup_path_in_tempdir}'.")

                os.remove(original_path)
                shutil.copy2(replacement_path, original_path)
                _replaced_files.append(original_path)
                logger.info(f"Replaced '{original_path}' with '{replacement_path}'.")

            except Exception as e:
                logger.error(f"Error during injection for '{original_path}': {e}")
                if original_path in self.backup_paths:
                    try:
                        shutil.move(self.backup_paths[original_path], original_path)
                        logger.info(f"Rolled back replacement for '{original_path}' due to error.")
                        _originals_to_restore.pop(original_path, None)
                        if original_path in _replaced_files: _replaced_files.remove(original_path)
                    except Exception as rb_e:
                        logger.critical(f"CRITICAL: Failed to roll back '{original_path}' after injection error: {rb_e}. Backup is at {self.backup_paths[original_path]}")
                all_successful = False
                break
        if all_successful:
            logger.info(f"{ANSIColors.CYAN}--- File injection process completed successfully ---{ANSIColors.RESET}")
        else:
            logger.error(f"{ANSIColors.CYAN}--- File injection process encountered errors ---{ANSIColors.RESET}")
        return all_successful

    def _restore_files(self) -> bool:
        global _originals_to_restore, _replaced_files
        logger.info(f"{ANSIColors.CYAN}--- Starting file restoration process ---{ANSIColors.RESET}")
        all_successful = True
        for original_path in reversed(list(self.backup_paths.keys())):
            if original_path in self.backup_paths:
                backup_path = self.backup_paths[original_path]
                try:
                    if os.path.exists(original_path):
                        is_replaced_by_us = original_path in _replaced_files
                        if is_replaced_by_us:
                            os.remove(original_path)
                            logger.info(f"Removed replaced file before restore: {original_path}")
                        else:
                            logger.warning(f"File '{original_path}' exists but was not marked as replaced by this injector. Will attempt to overwrite with backup.")
                    shutil.move(backup_path, original_path)
                    logger.info(f"Successfully restored '{original_path}' from backup '{backup_path}'.")
                    _originals_to_restore.pop(original_path, None)
                    if original_path in _replaced_files:
                        _replaced_files.remove(original_path)
                except Exception as e:
                    logger.error(f"Error restoring '{original_path}' from '{backup_path}': {e}. Backup remains at '{backup_path}'.")
                    all_successful = False
        
        if all_successful:
            logger.info(f"{ANSIColors.CYAN}--- File restoration process completed successfully ---{ANSIColors.RESET}")
        else:
            logger.error(f"{ANSIColors.CYAN}--- File restoration process encountered errors. Check backup directory and logs. ---{ANSIColors.RESET}")

        try:
            shutil.rmtree(self.backup_dir)
            logger.info(f"Removed temporary backup directory: {self.backup_dir}")
        except Exception as e:
            logger.warning(f"Could not remove temporary backup directory '{self.backup_dir}': {e}")
        return all_successful

    def run_command_with_injection(self, command: Union[str, List[str]], shell: bool = True, **kwargs) -> int:
        command_str_for_log = command if isinstance(command, str) else ' '.join(command)
        logger.info(f"{ANSIColors.CYAN}--- Target command to execute ---{ANSIColors.RESET}\n{command_str_for_log}\n{ANSIColors.CYAN}---------------------------------{ANSIColors.RESET}")
        logger.info(f"Working directory for command: {self.working_directory}")

        if not self._inject_files():
            logger.error("Aborting command execution due to file injection failure.")
            self._restore_files()
            return -1

        return_code = -1
        try:
            logger.info("Executing target command (output will stream directly)...")
            process = subprocess.run(
                command,
                shell=shell,
                cwd=self.working_directory,
                text=True,
                check=False,
                # No capture_output=True, so output streams
                **kwargs
            )
            return_code = process.returncode
            if return_code == 0:
                logger.info(f"Command executed successfully. Return code: {return_code}")
            else:
                logger.warning(f"Command finished with errors. Return code: {return_code}")
        except FileNotFoundError:
            msg = f"ERROR: Command or one of its components not found. Ensure the command and PATH are correct."
            logger.critical(msg) # Logger will color this
            print(f"{ANSIColors.ERROR}{msg}{ANSIColors.RESET}", file=sys.stderr) # Also print raw for clarity
            return_code = -127
        except Exception as e:
            msg = f"An unexpected error occurred while executing the command: {e}"
            logger.critical(msg) # Logger will color this
            print(f"{ANSIColors.ERROR}{msg}{ANSIColors.RESET}", file=sys.stderr) # Also print raw for clarity
            return_code = -1
        finally:
            logger.info(f"{ANSIColors.CYAN}--- Attempting to restore original files after command execution ---{ANSIColors.RESET}")
            if not self._restore_files():
                logger.critical("CRITICAL: File restoration failed after command execution. Manual check needed!")
            else:
                logger.info("Original files restored successfully.")
        return return_code

def main():
    parser = argparse.ArgumentParser(
        description="Injects files, runs a command, and restores original files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-o", "--original-files", nargs='+', required=True,
                        help="List of absolute paths to original files to be replaced.")
    parser.add_argument("-r", "--replacement-files", nargs='+', required=True,
                        help="List of absolute paths to replacement files. Must match order of original-files.")
    parser.add_argument("-c", "--command", required=True,
                        help="The command string to execute after file injection. \n"
                             "Example: \"python my_script.py --arg value\"\n"
                             "For multiline, ensure your shell handles it or quote appropriately:\n"
                             "  'echo \"line1\"; echo \"line2\"'")
    parser.add_argument("-w", "--working-directory", default=None,
                        help="Working directory for the command. Defaults to current directory.")
    parser.add_argument("-l", "--log-dir", default=None,
                        help="Base directory for log files. Defaults to 'logs' subdirectory next to this script.")
    parser.add_argument("--no-shell", action="store_true",
                        help="Execute command without shell. The --command argument will be split by spaces. "
                             "This is a simplified approach; for complex non-shell commands, use the Python API.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (DEBUG level) logging to console.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers: # Ensure all handlers get the level
            handler.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    command_to_run = args.command
    use_shell = not args.no_shell
    if not use_shell:
        command_to_run = args.command.split()
        logger.info(f"Executing command without shell, split into: {command_to_run}")

    try:
        injector = FileInjector(
            original_files=args.original_files,
            replacement_files=args.replacement_files,
            working_directory=args.working_directory,
            log_dir_base=args.log_dir
        )
        ret_code = injector.run_command_with_injection(
            command=command_to_run,
            shell=use_shell
        )
        
        # Since output is streamed, we don't print stdout/stderr here from captured variables
        logger.info(f"Injector script finished. Target command exited with code: {ret_code}")
        sys.exit(ret_code)

    except ValueError as ve:
        logger.error(f"Configuration Error: {ve}")
        sys.exit(2)
    except Exception as e:
        logger.critical(f"An unhandled exception occurred in the injector script: {e}", exc_info=True)
        sys.exit(3)

if __name__ == "__main__":
    main()