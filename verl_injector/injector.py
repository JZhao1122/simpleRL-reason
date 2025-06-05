# injector.py

import os
import shutil
import subprocess
import logging
import datetime
import atexit
import tempfile
from typing import List, Tuple, Dict
import argparse

# Global logger setup (console first, file handler added in main)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger = logging.getLogger("FileInjector")
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

# Globals for atexit cleanup
_originals_to_restore: Dict[str, str] = {}  # {target_path: backup_path}
_replaced_files: List[str] = []  # [target_path, ...]

def _cleanup_files():
    """Ensures files are restored on script exit."""
    if not _originals_to_restore and not _replaced_files:
        return

    logger.info("--- Initiating cleanup via atexit hook ---")
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
        logger.info("--- Atexit cleanup completed ---")
    elif _originals_to_restore or _replaced_files:
        logger.warning("--- Atexit cleanup finished, but some files might still need attention. Check logs. ---")
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
        self.log_dir_base = os.path.abspath(log_dir_base) if log_dir_base else os.path.join(os.path.dirname(__file__), "logs")

        if not os.path.exists(self.log_dir_base):
            os.makedirs(self.log_dir_base, exist_ok=True)
        
        # Setup file logging now that we have the log_dir_base
        log_file_path = os.path.join(self.log_dir_base, f"injection_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file_path}")


        self.backup_dir = tempfile.mkdtemp(prefix="file_injector_backup_")
        logger.info(f"Created temporary backup directory: {self.backup_dir}")
        self.backup_paths: Dict[str, str] = {}

    def _inject_files(self) -> bool:
        global _originals_to_restore, _replaced_files
        logger.info("--- Starting file injection process ---")
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
            logger.info("--- File injection process completed successfully ---")
        else:
            logger.error("--- File injection process encountered errors ---")
        return all_successful

    def _restore_files(self) -> bool:
        global _originals_to_restore, _replaced_files
        logger.info("--- Starting file restoration process ---")
        all_successful = True
        for original_path in reversed(list(self.backup_paths.keys())): # Use keys from backup_paths for restoration
            if original_path in self.backup_paths:
                backup_path = self.backup_paths[original_path]
                try:
                    if os.path.exists(original_path):
                         # Check if it's one of our replaced files before removing
                        is_replaced_by_us = False
                        # A more robust check would be to compare content or inodes if necessary,
                        # but for this simple case, we assume if it exists, it's our replacement.
                        if original_path in _replaced_files:
                            is_replaced_by_us = True

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
            logger.info("--- File restoration process completed successfully ---")
        else:
            logger.error("--- File restoration process encountered errors. Check backup directory and logs. ---")

        try:
            shutil.rmtree(self.backup_dir)
            logger.info(f"Removed temporary backup directory: {self.backup_dir}")
        except Exception as e:
            logger.warning(f"Could not remove temporary backup directory '{self.backup_dir}': {e}")
        return all_successful

    def run_command_with_injection(self, command: str, shell: bool = True, **kwargs) -> Tuple[int, str, str]:
        logger.info(f"--- Target command to execute ---\n{command}\n---------------------------------")
        logger.info(f"Working directory for command: {self.working_directory}")

        if not self._inject_files():
            logger.error("Aborting command execution due to file injection failure.")
            self._restore_files()
            return -1, "", "File injection failed prior to command execution."

        process = None
        stdout_output = ""
        stderr_output = ""
        return_code = -1

        try:
            logger.info("Executing target command...")
            process = subprocess.run(
                command,
                shell=shell,
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                check=False,
                **kwargs
            )
            return_code = process.returncode
            stdout_output = process.stdout
            stderr_output = process.stderr
            if return_code == 0:
                logger.info(f"Command executed successfully. Return code: {return_code}")
            else:
                logger.warning(f"Command finished with errors. Return code: {return_code}")
            if stdout_output:
                logger.debug(f"Command STDOUT:\n{stdout_output}") # Changed to debug for brevity
            if stderr_output:
                logger.debug(f"Command STDERR:\n{stderr_output}") # Changed to debug for brevity
        except FileNotFoundError:
            msg = f"ERROR: Command or one of its components not found. Ensure the command and PATH are correct."
            logger.critical(msg)
            stderr_output = msg
            return_code = -127
        except Exception as e:
            msg = f"An unexpected error occurred while executing the command: {e}"
            logger.critical(msg)
            stderr_output += f"\n{msg}"
            return_code = -1
        finally:
            logger.info("--- Attempting to restore original files after command execution ---")
            if not self._restore_files():
                logger.critical("CRITICAL: File restoration failed after command execution. Manual check needed!")
            else:
                logger.info("Original files restored successfully.")
        return return_code, stdout_output, stderr_output

def main():
    parser = argparse.ArgumentParser(description="Injects files, runs a command, and restores original files.")
    parser.add_argument("-o", "--original-files", nargs='+', required=True,
                        help="List of absolute paths to original files to be replaced.")
    parser.add_argument("-r", "--replacement-files", nargs='+', required=True,
                        help="List of absolute paths to replacement files. Must match order of original-files.")
    parser.add_argument("-c", "--command", required=True,
                        help="The command string to execute after file injection. "
                             "If using shell features or multiline, ensure proper quoting for your shell "
                             "or pass it as a single argument string.")
    parser.add_argument("-w", "--working-directory", default=None,
                        help="Working directory for the command. Defaults to current directory.")
    parser.add_argument("-l", "--log-dir", default=None,
                        help="Base directory for log files. Defaults to 'logs' subdirectory next to this script.")
    parser.add_argument("--no-shell", action="store_true",
                        help="Execute command without shell (command must be a list then, not supported via this CLI directly for simplicity, use shell=False in Python API). This flag makes `shell=False` if set.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (DEBUG level) logging to console.")


    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG) # Set console logger to DEBUG
        # Also ensure any file loggers get this level if we want verbose file logs too
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")


    use_shell = not args.no_shell

    try:
        injector = FileInjector(
            original_files=args.original_files,
            replacement_files=args.replacement_files,
            working_directory=args.working_directory,
            log_dir_base=args.log_dir
        )
        ret_code, stdout, stderr = injector.run_command_with_injection(
            command=args.command,
            shell=use_shell
        )

        if stdout:
            print("--- STDOUT ---")
            print(stdout)
        if stderr:
            print("--- STDERR ---", file=sys.stderr) # Print stderr to actual stderr
            print(stderr, file=sys.stderr)

        sys.exit(ret_code)

    except ValueError as ve:
        logger.error(f"Configuration Error: {ve}")
        sys.exit(2)
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
        sys.exit(3)

if __name__ == "__main__":
    import sys # For sys.exit and printing to stderr in main
    main()