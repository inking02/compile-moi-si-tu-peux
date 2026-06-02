"""
File: algo_utils.py

Description: This module contains utility functions for managing algorithm training iterations,
including a timer for tracking elapsed and estimated time, and a function for printing
iteration details.
"""

import threading
import time

# Global variables for timing and locking
_start_time = None
_lock = threading.Lock()  # Ensures thread-safe access to the timer

# Note: While technically we may not need locks for single-threaded access,
# they are included to ensure that only one model can access the timer at a time,
# preventing potential race conditions in multi-threaded scenarios.

def print_iter(iteration, epochs, cost, accuracy) -> None:
    """
    Prints the details of the current iteration, including elapsed time and estimated time left.

    Args:
        iteration (int): The current iteration number.
        epochs (int): The total number of epochs.
        cost (float or None): The current cost value, or None for the first iteration.
        accuracy (float or None): The current accuracy value, or None for the first iteration.

    Raises:
        RuntimeError: If the timer has not been started or if it's already running.
    """
    if iteration == 0:
        _start_timer()

    with _lock:
        if _start_time is None:
            raise RuntimeError("Timer has not been started. Please start the timer first.")

        elapsed_time = time.time() - _start_time

        if iteration > 0:
            estimated_time_left = (elapsed_time / iteration) * (epochs - iteration)
            estimated_time_left_str = f"{estimated_time_left:0.2f}s"
        else:
            estimated_time_left_str = "unknown"

        # Prepare iteration and epochs strings for formatting
        epochs_str = str(epochs)
        iteration_str = str(iteration).rjust(len(epochs_str))

        # Prepare the output line based on whether cost and accuracy are provided
        if cost is None or accuracy is None:  # For the first iteration and for QSVM
            output = (f"Iter: {iteration_str}/{epochs} | Elapsed Time: {elapsed_time:0.2f}s | "
                      f"Estimated Time Left: {estimated_time_left_str}")
        else:
            output = (f"Iter: {iteration_str}/{epochs} | Cost: {cost:0.5f} | Accuracy: {accuracy:0.5f} | "
                      f"Elapsed Time: {elapsed_time:0.2f}s | Estimated Time Left: {estimated_time_left_str}")

        # Clear the line and print the new output
        print(f"\r{output}{' ' * (len(output) + 50)}", end="")  # Adding extra spaces to clear any leftovers

    if iteration == epochs:
        print()
        _stop_timer()

def _start_timer() -> None:
    """
    Starts the timer for tracking elapsed time.

    Raises:
        RuntimeError: If the timer is already running.
    """
    global _start_time
    with _lock:
        if _start_time is not None:
            raise RuntimeError("Timer is already running. Please stop it before starting again.")
        _start_time = time.time()

def _stop_timer() -> None:
    """
    Stops the timer for tracking elapsed time.

    Raises:
        RuntimeError: If the timer is not running.
    """
    global _start_time
    with _lock:
        if _start_time is None:
            raise RuntimeError("Timer is not running. Please start the timer first.")
        _start_time = None
