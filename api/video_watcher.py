import os
import asyncio
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VideoHandler(FileSystemEventHandler):
    """Handles file system events for video files."""

    def __init__(self, sse_clients: List[asyncio.Queue]):
        """
        Initializes the handler with a list of SSE client queues.

        Args:
            sse_clients: A list of asyncio.Queue instances for notifying clients.
        """
        self.sse_clients = sse_clients
        logging.info(f"VideoHandler initialized with {len(sse_clients)} client queues initially.")

    def on_created(self, event):
        """
        Called when a file or directory is created.

        Args:
            event: The event object representing the file system event.
        """
        if event.is_directory:
            logging.debug(f"Ignoring directory creation: {event.src_path}")
            return

        if event.src_path.lower().endswith(".mp4"):
            filename = os.path.basename(event.src_path)
            logging.info(f"New MP4 file detected: {filename}")
            # Notify all connected SSE clients
            # Use a copy of the list to avoid issues if a client disconnects during iteration
            clients_to_notify = list(self.sse_clients)
            logging.info(f"Notifying {len(clients_to_notify)} clients about new file: {filename}")
            for queue in clients_to_notify:
                try:
                    # Use put_nowait as this handler runs in a separate thread
                    # managed by watchdog, not in the main asyncio event loop.
                    queue.put_nowait(filename)
                    logging.debug(f"Added '{filename}' to a client queue.")
                except asyncio.QueueFull:
                    logging.warning(f"Client queue is full. Could not add '{filename}'.")
                except Exception as e:
                    logging.error(f"Error adding '{filename}' to client queue: {e}")
        else:
            logging.debug(f"Ignoring non-MP4 file creation: {event.src_path}")


def start_watcher(path: str, sse_clients: List[asyncio.Queue]) -> Observer:
    """
    Starts the file system watcher.

    Args:
        path: The directory path to watch.
        sse_clients: The list of SSE client queues to notify.

    Returns:
        The Observer instance that was started.
    """
    if not os.path.isdir(path):
        logging.error(f"Watch directory does not exist or is not a directory: {path}")
        # Consider raising an exception or returning None
        raise ValueError(f"Invalid watch directory: {path}")

    event_handler = VideoHandler(sse_clients)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    logging.info(f"Started watching directory: {path}")
    return observer


def stop_watcher(observer: Observer):
    """
    Stops the file system watcher.

    Args:
        observer: The Observer instance to stop.
    """
    if observer and observer.is_alive():
        observer.stop()
        observer.join()  # Wait for the thread to finish
        logging.info("Stopped file system watcher.")
    else:
        logging.info("File system watcher was not running or already stopped.")