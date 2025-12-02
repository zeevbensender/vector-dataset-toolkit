"""Background worker infrastructure for non-blocking operations.

This module provides a Worker abstraction that runs tasks in a separate thread
with progress updates, completion signals, error signals, and cancellation support.
"""

from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class WorkerSignals(QObject):
    """Signals emitted by worker threads.
    
    Signals:
        started: Emitted when the worker starts.
        progress: Emitted with (current, total) progress values.
        result: Emitted with the result when work completes successfully.
        error: Emitted with (exception, traceback_str) when an error occurs.
        finished: Emitted when the worker finishes (success or error).
    """
    started = Signal()
    progress = Signal(int, int)  # current, total
    result = Signal(object)
    error = Signal(Exception, str)  # exception, traceback string
    finished = Signal()


class Worker(QRunnable):
    """A worker that runs a function in a background thread.
    
    Example:
        def long_task(progress_callback):
            for i in range(100):
                progress_callback(i, 100)
                time.sleep(0.1)
            return "Done!"
            
        worker = Worker(long_task)
        worker.signals.progress.connect(update_progress_bar)
        worker.signals.result.connect(handle_result)
        worker.signals.error.connect(handle_error)
        
        QThreadPool.globalInstance().start(worker)
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the worker.
        
        Args:
            fn: The function to run. If it accepts a 'progress_callback' keyword
                argument, it will receive a callback for progress updates.
            *args: Positional arguments to pass to fn.
            **kwargs: Keyword arguments to pass to fn.
        """
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the worker."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    @Slot()
    def run(self) -> None:
        """Execute the worker function."""
        import traceback

        self.signals.started.emit()

        try:
            # Check if the function accepts a progress_callback
            import inspect
            sig = inspect.signature(self.fn)
            if "progress_callback" in sig.parameters:
                self.kwargs["progress_callback"] = self._progress_callback

            result = self.fn(*self.args, **self.kwargs)
            
            if not self._cancelled:
                self.signals.result.emit(result)
        except Exception as e:
            if not self._cancelled:
                tb = traceback.format_exc()
                self.signals.error.emit(e, tb)
        finally:
            self.signals.finished.emit()

    def _progress_callback(self, current: int, total: int) -> None:
        """Internal progress callback that emits the progress signal."""
        if not self._cancelled:
            self.signals.progress.emit(current, total)


class WorkerManager:
    """Manager for background workers.
    
    Provides a central place to manage worker threads and their lifecycle.
    """

    def __init__(self, max_threads: int | None = None) -> None:
        """Initialize the worker manager.
        
        Args:
            max_threads: Maximum number of concurrent threads. If None, uses
                        the global thread pool's default.
        """
        self._pool = QThreadPool.globalInstance()
        if max_threads is not None:
            self._pool.setMaxThreadCount(max_threads)
        self._active_workers: list[Worker] = []

    def start(self, worker: Worker) -> None:
        """Start a worker.
        
        Args:
            worker: The worker to start.
        """
        self._active_workers.append(worker)
        worker.signals.finished.connect(lambda: self._on_worker_finished(worker))
        self._pool.start(worker)

    def run_task(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_progress: Callable[[int, int], None] | None = None,
        on_result: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception, str], None] | None = None,
        on_finished: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> Worker:
        """Create and start a worker for a function.
        
        Args:
            fn: The function to run.
            *args: Positional arguments for fn.
            on_progress: Callback for progress updates (current, total).
            on_result: Callback for successful completion (result).
            on_error: Callback for errors (exception, traceback).
            on_finished: Callback when work is done (success or error).
            **kwargs: Keyword arguments for fn.
            
        Returns:
            The created Worker instance.
        """
        worker = Worker(fn, *args, **kwargs)
        
        if on_progress:
            worker.signals.progress.connect(on_progress)
        if on_result:
            worker.signals.result.connect(on_result)
        if on_error:
            worker.signals.error.connect(on_error)
        if on_finished:
            worker.signals.finished.connect(on_finished)

        self.start(worker)
        return worker

    def _on_worker_finished(self, worker: Worker) -> None:
        """Handle worker completion."""
        if worker in self._active_workers:
            self._active_workers.remove(worker)

    def cancel_all(self) -> None:
        """Request cancellation of all active workers."""
        for worker in self._active_workers:
            worker.cancel()

    @property
    def active_count(self) -> int:
        """Return the number of active workers."""
        return len(self._active_workers)

    def wait_for_done(self, timeout_ms: int = -1) -> bool:
        """Wait for all workers to complete.
        
        Args:
            timeout_ms: Maximum time to wait in milliseconds. -1 for infinite.
            
        Returns:
            True if all workers completed, False if timeout occurred.
        """
        return self._pool.waitForDone(timeout_ms)
