"""Batch execution framework for parallel processing."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed


class BatchExecutor[TInput, TOutput](ABC):
    """Generic batch execution with threading.

    Provides a unified pattern for parallel execution of items.
    Subclasses implement execute_one() to define the work for each item.
    """

    def __init__(self, parallel: int = 1):
        """
        Args:
            parallel: Number of parallel execution threads.
        """
        self.parallel = parallel

    @abstractmethod
    def execute_one(self, item: TInput) -> TOutput:
        """Execute a single item.

        Args:
            item: The item to process.

        Returns:
            The result of processing the item.
        """
        ...

    def run(self, items: list[TInput]) -> list[TOutput]:
        """Run all items with parallel execution.

        Args:
            items: List of items to process.

        Returns:
            List of results in arbitrary order (for parallel execution).
        """
        results: list[TOutput] = []

        if self.parallel <= 1:
            for item in items:
                results.append(self.execute_one(item))
        else:
            with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                futures = {executor.submit(self.execute_one, item): item for item in items}
                for future in as_completed(futures):
                    results.append(future.result())

        return results
