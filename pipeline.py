from typing import Dict, Any, List, Callable, AsyncGenerator, TypeVar, Protocol, Union, runtime_checkable, Optional, Tuple, Iterator, Iterable
import logging
from dataclasses import dataclass
from enum import Enum
from collections import deque
import asyncio

logger = logging.getLogger(__name__)

T = TypeVar('T')
U = TypeVar('U')

@runtime_checkable
class MapFunction(Protocol[T, U]):
    async def __call__(self, items: List[T], config: Dict[str, Any] = None) -> List[U]: ...

@runtime_checkable
class FilterFunction(Protocol[T]):
    async def __call__(self, items: List[T], config: Dict[str, Any] = None) -> List[T]: ...

class StepType(Enum):
    MAP = "map"
    FILTER = "filter"
    TRANSFORM = "transform"  # For backwards compatibility
    PIPELINE = "pipeline"    # For nested pipelines

class ExecutionMode(Enum):
    IMMEDIATE = "immediate"  # Process items one at a time
    BATCH = "batch"         # Process items in batches
    ALL = "all"            # Wait for all items before processing

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    input_size: int=10  # How many items to send to the function
    output_size: int=10  # How many items to yield at a time
    execution_mode: ExecutionMode = ExecutionMode.IMMEDIATE  # How to execute the step

    @classmethod
    def from_spec(cls, spec: Union[int, 'BatchConfig', Dict[str, Any]] = None) -> 'BatchConfig':
        """Create a BatchConfig from various specification formats.
        
        Args:
            spec: Can be:
                - int: Used for both input and output size
                - BatchConfig: Used directly
                - Dict: Used to set specific fields
                - None: Uses default values
        """
        if spec is None:
            return cls()
        if isinstance(spec, BatchConfig):
            return spec
        if isinstance(spec, int):
            return cls(input_size=spec, output_size=spec)
        if isinstance(spec, dict):
            return cls(**spec)
        raise ValueError(f"Invalid BatchConfig specification: {spec}")

@dataclass
class PipelineConfig:
    """Configuration for Pipeline behavior."""
    max_queue_size: Optional[int] = None  # Maximum number of items in queue (None for unlimited)
    auto_process: bool = False  # Whether to automatically process items when added
    batch_size: Optional[int] = None  # Default batch size for steps that don't specify one

def batch(func: Union[MapFunction[T, U], FilterFunction[T]], 
          input_batch_size: Optional[int] = None,
          output_batch_size: Optional[int] = None) -> Union[MapFunction[T, U], FilterFunction[T]]:
    """Wrapper to indicate a function should process items in batches.
    
    Args:
        func: The function to wrap
        input_batch_size: Number of items to process at once (default: 10)
        output_batch_size: Number of items to yield at once (default: same as input_batch_size)
    """
    # Default input batch size to 10 if not specified
    input_size = input_batch_size if input_batch_size is not None else 10
    # Default output batch size to input batch size if not specified
    output_size = output_batch_size if output_batch_size is not None else input_size
    
    batch_config = BatchConfig(input_size, output_size)

    # Create a wrapper function that will hold our attributes
    async def wrapper(items: List[Any], config: Dict[str, Any] = None) -> List[Any]:
        return await func(items, config)
    
    # Set attributes on the wrapper function
    wrapper._execution_mode = ExecutionMode.BATCH
    wrapper._batch_config = batch_config
    
    return wrapper

def all(func: Union[MapFunction[T, U], FilterFunction[T]], 
        output_batch_size: Optional[int] = None) -> Union[MapFunction[T, U], FilterFunction[T]]:
    """Wrapper to indicate a function should wait for all items before processing.
    
    Args:
        func: The function to wrap
        output_batch_size: Optional size for output batching
    """
    # Create a wrapper function that will hold our attributes
    async def wrapper(items: List[Any], config: Dict[str, Any] = None) -> List[Any]:
        return await func(items, config)
    
    # Set attributes on the wrapper function
    wrapper._execution_mode = ExecutionMode.ALL
    if output_batch_size is not None:
        wrapper._batch_config = BatchConfig(None, output_batch_size)
    
    return wrapper

@dataclass
class PipelineStep:
    """A step in the pipeline with its configuration."""
    name: str
    func: Union[MapFunction, FilterFunction]
    step_type: StepType
    enabled: bool = True
    config: Dict[str, Any] = None
    execution_mode: ExecutionMode = ExecutionMode.IMMEDIATE
    batch_config: Optional[BatchConfig] = None

    def __post_init__(self):
        """Set execution mode and batch config from function if specified."""
        if hasattr(self.func, '_execution_mode'):
            self.execution_mode = getattr(self.func, '_execution_mode')
        if hasattr(self.func, '_batch_config'):
            self.batch_config = getattr(self.func, '_batch_config')
        elif self.batch_config is None:
            self.batch_config = BatchConfig(1, 1)  # Default to processing one at a time

class Pipeline:
    """Manages the execution flow of processing steps."""
    
    def __init__(self, 
                 steps: Optional[List[PipelineStep]] = None,
                 config: Optional[Dict[str, Dict[str, Any]]] = {},
                 initial_items: Optional[Iterable[Any]] = None,
                 pipeline_config: Optional[PipelineConfig] = None) -> None:
        """Initialize a pipeline with optional initial configuration.
        
        Args:
            steps: Initial processing steps to add
            config: Initial configuration for steps
            initial_items: Initial items to add to the queue
            pipeline_config: Configuration for pipeline behavior
        """
        self.steps: List[PipelineStep] = steps or []
        self.config: Dict[str, Dict[str, Any]] = config or {}
        self.pipeline_config = pipeline_config or PipelineConfig()
        
        # Initialize queue with optional max size from config
        self.queue: deque = deque(maxlen=self.pipeline_config.max_queue_size)
        self.is_processing: bool = False
        
        # Add any initial items
        if initial_items:
            self.extend(initial_items)
            
        # Apply any initial config
        if config:
            self.set_config(config)
            
        # Auto-process if configured and we have initial items
        if self.pipeline_config.auto_process and initial_items:
            asyncio.create_task(self._auto_process())
            
    async def initialize(self):
        """Initialize async resources."""
        # Initialize any async resources needed by steps
        for step in self.steps:
            if hasattr(step.func, '__aenter__'):
                try:
                    await step.func.__aenter__()
                except Exception as e:
                    logger.error(f"Error initializing step {step.name}: {str(e)}", exc_info=True)
        return self
            
    async def __aenter__(self):
        """Enter async context."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        # Clean up any resources
        for step in self.steps:
            if hasattr(step.func, '__aexit__'):
                try:
                    await step.func.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error cleaning up step {step.name}: {str(e)}", exc_info=True)
                    
    def __len__(self) -> int:
        """Return number of items in queue."""
        return len(self.queue)
        
    def __iter__(self) -> Iterator[Any]:
        """Return iterator over queued items."""
        return iter(self.queue)
        
    def __bool__(self) -> bool:
        """Return True if queue has items."""
        return bool(self.queue)

    def append(self, item: Any) -> None:
        """Add an item to the pipeline's processing queue."""
        self.queue.append(item)
        if self.pipeline_config.auto_process:
            asyncio.create_task(self._auto_process())
            
    def extend(self, items: List[Any]) -> None:
        """Add multiple items to the pipeline's processing queue."""
        self.queue.extend(items)
        if self.pipeline_config.auto_process:
            asyncio.create_task(self._auto_process())
            
    def clear(self) -> None:
        """Remove all items from the queue."""
        self.queue.clear()
        
    def copy(self) -> deque:
        """Return a shallow copy of the queue."""
        return self.queue.copy()
        
    def count(self, value: Any) -> int:
        """Return number of occurrences of value in queue."""
        return self.queue.count(value)
        
    def pop(self) -> Any:
        """Remove and return an item from the right side of queue."""
        return self.queue.pop()
        
    def popleft(self) -> Any:
        """Remove and return an item from the left side of queue."""
        return self.queue.popleft()
        
    def remove(self, value: Any) -> None:
        """Remove first occurrence of value in queue."""
        self.queue.remove(value)
        
    def reverse(self) -> None:
        """Reverse the elements of queue in-place."""
        self.queue.reverse()
        
    def rotate(self, n: int = 1) -> None:
        """Rotate the queue n steps to the right (or left if negative)."""
        self.queue.rotate(n)
        
    def maxlen(self) -> Optional[int]:
        """Return maximum size of queue or None if unbounded."""
        return self.queue.maxlen
        
    @property
    def is_empty(self) -> bool:
        """Return True if queue is empty."""
        return len(self.queue) == 0

    async def process_queue(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process all items in the queue through the pipeline.
        
        Yields:
            Dictionary containing step results and metadata
        """
        if self.is_processing:
            return
            
        self.is_processing = True
        try:
            while self.queue:
                # Process items in batches based on first step's batch config
                batch_size = self.steps[0].batch_config.input_size if self.steps else 1
                batch = []
                
                while len(batch) < batch_size and self.queue:
                    batch.append(self.queue.popleft())
                
                async for result in self.execute(batch):
                    yield result
        finally:
            self.is_processing = False
    
    async def execute(self, input_data: List[Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process data through pipeline steps.
        
        For immediate mode steps, results will be processed in parallel as they arrive.
        For all mode steps, we'll wait for all results before proceeding.
        """
        current_data = input_data
        step_queues = []  # Queue for each step
        step_tasks = []  # Tasks for each step
        results_queue = asyncio.Queue()

        async def process_step_result(step_idx: int, result: Any, next_queue: Optional[asyncio.Queue] = None):
            """Process a single result from a step and optionally forward it to next step."""
            if result:  # Skip empty results
                result = result if isinstance(result, list) else [result]
                
                # Create result dict with metadata
                result_dict = {
                    'step': self.steps[step_idx].name,
                    'data': result,
                    'type': self.steps[step_idx].step_type.value,
                    'mode': self.steps[step_idx].batch_config.execution_mode.value,
                    'is_final': step_idx == len(self.steps) - 1
                }
                
                # Put result in results queue for yielding
                await results_queue.put(result_dict)
                
                # If there's a next step, put each result individually for immediate mode
                if next_queue is not None:
                    next_step = self.steps[step_idx + 1]
                    if next_step.batch_config.execution_mode == ExecutionMode.IMMEDIATE:
                        # For immediate mode, send each item individually
                        for item in result:
                            await next_queue.put([item])
                    else:
                        # For ALL mode, send the entire batch
                        await next_queue.put(result)
                        
        async def process_step(step_idx: int, step: PipelineStep, input_queue: asyncio.Queue, next_queue: Optional[asyncio.Queue] = None):
            """Process all items from input queue through a single step."""
            try:
                while True:
                    try:
                        # Get next batch of items from input queue
                        batch = []
                        
                        # Get first item
                        try:
                            data = await input_queue.get()
                            batch.extend(data)
                            input_queue.task_done()
                            
                            # If immediate mode, process this item right away
                            if step.batch_config.execution_mode == ExecutionMode.IMMEDIATE:
                                # Try to get more items that arrived while processing
                                while not input_queue.empty():
                                    try:
                                        more_data = input_queue.get_nowait()
                                        batch.extend(more_data)
                                        input_queue.task_done()
                                    except asyncio.QueueEmpty:
                                        break
                        except asyncio.CancelledError:
                            break
                        
                        if not batch:
                            continue
                            
                        # Process the batch
                        if step.step_type == StepType.PIPELINE:
                            async for result in step.func(batch, step.config):
                                await process_step_result(step_idx, result, next_queue)
                        else:
                            async for result in step.func(batch, step.config):
                                await process_step_result(step_idx, result, next_queue)
                                
                    except asyncio.CancelledError:
                        break
                        
            except Exception as e:
                logger.error(f"Error in step {step.name}: {str(e)}", exc_info=True)
            finally:
                if hasattr(step.func, '__aexit__'):
                    try:
                        await step.func.__aexit__(None, None, None)
                    except Exception as e:
                        logger.error(f"Error cleaning up step {step.name}: {str(e)}", exc_info=True)

        try:
            # Create queues for each step
            for i in range(len(self.steps)):
                step_queues.append(asyncio.Queue())
            
            # Create tasks for each step
            for i, step in enumerate(self.steps):
                if not step.enabled:
                    continue
                    
                # Get input and output queues for this step
                input_queue = step_queues[i]
                next_queue = step_queues[i + 1] if i < len(self.steps) - 1 else None
                
                # Create and start task for this step
                task = asyncio.create_task(process_step(i, step, input_queue, next_queue))
                step_tasks.append(task)
            
            # Put initial data in first queue
            if self.steps and self.steps[0].batch_config.execution_mode == ExecutionMode.IMMEDIATE:
                # For immediate mode, send each item individually
                for item in current_data:
                    await step_queues[0].put([item])
            else:
                # For ALL mode, send the entire batch
                await step_queues[0].put(current_data)
            
            # Process results as they arrive
            while True:
                try:
                    # Check if all steps are done and queues are empty
                    all_done = True
                    for i, queue in enumerate(step_queues):
                        if not queue.empty() or any(not t.done() for t in step_tasks):
                            all_done = False
                            break
                    
                    if all_done and results_queue.empty():
                        break
                        
                    # Get and yield next result
                    try:
                        result = await asyncio.wait_for(results_queue.get(), timeout=0.1)
                        yield result
                    except asyncio.TimeoutError:
                        continue
                        
                except Exception as e:
                    logger.error(f"Error processing results: {str(e)}", exc_info=True)
                    break
                    
        finally:
            # Cancel all tasks
            for task in step_tasks:
                if not task.done():
                    task.cancel()
                    
            # Wait for tasks to complete
            await asyncio.gather(*step_tasks, return_exceptions=True)

    def add_pipeline(self, name: str, pipeline: 'Pipeline', enabled: bool = True) -> None:
        """
        Add another pipeline as a step.
        
        Args:
            name: Unique name for this pipeline
            pipeline: Pipeline instance to add as a step
            enabled: Whether this step is enabled by default
        """
        async def pipeline_wrapper(data: List[Any], config: Dict[str, Any] = None) -> AsyncGenerator[Any, None]:
            """Wrapper function for nested pipeline execution."""
            async for result in pipeline.execute(data):
                # Pass through all results directly to preserve metadata
                yield result

        self.steps.append(PipelineStep(
            name=name,
            func=pipeline_wrapper,
            step_type=StepType.PIPELINE,
            enabled=enabled,
            config=self.config
        ))


    def add_map(self, name: str, func: Callable, *, config: Dict = None, enabled: bool = True,
                execution_mode: ExecutionMode = None) -> None:
        """Add a map step to the pipeline."""
        batch_config = BatchConfig(execution_mode=execution_mode) if execution_mode else BatchConfig()
        self.add_step(PipelineStep(
            name=name,
            func=func,
            step_type=StepType.MAP,
            config=config or {},
            batch_config=batch_config,
            enabled=enabled
        ))

    def add_filter(self, name: str, func: Callable, *, config: Dict = None,
                execution_mode: ExecutionMode = None, enabled: bool = True) -> None:
        """Add a filter step to the pipeline."""
        batch_config = BatchConfig(execution_mode=execution_mode) if execution_mode else BatchConfig()
        self.add_step(PipelineStep(
            name=name,
            func=func,
            step_type=StepType.FILTER,
            config=config or {},
            batch_config=batch_config,
            enabled=enabled
        ))

    def add_transform(self, name: str, func: Callable, *, config: Dict = None,
                    execution_mode: ExecutionMode = None, enabled: bool = True) -> None:
        """Add a transform step to the pipeline."""
        if execution_mode:
            batch_config = BatchConfig(execution_mode=execution_mode)
        self.add_step(PipelineStep(
            name=name,
            func=func,
            step_type=StepType.TRANSFORM,
            config=config or {},
            batch_config=batch_config or BatchConfig(),
            enabled=enabled
        ))

    def add_step(self, step: PipelineStep) -> None:
        """
        Add a pipeline step.
        
        Args:
            step: PipelineStep instance to add
        """
        self.steps.append(step)
        
    def set_config(self, config: Dict[str, Dict[str, Any]]) -> None:
        """
        Set configuration for pipeline steps.
        
        Args:
            config: Dictionary mapping step names to their configurations
        """
        self.config = config
        for step in self.steps:
            step.config = config.get(step.name, {})

    async def _auto_process(self) -> None:
        """Automatically process queued items when auto_process is enabled."""
        async for result in self.process_queue():
            # In auto-process mode, we just discard results
            # Users should use process_queue() directly if they need results
            pass
