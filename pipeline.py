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
            
    async def _auto_process(self) -> None:
        """Automatically process queued items when auto_process is enabled."""
        async for result in self.process_queue():
            # In auto-process mode, we just discard results
            # Users should use process_queue() directly if they need results
            pass
            
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
        if not isinstance(input_data, list):
            raise TypeError("Pipeline input must be a list")

        current_data = input_data

        def chunk_results(results: List[Any], size: int) -> List[List[Any]]:
            """Split results into chunks of specified size."""
            return [results[i:i + size] for i in range(0, len(results), size)]

        for i, step in enumerate(self.steps):
            if not step.enabled:
                continue

            try:
                logger.info(f"Processing {len(current_data)} items in step {step.name} with {step.batch_config.execution_mode.value} mode")
                
                if step.step_type == StepType.PIPELINE:
                    async for result in step.func(current_data, step.config):
                        yield result
                        logger.info(f"Got pipeline result: {result.get('type')} - {result.get('step')}")
                        if isinstance(result, dict) and 'data' in result:
                            current_data = result['data'] if isinstance(result['data'], list) else [result['data']]
                else:
                    result = await step.func(current_data, step.config)
                    result = result if isinstance(result, list) else [result]
                    
                    result_chunks = chunk_results(result, step.batch_config.output_size)
                    for chunk in result_chunks:
                        yield {
                            'step': step.name,
                            'data': chunk,
                            'type': step.step_type,
                            'mode': step.execution_mode.value,
                            'is_final': i == len(self.steps) - 1
                        }
                    current_data = result

            except Exception as e:
                logger.error(f"Error in pipeline step {step.name}: {str(e)}", exc_info=True)
                yield {
                    'step': step.name,
                    'error': str(e),
                    'data': current_data,
                    'type': step.step_type.value,
                    'mode': step.batch_config.execution_mode.value
                }  
    def add_pipeline(self, name: str, pipeline: 'Pipeline', enabled: bool = True) -> None:
        """
        Add another pipeline as a step.
        
        Args:
            name: Unique name for this pipeline
            pipeline: Pipeline instance to add as a step
            enabled: Whether this step is enabled by default
        """
        async def pipeline_wrapper(data: List[Any], config: Dict[str, Any] = None) -> AsyncGenerator[Any, None]:
            buffer = []
            async for result in pipeline.execute(data):
                if isinstance(result, dict) and 'data' in result:
                    buffer.extend(result['data'] if isinstance(result['data'], list) else [result['data']])
                else:
                    buffer.append(result)
            for item in buffer:
                yield item

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
