import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
import simpy
import time
from data_loading_script import BaseDataLoader, RandomDataLoader
from model_executor_helper import ModelExecutorHelper


class MetricKeys(Enum):
    PREFILL_SUBMITTED = "prefill_submitted"
    PREFILL_COMPLETED = "prefill_completed"
    DECODE_SUBMITTED = "decode_submitted"
    COMPLETED = "completed"
    TOTAL_LATENCY = "total_latency"
    AVERAGE_LATENCY = "average_latency"


@dataclass
class Request:
    """Represents a processing request with token information."""

    id: int
    arrival_time: float
    input_tokens: int
    output_tokens: int
    remaining_tokens: int = field(init=False)
    prefill_remaining: int = field(init=False)

    def __post_init__(self) -> None:
        self.remaining_tokens = self.output_tokens
        self.prefill_remaining = self.input_tokens

    def __repr__(self) -> str:
        return f"Request({self.id}, decode_left={self.remaining_tokens}, prefill_left={self.prefill_remaining})"


@dataclass
class _BatchItem:
    req: Request
    tokens: int


@dataclass
class SimulationConfig:
    model_execution_helper: ModelExecutorHelper
    data_loader: Optional[BaseDataLoader] = None
    seed: int = 42
    simulation_time: float = 60.0
    max_batch_size: int = 4
    arrival_rate: float = 0.8
    min_tokens: int = 2
    max_tokens: int = 6
    token_decode_time: float = 0.2
    token_prefill_time: float = 0.1
    prefill_chunk_size: int = 2
    scheduler_timeout: float = 0.01
    max_output_tokens: int = 64
    num_gpus: int = 1
    inter_arrival_fn: Callable[[float], float] = field(
        default_factory=lambda: random.expovariate
    )
    enable_tracing: bool = False
    trace_output_file: str = "simulation_trace.json"

    def __post_init__(self) -> None:
        if self.data_loader is None:
            self.data_loader = RandomDataLoader(
                max_requests=1000,
                min_tokens=self.min_tokens,
                max_tokens=self.max_tokens,
                max_output_tokens=self.max_output_tokens,
                arrival_rate=self.arrival_rate,
                inter_arrival_fn=self.inter_arrival_fn,
            )
        if self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        if self.max_batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.simulation_time <= 0:
            raise ValueError("simulation_time must be positive")
        if self.token_decode_time < 0 or self.token_prefill_time < 0:
            raise ValueError("Token processing times must be non-negative")
        if self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if self.num_gpus <= 0:
            raise ValueError("num_gpus must be positive")

    # TODO: Add the simulation time
    def decode_time(self, total_tokens: int) -> float:
        """Calculate decode time for given number of tokens."""
        return total_tokens * self.token_decode_time

    def prefill_time(self, batch: List[_BatchItem]) -> float:
        seq_lens = [batch_item.req.prefill_remaining for batch_item in batch]
        return self.model_execution_helper.prefill(seq_lens, cached_context_lens=None)


class SimpleTracer:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.events = []

    def event(
        self, name: str, timestamp: float, phase: str = "i", gpu_id: int = 0, **args
    ):
        if self.enabled:
            self.events.append(
                {
                    "name": name,
                    "ts": int(timestamp * 1000000),
                    "ph": phase,
                    "pid": gpu_id,
                    "tid": 0,
                    "args": args,
                }
            )

    def save(self, filename: str):
        if self.enabled:
            with open(filename, "w") as f:
                json.dump({"traceEvents": self.events}, f)


class MetricsCollector:
    """Centralized metrics collection and calculation."""

    def __init__(self) -> None:
        self.metrics: Dict[str, float] = {
            key.value: 0.0 for key in MetricKeys if key != MetricKeys.AVERAGE_LATENCY
        }

    def increment(self, key: MetricKeys, value: float = 1.0) -> None:
        """Increment a metric by the given value."""
        self.metrics[key.value] += value

    def get_results(self) -> Dict[str, float]:
        """Calculate and return final metrics including average latency."""
        completed = self.metrics[MetricKeys.COMPLETED.value]
        avg_latency = (
            self.metrics[MetricKeys.TOTAL_LATENCY.value] / completed
            if completed > 0
            else 0.0
        )
        return {**self.metrics, MetricKeys.AVERAGE_LATENCY.value: avg_latency}


class GpuExecutor(ABC):
    def _setup_scheduler(
        self, env, gpu_resource, config, metrics, tracer, name, gpu_id
    ):
        """Initialize common scheduler components."""
        self.env = env
        self.gpu_resource = gpu_resource
        self.config = config
        self.metrics = metrics
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(name)
        self.tracer = tracer
        self.queue = simpy.Store(env)
        self.env.process(self._run())

    def _gpu_execute(self, name: str, duration: float, **trace_args):
        with self.gpu_resource.request() as gpu_req:
            yield gpu_req
            self.tracer.event(
                name, self.env.now, phase="B", gpu_id=self.gpu_id, **trace_args
            )
            yield self.env.timeout(duration)
            self.tracer.event(name, self.env.now, phase="E", gpu_id=self.gpu_id)


    @abstractmethod
    def _run(self) -> Generator[Any, None, None]:
        pass


class PrefillScheduler(GpuExecutor):
    def __init__(
        self,
        env: simpy.Environment,
        gpu_resource: simpy.Resource,
        config: SimulationConfig,
        decode_scheduler: "DecodeScheduler",
        metrics: MetricsCollector,
        tracer: SimpleTracer,
        gpu_id: int,
    ) -> None:
        self._setup_scheduler(
            env, gpu_resource, config, metrics, tracer, "Prefill", gpu_id
        )
        self.decode_scheduler = decode_scheduler

    def add_request(self, request: Request) -> None:
        """Add request for prefill processing."""
        self.queue.put(request)
        self.metrics.increment(MetricKeys.PREFILL_SUBMITTED)
        self.logger.debug(f"GPU{self.gpu_id}: Queued {request} at {self.env.now:.2f}")

    def _run(self) -> Generator[simpy.Event, None, None]:
        """Process prefill requests in batches up to chunk budget."""
        while True:
            first_request = yield self.queue.get()
            if isinstance(first_request, Request):
                batch, total_tokens = yield from self._collect_batch(first_request)
            
                yield from self._execute_prefill_step(batch, total_tokens)
                self._finish(batch)
            

    def _collect_batch(self, first_request: Request) -> Generator[simpy.Event, None, Tuple[List[_BatchItem], int]]:
        batch, used, limit = [], 0, self.config.prefill_chunk_size
        
        tokens = min(limit - used, first_request.prefill_remaining)
        batch.append(_BatchItem(first_request, tokens))
        used += tokens
        
        # Try to collect more requests without blocking
        while used < limit:
            try:
                additional_req_event = self.queue.get() | self.env.timeout(0)
                additional_req = yield additional_req_event
                if isinstance(additional_req, Request):
                    tokens = min(limit - used, additional_req.prefill_remaining)
                    batch.append(_BatchItem(additional_req, tokens))
                    used += tokens
                else:
                    break  # Timeout occurred
            except:
                break
        
        return batch, used

    def _execute_prefill_step(
        self,
        batch: List[_BatchItem],
        total_tokens: int,
    ) -> Generator[simpy.Event, None, None]:
        """Execute the prefill batch."""
        delay = self.config.prefill_time(batch)
        self.logger.info(
            f"GPU{self.gpu_id} [{self.env.now:.2f}] Prefill {sum(item.tokens for item in batch)} tokens "
            f"from {len(batch)} requests ({delay:.2f}s)"
        )
        yield from self._gpu_execute(
            "Prefill Batch",
            delay,
            tokens=sum(item.tokens for item in batch),
            requests=len(batch),
        )

    def _finish(self, batch: List[_BatchItem]) -> None:
        """Finish processing the batch and handle completions."""
        for item in batch:
            item.req.prefill_remaining -= item.tokens
            if item.req.prefill_remaining > 0:
                self.queue.put(item.req)
            else:
                self.metrics.increment(MetricKeys.PREFILL_COMPLETED)
                self.logger.info(
                    f"GPU{self.gpu_id} [{self.env.now:.2f}] Prefill complete {item.req}"
                )
                self.decode_scheduler.add_request(item.req)


class DecodeScheduler(GpuExecutor):
    def __init__(
        self,
        env: simpy.Environment,
        gpu_resource: simpy.Resource,
        config: SimulationConfig,
        metrics: MetricsCollector,
        tracer: SimpleTracer,
        gpu_id: int,
    ) -> None:
        self._setup_scheduler(
            env, gpu_resource, config, metrics, tracer, "Decode", gpu_id
        )
        self.running: list[Request] = []

    def add_request(self, request: Request) -> None:
        self.queue.put(request)
        self.metrics.increment(MetricKeys.DECODE_SUBMITTED)
        self.logger.debug(f"GPU{self.gpu_id}: Queued {request} at {self.env.now:.2f}")

    def _run(self) -> Generator[simpy.Event, None, None]:
        while True:
            if not self.running:
                first_request = yield self.queue.get()
                if isinstance(first_request, Request):
                    self.running.append(first_request)
            
            # Try to collect more requests without blocking
            while len(self.running) < self.config.max_batch_size:
                try:
                    additional_req_event = self.queue.get() | self.env.timeout(0)
                    additional_req = yield additional_req_event
                    if isinstance(additional_req, Request):
                        self.running.append(additional_req)
                    else:
                        break  # Timeout occurred
                except:
                    break
            
            yield from self._execute_decode_step()
            self._finish_decode_step()

    def _execute_decode_step(self) -> Generator[simpy.Event, None, None]:
        """Execute a single decode step for the current batch."""
        step_time = self.config.decode_time(len(self.running))
        self.logger.info(
            f"GPU{self.gpu_id} [{self.env.now:.2f}] Decode step "
            f"batch={self.running} size={len(self.running)}"
        )
        yield from self._gpu_execute(
            "Decode Step", step_time, batch_size=len(self.running)
        )

    def _finish_decode_step(self) -> None:
        """Process decode step results and handle completions."""
        still_running: list[Request] = []
        for req in self.running:
            req.remaining_tokens -= 1
            if req.remaining_tokens > 0:
                still_running.append(req)
            else:
                latency = self.env.now - req.arrival_time
                self.metrics.increment(MetricKeys.COMPLETED)
                self.metrics.increment(MetricKeys.TOTAL_LATENCY, latency)
                self.logger.info(
                    f"GPU{self.gpu_id} [{self.env.now:.2f}] Finished {req}, "
                    f"latency={latency:.2f}"
                )
        self.running = still_running


class RoundRobinScheduler:
    def __init__(
        self,
        env: simpy.Environment,
        config: SimulationConfig,
        metrics: MetricsCollector,
        tracer: SimpleTracer,
    ):
        self.env = env
        self.config = config
        self.metrics = metrics
        self.tracer = tracer
        self.current_gpu = 0

        self.gpus = [simpy.Resource(env, capacity=1) for _ in range(config.num_gpus)]
        self.decode_schedulers = []
        self.prefill_schedulers = []
        for gpu_id, gpu in enumerate(self.gpus):
            decoder = DecodeScheduler(env, gpu, config, metrics, tracer, gpu_id)
            prefill = PrefillScheduler(
                env, gpu, config, decoder, metrics, tracer, gpu_id
            )
            self.prefill_schedulers.append(prefill)
            self.decode_schedulers.append(decoder)

    def add_request(self, request: Request) -> None:
        """Add request to next GPU in round-robin fashion."""
        gpu_id = self.current_gpu
        self.prefill_schedulers[gpu_id].add_request(request)
        self.current_gpu = (self.current_gpu + 1) % self.config.num_gpus
        logging.getLogger("RoundRobin").info(f"Assigned {request} to GPU{gpu_id}")


class Simulation:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        random.seed(config.seed)
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
        )
        self.env = simpy.Environment()
        self.metrics = MetricsCollector()
        self.tracer = SimpleTracer(config.enable_tracing)
        self.scheduler = RoundRobinScheduler(
            self.env, config, self.metrics, self.tracer
        )
        self.env.process(self._generate_requests())
        data_loader = self.config.data_loader
        assert data_loader is not None
        self.requests_data = data_loader.get_requests()
        self.inter_arrival_times = data_loader.get_inter_arrival_times()

    def _generate_requests(self) -> Generator[simpy.Event, None, None]:
        req_id = 0
        for request_data, delay in zip(self.requests_data, self.inter_arrival_times):
            yield self.env.timeout(delay)
            req_id += 1
            self.scheduler.add_request(
                Request(
                    req_id,
                    self.env.now,
                    request_data.input_tokens,
                    request_data.output_tokens,
                )
            )

    def run(self) -> Dict[str, float]:
        """Run the simulation and return results."""
        self.env.run(until=self.config.simulation_time)
        self.tracer.save(self.config.trace_output_file)
        return self.metrics.get_results()

if __name__ == "__main__":
    model_execution_helper = ModelExecutorHelper(
        onnx_model_decode=ModelExecutorHelper.get_onnx_model_prefill_from_folder(
            "profile_output_a100", "deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B_TP_1"
        ),
        onnx_model_prefill=ModelExecutorHelper.get_onnx_model_prefill_from_folder(
            "profile_output_a100", "deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B_TP_1"
        ),
    )
    logging.disable(logging.CRITICAL)
    dataloader = RandomDataLoader(
        min_tokens=64,
        max_tokens=128,
        max_output_tokens=128,
        max_requests=100000, 
        arrival_rate=50
    )
    config = SimulationConfig(model_execution_helper=model_execution_helper, data_loader=dataloader)
    config.enable_tracing = True
    config.num_gpus = 1024
    config.arrival_rate = config.arrival_rate * config.num_gpus
    config.simulation_time = 1800.0
    sim = Simulation(config)
    start_time = time.perf_counter()
    results = sim.run()
    end_time = time.perf_counter()
    print("Results:", results)
    print(f"Simulation took {end_time - start_time:.2f} seconds")
