import simpy
import random
import logging
import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, List, Dict, Optional, Generator, Tuple, Any


class MetricKeys(Enum):
    PREFILL_SUBMITTED = "prefill_submitted"
    PREFILL_COMPLETED = "prefill_completed"
    DECODE_SUBMITTED = "decode_submitted"
    COMPLETED = "completed"
    TOTAL_LATENCY = "total_latency"
    AVERAGE_LATENCY = "average_latency"


@dataclass
class SimulationConfig:
    seed: int = 42
    simulation_time: float = 60.0
    max_batch_size: int = 4
    arrival_rate: float = 0.8
    min_tokens: int = 2
    max_tokens: int = 6
    token_decode_time: float = 0.2  # seconds per token for decode
    token_prefill_time: float = 0.1  # seconds per token for prefill
    prefill_chunk_size: int = 2  # tokens per prefill chunk
    scheduler_timeout: float = 0.01  # timeout when queues are empty
    max_output_tokens: int = 64
    num_gpus: int = 1  # number of GPUs for round robin scheduling
    inter_arrival_fn: Callable[[float], float] = field(
        default_factory=lambda: random.expovariate
    )
    enable_tracing: bool = False
    trace_output_file: str = "simulation_trace.json"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        if self.max_batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.simulation_time <= 0:
            raise ValueError("simulation_time must be positive")
        if (
            self.min_tokens <= 0
            or self.max_tokens <= 0
            or self.min_tokens > self.max_tokens
        ):
            raise ValueError("Invalid token range")
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

    def prefill_time(self, tokens: int) -> float:
        """Calculate prefill time for given number of tokens."""
        return tokens * self.token_prefill_time


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


class SimpleTracer:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.events = []
    
    def event(self, name: str, timestamp: float, phase: str = "i", **args):
        if self.enabled:
            self.events.append({
                "name": name, "ts": int(timestamp * 1000000), 
                "ph": phase, "pid": 0, "tid": 0, "args": args
            })
    
    def save(self, filename: str):
        if self.enabled:
            with open(filename, 'w') as f:
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


class GpuExecutor:
    def _setup_scheduler(self, env, gpu_resource, config, metrics, tracer, name):
        """Initialize common scheduler components."""
        self.env = env
        self.gpu_resource = gpu_resource
        self.config = config
        self.metrics = metrics
        self.logger = logging.getLogger(name)
        self.tracer = tracer
        self.queue = deque()
        self.env.process(self._run())

    def _gpu_execute(self, name: str, duration: float, **trace_args):
        with self.gpu_resource.request() as gpu_req:
            yield gpu_req
            self.tracer.event(name, self.env.now, phase="B", **trace_args)
            yield self.env.timeout(duration)
            self.tracer.event(name, self.env.now, phase="E")

    def _idle_wait(self):
        yield self.env.timeout(self.config.scheduler_timeout)


class PrefillScheduler(GpuExecutor):
    def __init__(
        self,
        env: simpy.Environment,
        gpu_resource: simpy.Resource,
        config: SimulationConfig,
        decode_scheduler: "DecodeScheduler",
        metrics: MetricsCollector,
        tracer: SimpleTracer,
    ) -> None:
        self._setup_scheduler(env, gpu_resource, config, metrics, tracer, "Prefill")
        self.decode_scheduler = decode_scheduler

    def add_request(self, request: Request) -> None:
        """Add request for prefill processing."""
        self.queue.append(request)
        self.metrics.increment(MetricKeys.PREFILL_SUBMITTED)
        self.logger.debug(f"Queued {request} at {self.env.now:.2f}")

    def _run(self) -> Generator[simpy.Event, None, None]:
        """Process prefill requests in batches up to chunk budget."""
        while True:
            if not self.queue:
                yield from self._idle_wait()
                continue

            batch, total_tokens = self._collect_batch()
            if total_tokens == 0:
                yield from self._idle_wait()
                continue

            yield from self._execute(batch, total_tokens)
            self._finish(batch)

    def _collect_batch(self) -> Tuple[List[_BatchItem], int]:
        """Collect requests into a batch up to the chunk budget."""
        batch, used, limit = [], 0, self.config.prefill_chunk_size
        while self.queue and used < limit:
            req = self.queue.popleft()
            tokens = min(limit - used, req.prefill_remaining)
            batch.append(_BatchItem(req, tokens))
            used += tokens
        return batch, used

    def _execute(
        self,
        batch: List[_BatchItem],
        total_tokens: int,
    ) -> Generator[simpy.Event, None, None]:
        """Execute the prefill batch."""
        delay = self.config.prefill_time(total_tokens)
        self.logger.info(
            f"[{self.env.now:.2f}] Prefill {total_tokens} tokens "
            f"from {len(batch)} requests ({delay:.2f}s)"
        )
        yield from self._gpu_execute("Prefill Batch", delay, tokens=total_tokens, requests=len(batch))

    def _finish(self, batch: List[_BatchItem]) -> None:
        """Finish processing the batch and handle completions."""
        for item in batch:
            item.req.prefill_remaining -= item.tokens
            if item.req.prefill_remaining > 0:
                self.queue.append(item.req)
            else:
                self.metrics.increment(MetricKeys.PREFILL_COMPLETED)
                self.logger.info(f"[{self.env.now:.2f}] Prefill complete {item.req}")
                self.decode_scheduler.add_request(item.req)


class DecodeScheduler(GpuExecutor):
    def __init__(
        self,
        env: simpy.Environment,
        gpu_resource: simpy.Resource,
        config: SimulationConfig,
        metrics: MetricsCollector,
        tracer: SimpleTracer,
    ) -> None:
        self._setup_scheduler(env, gpu_resource, config, metrics, tracer, "Decode")
        self.running: list[Request] = []

    def add_request(self, request: Request) -> None:
        self.queue.append(request)
        self.metrics.increment(MetricKeys.DECODE_SUBMITTED)
        self.logger.debug(f"Queued {request} at {self.env.now:.2f}")

    def _run(self) -> Generator[simpy.Event, None, None]:
        """Continuous‑batch decode loop."""
        while True:
            while self.queue and len(self.running) < self.config.max_batch_size:
                self.running.append(self.queue.popleft())

            if not self.running:
                self.logger.debug(f"[{self.env.now:.2f}] Decode idle")
                yield from self._idle_wait()
                continue
            yield from self._execute_decode_step()
            self._finish_decode_step()

    def _execute_decode_step(self) -> Generator[simpy.Event, None, None]:
        """Execute a single decode step for the current batch."""
        step_time = self.config.decode_time(len(self.running))
        self.logger.info(
            f"[{self.env.now:.2f}] Decode step "
            f"batch={self.running} size={len(self.running)}"
        )
        yield from self._gpu_execute("Decode Step", step_time, batch_size=len(self.running))

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
                    f"[{self.env.now:.2f}] Finished {req}, "
                    f"latency={latency:.2f}"
                )
        self.running = still_running


class RoundRobinScheduler:
    def __init__(self, env: simpy.Environment, config: SimulationConfig, metrics: MetricsCollector, tracer: SimpleTracer):
        self.env = env
        self.config = config
        self.metrics = metrics
        self.tracer = tracer
        self.current_gpu = 0
        
        # Create GPU resources and schedulers
        self.gpus = [simpy.Resource(env, capacity=1) for _ in range(config.num_gpus)]
        self.decode_schedulers = []
        self.prefill_schedulers = []
        for i, gpu in enumerate(self.gpus):
            decoder = DecodeScheduler(env, gpu, config, metrics, tracer)
            prefill = PrefillScheduler(env, gpu, config, decoder, metrics, tracer)
            self.prefill_schedulers.append(prefill)
            self.decode_schedulers.append(decoder)
    
    def add_request(self, request: Request) -> None:
        """Add request to next GPU in round-robin fashion."""
        gpu_id = self.current_gpu
        self.prefill_schedulers[gpu_id].add_request(request)
        self.current_gpu = (self.current_gpu + 1) % self.config.num_gpus
        logging.getLogger("RoundRobin").debug(f"Assigned {request} to GPU {gpu_id}")


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
        self.scheduler = RoundRobinScheduler(self.env, config, self.metrics, self.tracer)
        self.env.process(self._generate_requests())

    def _generate_requests(self) -> Generator[simpy.Event, None, None]:
        """Generate requests according to arrival rate."""
        req_id = 0
        while True:
            yield self.env.timeout(
                self.config.inter_arrival_fn(self.config.arrival_rate)
            )
            req_id += 1
            input_tokens = random.randint(
                self.config.min_tokens, self.config.max_tokens
            )
            output_tokens = random.randint(
                self.config.min_tokens, self.config.max_output_tokens
            )
            self.scheduler.add_request(
                Request(req_id, self.env.now, input_tokens, output_tokens)
            )

    def run(self) -> Dict[str, float]:
        """Run the simulation and return results."""
        self.env.run(until=self.config.simulation_time)
        self.tracer.save(self.config.trace_output_file)
        return self.metrics.get_results()


def run_simulation(config: Optional[SimulationConfig] = None) -> Dict[str, float]:
    """Run simulation with optional configuration."""
    return Simulation(config or SimulationConfig()).run()


if __name__ == "__main__":
    config = SimulationConfig()
    config.enable_tracing = True
    config.num_gpus = 2  # Test with 2 GPUs
    config.simulation_time = 10.0  # Shorter simulation for demo
    results = run_simulation(config)
    print("Results:", results)
