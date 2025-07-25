import simpy
import random
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, List, Dict, Optional, Generator, Tuple


# ---- Constants ----
class MetricKeys(Enum):
    """Enumeration of metric keys to prevent typos."""

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
    batch_size: int = 4
    arrival_rate: float = 0.8
    min_tokens: int = 2
    max_tokens: int = 6
    token_decode_time: float = 0.2  # seconds per token for decode
    token_prefill_time: float = 0.1  # seconds per token for prefill
    prefill_chunk_size: int = 2  # tokens per prefill chunk
    scheduler_timeout: float = 0.01  # timeout when queues are empty
    max_output_tokens: int = 64
    inter_arrival_fn: Callable[[float], float] = field(
        default_factory=lambda: random.expovariate
    )

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        if self.batch_size <= 0:
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

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) for backward compatibility."""
        return self.input_tokens + self.output_tokens

    def __repr__(self) -> str:
        return f"Request({self.id}, decode_left={self.remaining_tokens}, prefill_left={self.prefill_remaining})"


@dataclass
class _BatchItem:
    """Helper class for batching prefill processing."""

    req: Request
    tokens: int


# ---- Metrics Collection ----
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


# ---- Base Scheduler ----
class BaseScheduler(ABC):
    """Abstract base class for request schedulers."""

    def __init__(
        self,
        env: simpy.Environment,
        gpu_resource: simpy.Resource,
        config: SimulationConfig,
        metrics: MetricsCollector,
        name: str,
    ) -> None:
        self.env = env
        self.gpu_resource = gpu_resource
        self.config = config
        self.metrics = metrics
        self.logger = logging.getLogger(name)
        self.env.process(self._run())

    @abstractmethod
    def _run(self) -> Generator[simpy.Event, None, None]:
        """Main scheduler loop - must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add request for processing - must be implemented by subclasses."""
        raise NotImplementedError


# ---- Prefill Scheduler ----
class PrefillScheduler(BaseScheduler):
    """Handles prefill processing of requests in batches with chunk budget."""

    def __init__(
        self,
        env: simpy.Environment,
        gpu_resource: simpy.Resource,
        config: SimulationConfig,
        decode_scheduler: "DecodeScheduler",
        metrics: MetricsCollector,
    ) -> None:
        super().__init__(env, gpu_resource, config, metrics, name="Prefill")
        self.decode_scheduler = decode_scheduler
        self.queue: Deque[Request] = deque()

    def add_request(self, request: Request) -> None:
        """Add request for prefill processing."""
        self.queue.append(request)
        self.metrics.increment(MetricKeys.PREFILL_SUBMITTED)
        self.logger.debug(f"Queued {request} at {self.env.now:.2f}")

    def _run(self) -> Generator[simpy.Event, None, None]:
        """Process prefill requests in batches up to chunk budget."""
        while True:
            if not self.queue:
                yield self.env.timeout(self.config.scheduler_timeout)
                continue

            batch, total_tokens = self._collect_batch()
            if total_tokens == 0:
                yield self.env.timeout(self.config.scheduler_timeout)
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
        with self.gpu_resource.request() as gpu_req:
            yield gpu_req
            yield self.env.timeout(delay)

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


class DecodeScheduler(BaseScheduler):
    """Handles decode processing of requests in batches."""

    def __init__(
        self,
        env: simpy.Environment,
        gpu_resource: simpy.Resource,
        config: SimulationConfig,
        metrics: MetricsCollector,
    ) -> None:
        super().__init__(env, gpu_resource, config, metrics, name="Decode")
        self.queue: Deque[Request] = deque()
        self.running: list[Request] = []

    def add_request(self, request: Request) -> None:
        self.queue.append(request)
        self.metrics.increment(MetricKeys.DECODE_SUBMITTED)
        self.logger.debug(f"Queued {request} at {self.env.now:.2f}")

    def _run(self) -> Generator[simpy.Event, None, None]:
        """Continuous‑batch decode loop."""
        while True:
            while self.queue and len(self.running) < self.config.batch_size:
                self.running.append(self.queue.popleft())

            if not self.running:
                self.logger.debug(f"[{self.env.now:.2f}] Decode idle")
                yield self.env.timeout(self.config.scheduler_timeout)
                continue

            with self.gpu_resource.request() as gpu_req:
                yield gpu_req
                step_tokens = len(self.running)
                step_time = self.config.decode_time(step_tokens)
                self.logger.info(
                    f"[{self.env.now:.2f}] Decode step "
                    f"batch={self.running} size={len(self.running)}"
                )
                yield self.env.timeout(step_time)

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


# ---- Simulation ----
class Simulation:
    """Main simulation orchestrator."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        random.seed(config.seed)
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
        )
        self.env = simpy.Environment()
        self.gpu = simpy.Resource(self.env, capacity=1)
        self.metrics = MetricsCollector()
        self.decoder = DecodeScheduler(self.env, self.gpu, self.config, self.metrics)
        self.prefill = PrefillScheduler(
            self.env, self.gpu, self.config, self.decoder, self.metrics
        )
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
            self.prefill.add_request(
                Request(req_id, self.env.now, input_tokens, output_tokens)
            )

    def run(self) -> Dict[str, float]:
        """Run the simulation and return results."""
        self.env.run(until=self.config.simulation_time)
        return self.metrics.get_results()


# ---- Entry Point ----
def run_simulation(config: Optional[SimulationConfig] = None) -> Dict[str, float]:
    """Run simulation with optional configuration."""
    return Simulation(config or SimulationConfig()).run()


if __name__ == "__main__":
    results = run_simulation()
    print("Results:", results)
