import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Generator, List, Tuple
import simpy
import time
from data_loading_script import BaseDataLoader, RandomDataLoader
from model_executor_helper import ModelExecutorHelper
from pprint import pprint
from heapq import heappop, heappush, heapify

class MetricKeys(Enum):
    PREFILL_SUBMITTED = "prefill_submitted"
    PREFILL_COMPLETED = "prefill_completed"
    DECODE_SUBMITTED = "decode_submitted"
    COMPLETED = "completed"
    TOTAL_LATENCY = "total_latency"
    AVERAGE_LATENCY = "average_latency"
    TOTAL_TTFT = "total_ttft"
    AVERAGE_TTFT = "average_ttft"
    TOTAL_TPOT = "total_tpot"
    AVERAGE_TPOT = "average_tpot"
    MAKESPAN_KEY = "makespan"

class SchedulerPolicy(Enum):
    ROUND_ROBIN = auto()
    LJF = auto()

@dataclass
class Request:
    """Represents a processing request with token information."""

    id: int
    arrival_time: float
    target_input_tokens: int
    target_output_tokens: int
    prefill_generated: int = field(default=0, init=False)
    output_generated: int = field(default=0, init=False)
    first_token_time: float = field(default=0.0, init=False)
    completion_time: float = field(default=0.0, init=False)

    @property
    def prefill_remaining(self) -> int:
        return self.target_input_tokens - self.prefill_generated
    
    @property
    def remaining_output_tokens(self) -> int:
        return self.target_output_tokens - self.output_generated
    
    @property
    def sequence_length(self) -> int:
        return self.prefill_generated + self.output_generated

    def __repr__(self) -> str:
        return f"Request({self.id}, prefill_gen={self.prefill_generated}, decode_gen={self.output_generated}, seq_len={self.sequence_length}, expected=({self.target_input_tokens}, {self.target_output_tokens}))"


@dataclass
class _BatchItem:
    req: Request
    tokens_for_req: int


@dataclass
class SimulationConfig:
    model_execution_helper: ModelExecutorHelper
    data_loader: BaseDataLoader
    simulation_time: float = 60.0
    max_batch_size: int = 32
    prefill_chunk_size: int = 8192
    num_gpus: int = 1
    enable_tracing: bool = False
    trace_output_file: str = "simulation_trace.json"
    decode_log_interval: int = 40
    batch_mode: bool = False
    scheduler_policy: SchedulerPolicy = SchedulerPolicy.ROUND_ROBIN

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.simulation_time <= 0:
            raise ValueError("simulation_time must be positive")
        if self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if self.num_gpus <= 0:
            raise ValueError("num_gpus must be positive")

    def decode_time(self, batch: List[_BatchItem]) -> float:
        """Calculate decode time for given number of tokens."""
        seq_lens = [batch_item.req.sequence_length for batch_item in batch]
        return self.model_execution_helper.decode(seq_lens)

    def prefill_time(self, batch: List[_BatchItem]) -> float:
        cached_context_lens = [batch_item.req.sequence_length for batch_item in batch]
        seq_lens = [batch_item.tokens_for_req for batch_item in batch]
        return self.model_execution_helper.prefill(
            seq_lens, cached_context_lens=cached_context_lens
        )


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
            key.value: 0.0 for key in MetricKeys if key not in {MetricKeys.AVERAGE_LATENCY, MetricKeys.AVERAGE_TTFT, MetricKeys.AVERAGE_TPOT}
        }
        self.metrics[MetricKeys.MAKESPAN_KEY.value] = 0.0

    def increment(self, key: MetricKeys, value: float = 1.0) -> None:
        """Increment a metric by the given value."""
        self.metrics[key.value] += value

    def get_results(self) -> Dict[str, float]:
        """Calculate and return final metrics including averages."""
        completed = self.metrics[MetricKeys.COMPLETED.value]
        avg_latency = (
            self.metrics[MetricKeys.TOTAL_LATENCY.value] / completed
            if completed > 0
            else 0.0
        )
        avg_ttft = (
            self.metrics[MetricKeys.TOTAL_TTFT.value] / completed
            if completed > 0
            else 0.0
        )
        avg_tpot = (
            self.metrics[MetricKeys.TOTAL_TPOT.value] / completed
            if completed > 0
            else 0.0
        )
        return {
            **self.metrics, 
            MetricKeys.AVERAGE_LATENCY.value: avg_latency,
            MetricKeys.AVERAGE_TTFT.value: avg_ttft,
            MetricKeys.AVERAGE_TPOT.value: avg_tpot
        }


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

    def _gpu_execute(self, name: str, duration: float, priority: int, **trace_args):
        with self.gpu_resource.request(priority=priority) as gpu_req:
            yield gpu_req
            self.tracer.event(
                name, self.env.now, phase="B", gpu_id=self.gpu_id, **trace_args
            )
            yield self.env.timeout(duration)
            self.tracer.event(name, self.env.now, phase="E", gpu_id=self.gpu_id)

    @abstractmethod
    def _run(self) -> Generator:
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
        self.inflight = False

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
                self.inflight = True
                batch, total_tokens = yield from self._collect_batch(first_request)

                yield from self._execute_prefill_step(batch, total_tokens)
                self._finish(batch)
                self.inflight = bool(self.queue.items)

    def _collect_batch(
        self, first_request: Request
    ) -> Generator[simpy.Event, None, Tuple[List[_BatchItem], int]]:
        batch, used, limit = [], 0, self.config.prefill_chunk_size

        assert first_request.prefill_remaining > 0, (
            f"Request {first_request.id} has 0 prefill tokens, would cause infinite loop"
        )
        tokens = min(limit - used, first_request.prefill_remaining)
        batch.append(_BatchItem(first_request, tokens))
        used += tokens

        while used < limit:
            try:
                queue_event = self.queue.get()
                timeout_event = self.env.timeout(0)
                result = yield (queue_event | timeout_event)

                if result and queue_event in result:
                    additional_req = result[queue_event]
                    assert additional_req.prefill_remaining > 0, (
                        f"Request {additional_req.id} has 0 prefill tokens, would cause infinite loop"
                    )
                    tokens = min(limit - used, additional_req.prefill_remaining)
                    batch.append(_BatchItem(additional_req, tokens))
                    used += tokens
                else:
                    queue_event.cancel() 
                    break
            except simpy.Interrupt:
                break
            except Exception as e:
                self.logger.error(f"Batching error in prefill: {e}")
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
            f"GPU{self.gpu_id} [{self.env.now:.2f}] Prefill batch. #new-seq: {len(batch)} #new-token: {total_tokens} tokens "
            f"f({delay:.2f}s)"
        )
        yield from self._gpu_execute(
            "Prefill Batch",
            delay,
            priority=0,
            tokens=total_tokens,
            requests=len(batch),
        )

    def _finish(self, batch: List[_BatchItem]) -> None:
        """Finish processing the batch and handle completions."""
        for item in batch:
            item.req.prefill_generated += item.tokens_for_req
            if item.req.prefill_remaining > 0:
                self.queue.put(item.req)
            else:
                # After prefill completion, generate the first output token
                item.req.output_generated += 1
                item.req.first_token_time = self.env.now
                self.metrics.increment(MetricKeys.PREFILL_COMPLETED)
                self.logger.debug(
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
        self.running: List[Request] = []
        self.prefill_ref = None
        self.forward_ct_decode = 0

    def add_request(self, request: Request) -> None:
        self.queue.put(request)
        self.metrics.increment(MetricKeys.DECODE_SUBMITTED)
        self.logger.debug(f"GPU{self.gpu_id}: Queued {request} at {self.env.now:.2f}")

    def set_prefill_ref(self, prefill: "PrefillScheduler"):
        self.prefill_ref = prefill

    def _prefill_ready(self) -> bool:
        return bool(
            self.prefill_ref
            and (self.prefill_ref.queue.items or self.prefill_ref.inflight)
        )

    def _run(self) -> Generator[simpy.Event, None, None]:
        while True:
            if not self.running:
                first_request = yield self.queue.get()
                if isinstance(first_request, Request):
                    assert first_request.remaining_output_tokens > 0, (
                        f"Request {first_request.id} has 0 decode tokens, should not be in decode scheduler"
                    )
                    self.running.append(first_request)

            while len(self.running) < self.config.max_batch_size:
                try:
                    queue_event = self.queue.get()
                    timeout_event = self.env.timeout(0)
                    result = yield (queue_event | timeout_event)

                    if result and queue_event in result:
                        additional_req = result[queue_event]
                        assert additional_req.remaining_output_tokens > 0, (
                            f"Request {additional_req.id} has 0 decode tokens, should not be in decode scheduler"
                        )
                        self.running.append(additional_req)
                    else:
                        queue_event.cancel()
                        break
                except simpy.Interrupt:
                    break
                except Exception as e:
                    self.logger.error(f"Batching error in decode: {e}")
                    break

            yield from self._execute_decode_step()
            self._finish_decode_step()

    def _execute_decode_step(self) -> Generator[simpy.Event, None, None]:
        """Execute a single decode step for the current batch."""
        batch_items = [_BatchItem(req, 1) for req in self.running]
        step_time = self.config.decode_time(batch_items)
        yield from self._gpu_execute(
            "Decode Step",
            step_time,
            priority=1,
            batch_size=len(self.running),
        )

    def _finish_decode_step(self) -> None:
        """Process decode step results and handle completions."""
        still_running: List[Request] = []
        for req in self.running:
            req.output_generated += 1
            if req.remaining_output_tokens > 0:
                still_running.append(req)
            else:
                req.completion_time = self.env.now
                latency = self.env.now - req.arrival_time
                ttft = req.first_token_time - req.arrival_time
                tpot = (req.completion_time - req.first_token_time) / max(1, req.target_output_tokens - 1)
                
                self.metrics.increment(MetricKeys.COMPLETED)
                self.metrics.increment(MetricKeys.TOTAL_LATENCY, latency)
                self.metrics.increment(MetricKeys.TOTAL_TTFT, ttft)
                self.metrics.increment(MetricKeys.TOTAL_TPOT, tpot)
                self.metrics.metrics[MetricKeys.MAKESPAN_KEY.value] = max(
                    self.metrics.metrics[MetricKeys.MAKESPAN_KEY.value], req.completion_time
                )
                self.logger.info(
                    f"GPU{self.gpu_id} [{self.env.now:.2f}] Finished {req}, "
                    f"latency={latency:.2f}, ttft={ttft:.2f}, tpot={tpot:.2f}"
                )
        self.running = still_running
        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        if self.forward_ct_decode % self.config.decode_log_interval == 0:
            total_tokens = sum(req.sequence_length for req in self.running)
            self.logger.info(
                f"GPU{self.gpu_id} [{self.env.now:.2f}] Decode batch. "
                f"#running-req: {len(self.running)}, "
                f"#token: {total_tokens}"
            )


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

        self.gpus = [simpy.PriorityResource(env, capacity=1) for _ in range(config.num_gpus)]
        self.decode_schedulers = []
        self.prefill_schedulers = []
        for gpu_id, gpu in enumerate(self.gpus):
            decoder = DecodeScheduler(env, gpu, config, metrics, tracer, gpu_id)
            prefill = PrefillScheduler(
                env, gpu, config, decoder, metrics, tracer, gpu_id
            )
            self.prefill_schedulers.append(prefill)
            self.decode_schedulers.append(decoder)
            decoder.set_prefill_ref(prefill)

    def add_request(self, request: Request) -> None:
        """Add request to next GPU in round-robin fashion."""
        gpu_id = self.current_gpu
        self.prefill_schedulers[gpu_id].add_request(request)
        self.current_gpu = (self.current_gpu + 1) % self.config.num_gpus
        logging.getLogger("RoundRobin").debug(f"Assigned {request} to GPU{gpu_id} at {self.env.now}")

class LJFRoundRobinScheduler(RoundRobinScheduler):
    def __init__(self, env, config, metrics, tracer):
        super().__init__(env, config, metrics, tracer)
        self._buffer = []
        assert config.batch_mode, "LJF scheduler requires batch mode to be enabled"

    @staticmethod
    def _job_size(req: Request) -> int:
        return req.target_input_tokens + req.target_output_tokens

    def add_request(self, request: Request) -> None:
        self._buffer.append(request) 

    def flush(self) -> None:
        self._buffer.sort(key=self._job_size, reverse=True)
        gpu_loads = [(0.0, i) for i in range(self.config.num_gpus)]
        requests_per_gpu = {}
        for req in self._buffer:
            gpu_load, gpu_id = heappop(gpu_loads)
            gpu_load += self._job_size(req)
            heappush(gpu_loads, (gpu_load, gpu_id))
            requests_per_gpu[gpu_id] = requests_per_gpu.get(gpu_id, []) + [req]
        
        for gpu_id in requests_per_gpu.keys():
             requests_per_gpu[gpu_id].sort(key=self._job_size, reverse=True)

        for gpu_id, reqs in requests_per_gpu.items():
            for req in reqs:
                self.prefill_schedulers[gpu_id].add_request(req)
        self._buffer.clear()

class Simulation:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.env = simpy.Environment()
        self.metrics = MetricsCollector()
        self.tracer = SimpleTracer(config.enable_tracing)
        if config.scheduler_policy == SchedulerPolicy.LJF:
            self.scheduler = LJFRoundRobinScheduler(
                self.env, config, self.metrics, self.tracer
            )
        else:
            self.scheduler = RoundRobinScheduler(
                self.env, config, self.metrics, self.tracer
            )
        self.env.process(self._generate_requests())
        data_loader = self.config.data_loader
        assert data_loader is not None
        self.requests_data = data_loader.get_requests()
        self.inter_arrival_times = data_loader.get_inter_arrival_times()
        assert len(self.requests_data) == len(self.inter_arrival_times), (
            f"Length mismatch: requests_data ({len(self.requests_data)}) != inter_arrival_times ({len(self.inter_arrival_times)})"
        )

    def _generate_requests(self) -> Generator[simpy.Event, None, None]:
        req_id = 0
        for request_data, delay in zip(self.requests_data, self.inter_arrival_times):
            if not self.config.batch_mode:
                yield self.env.timeout(delay)
            req_id += 1
            self.scheduler.add_request(
                Request(
                    id=req_id,
                    arrival_time=self.env.now,
                    target_input_tokens=request_data.input_tokens,
                    target_output_tokens=request_data.output_tokens,
                )
            )
        if self.config.scheduler_policy == SchedulerPolicy.LJF:
            self.scheduler.flush()

    def run(self) -> Dict[str, float]:
        """Run the simulation and return results."""
        self.env.run(until=self.config.simulation_time)
        self.tracer.save(self.config.trace_output_file)
        return self.metrics.get_results()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s: %(message)s",
        filename="simulation.log",
        filemode="w"
    )
    folder_name = "profile_output_a100"
    model_name = "Qwen_Qwen3_8B_TP_1"
    model_execution_helper = ModelExecutorHelper(
        onnx_model_decode=ModelExecutorHelper.get_onnx_model_decode_from_folder(
            folder_name, model_name
        ),
        onnx_model_prefill=ModelExecutorHelper.get_onnx_model_prefill_from_folder(
            folder_name, model_name
        ),
    )

    # logging.disable(logging.CRITICAL)
    random.seed(42)
    dataloader = RandomDataLoader(
        min_tokens=800,
        max_tokens=2000,
        min_output_tokens=128,
        max_output_tokens=16000,
        max_requests=1024,
        arrival_rate=20,
    )
    config = SimulationConfig(
        model_execution_helper=model_execution_helper, 
        data_loader=dataloader,
        prefill_chunk_size=8192,
        batch_mode=True,
        scheduler_policy=SchedulerPolicy.ROUND_ROBIN
    )
    config.enable_tracing = True
    config.num_gpus = 8
    config.simulation_time = 1200.0
    sim = Simulation(config)

    # print(f"Running scheduler policy", config.scheduler_policy.name)
    start_time = time.perf_counter()
    results = sim.run()
    end_time = time.perf_counter()
    rounded_results = {k: round(v, 6) if isinstance(v, float) else v for k, v in results.items()}
    rounded_results["average_ttft"] *= 1e3
    rounded_results["average_tpot"] *= 1e3
    pprint(rounded_results)
    print(f"Simulation took {end_time - start_time:.2f} seconds")