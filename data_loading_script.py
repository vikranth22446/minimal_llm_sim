import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional

import pandas as pd


@dataclass
class RequestData:
    input_tokens: int
    output_tokens: int
    arrival_time: Optional[float] = None


class BaseDataLoader(ABC):
    @abstractmethod
    def get_requests(self) -> List[RequestData]:
        pass

    @abstractmethod
    def get_inter_arrival_times(self) -> List[float]:
        pass


class DataLoader(BaseDataLoader):
    def __init__(self, data_file: str, max_requests=1000, rps_scale=1.0, max_rps=60):
        """Initialize data loader with CSV file path."""
        self.data_file = data_file
        self.df = None
        self.max_requests = max_requests
        self.rps_scale = rps_scale
        self.max_rps = max_rps
        self._load_data()

    def _load_data(self) -> None:
        """Load and validate CSV data."""
        try:
            self.df = pd.read_csv(self.data_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        self._validate_columns()
        self._process_timestamps()

    def _validate_columns(self) -> None:
        """Validate required columns exist in CSV."""
        required_cols = ["num_prefill_tokens", "num_decode_tokens"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _process_timestamps(self) -> None:
        """Process arrival timestamps if available."""
        if "arrived_at" in self.df.columns:
            valid_mask = self.df["arrived_at"] > 300
            self.df = self.df[valid_mask].copy()
            self.df["arrival_time"] = self.df["arrived_at"].astype(float)
            self.df["inter_arrival"] = self.df["arrival_time"].diff().fillna(0.0)
        else:
            self.df["arrival_time"] = None
            self.df["inter_arrival"] = None

    def get_requests(self) -> List[RequestData]:
        """Get all requests as a list of RequestData objects."""
        max_requests = self.max_requests
        df_subset = self.df.head(max_requests) if max_requests else self.df

        requests = []
        for _, row in df_subset.iterrows():
            request = RequestData(
                input_tokens=int(row["num_prefill_tokens"]),
                output_tokens=int(row["num_decode_tokens"]),
                arrival_time=row.get("arrival_time"),
            )
            requests.append(request)

        return requests

    def get_inter_arrival_times(self) -> List[float]:
        """Get inter-arrival times for request generation."""
        max_rps = self.max_rps
        rps_scale = self.rps_scale
        if (
            "inter_arrival" in self.df.columns
            and self.df["inter_arrival"].notna().any()
        ):
            times = self.df["inter_arrival"].fillna(0.0).tolist()
            if max_rps is not None and max_rps > 0:
                min_interval = 1.0 / max_rps
                times = [min(t, min_interval) if t > 0 else t for t in times]
            times = [t * rps_scale for t in times]
        else:
            num_requests = len(self.df)
            if max_rps is not None and max_rps > 0:
                base_interval = 1.0 / max_rps
            else:
                base_interval = 1.0
            times = [base_interval * rps_scale] * num_requests
        return times

    def get_statistics(self) -> Dict[str, float]:
        """Get basic statistics about the loaded data."""
        return {
            "total_requests": len(self.df),
            "avg_input_tokens": self.df["num_prefill_tokens"].mean(),
            "avg_output_tokens": self.df["num_decode_tokens"].mean(),
            "max_input_tokens": self.df["num_prefill_tokens"].max(),
            "max_output_tokens": self.df["num_decode_tokens"].max(),
            "min_input_tokens": self.df["num_prefill_tokens"].min(),
            "min_output_tokens": self.df["num_decode_tokens"].min(),
        }

    def generate_requests(self) -> Generator[RequestData, None, None]:
        """Generator that yields requests with proper timing."""
        requests = self.get_requests()
        rps_scale = self.rps_scale
        max_rps = self.max_rps
        inter_arrival_times = self.get_inter_arrival_times(rps_scale, max_rps)

        for request, delay in zip(requests, inter_arrival_times):
            yield request, delay


class RandomDataLoader(BaseDataLoader):
    def __init__(
        self,
        max_requests=1000,
        min_tokens: int = 2,
        max_tokens: int = 6,
        max_output_tokens: int = 64,
        arrival_rate: float = 0.8,
        inter_arrival_fn: Callable[[float], float] = random.expovariate,
    ):
        self.min_tokens = int(min_tokens)
        self.max_tokens = int(max_tokens)
        self.max_output_tokens = int(max_output_tokens)
        self.arrival_rate = arrival_rate
        self.inter_arrival_fn = inter_arrival_fn
        self.max_requests = max_requests

    def get_requests(self) -> List[RequestData]:
        return [
            RequestData(
                input_tokens=random.randint(self.min_tokens, self.max_tokens),
                output_tokens=random.randint(self.min_tokens, self.max_output_tokens),
            )
            for _ in range(self.max_requests)
        ]

    def get_inter_arrival_times(self) -> List[float]:
        return [self.inter_arrival_fn(self.arrival_rate) for _ in range(self.max_requests)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and analyze request data from CSV"
    )
    parser.add_argument(
        "--data_file", type=str, required=True, help="Path to CSV data file"
    )
    parser.add_argument(
        "--max_requests", type=int, help="Maximum number of requests to load"
    )
    parser.add_argument("--stats", action="store_true", help="Show data statistics")

    args = parser.parse_args()

    try:
        loader = DataLoader(args.data_file)

        if args.stats:
            stats = loader.get_statistics()
            print("Data Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value:.2f}")

        requests = loader.get_requests(args.max_requests)
        print(f"\nLoaded {len(requests)} requests")

        if requests:
            print(
                f"Sample request: input_tokens={requests[0].input_tokens}, output_tokens={requests[0].output_tokens}"
            )

    except Exception as e:
        print(f"Error: {e}")
