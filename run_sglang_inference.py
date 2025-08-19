import asyncio
import aiohttp
import time
import random
import json
import sys
import traceback
from typing import Optional
from tqdm import tqdm
from data_loading_script import RandomDataLoader
import requests

async def send_request(session, req_data, url, req_id):
    prompt = [random.randint(1, 1000) for _ in range(req_data.input_tokens)]
    payload = {
       "input_ids": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": req_data.output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
        "lora_path": None,
        "return_logprob": False,
        "logprob_start_len": -1,
    }
    
    st = time.perf_counter()
    ttft = 0.0
    most_recent_timestamp = st
    last_output_len = 0
    generated_text = ""
    output_len = 0
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                return {
                    "req_id": req_id,
                    "input_tokens": req_data.input_tokens,
                    "output_tokens": req_data.output_tokens,
                    "latency": time.perf_counter() - st,
                    "success": False,
                    "status_code": response.status
                }
            
            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                    
                chunk = chunk_bytes.decode("utf-8")
                if chunk.startswith("data: "):
                    chunk = chunk[6:]
                    
                if chunk == "[DONE]":
                    continue
                    
                try:
                    data = json.loads(chunk)
                    if "text" in data and data["text"]:
                        timestamp = time.perf_counter()
                        generated_text = data["text"]
                        output_len = data["meta_info"]["completion_tokens"]
                        
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                        
                        most_recent_timestamp = timestamp
                        last_output_len = output_len
                except json.JSONDecodeError:
                    continue
            
            latency = time.perf_counter() - st
            tpot = (latency - ttft) / max(output_len - 1, 1) if ttft > 0 and output_len > 1 else 0
            
            return {
                "req_id": req_id,
                "input_tokens": req_data.input_tokens,
                "output_tokens": req_data.output_tokens,
                "latency": latency,
                "ttft": ttft or latency,
                "tpot": tpot,
                "success": True,
                "tokens_generated": output_len,
                "generated_text": generated_text
            }
            
    except Exception as e:
        return {
            "req_id": req_id,
            "input_tokens": req_data.input_tokens,
            "output_tokens": req_data.output_tokens,
            "latency": time.perf_counter() - st,
            "success": False,
            "error": str(e)
        }

async def run_inference():
    random.seed(42)
    dataloader = RandomDataLoader(
        min_tokens=128,
        max_tokens=128,
        min_output_tokens=128,
        max_output_tokens=128,
        max_requests=2,
        arrival_rate=1,
    )
    
    requests_data = dataloader.get_requests()
    inter_arrival_times = dataloader.get_inter_arrival_times()
    base_url = "http://0.0.0.0:30000"
    url = f"{base_url}/generate"

    r = requests.post(f"{base_url}/flush_cache")

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, (req_data, delay) in enumerate(zip(requests_data, inter_arrival_times)):
            await asyncio.sleep(delay)
            task = asyncio.create_task(send_request(session, req_data, url, i))
            tasks.append(task)
        metrics = await asyncio.gather(*tasks)
    
    for m in metrics:
        if m["success"]:
            print(f"Request {m['req_id']}: {m['input_tokens']}→{m['output_tokens']} tokens, "
                  f"latency: {m['latency']:.2f}s, ttft: {m['ttft'] * 1e3:.2f}ms, tpot: {m['tpot'] * 1e3:.3f}ms")
        else:
            print(f"Request {m['req_id']} failed: {m.get('error', 'unknown')}")
    
    successful = [m for m in metrics if m["success"]]
    if successful:
        total_latency = sum(m["latency"] for m in successful)
        avg_latency = sum(m["latency"] for m in successful) / len(successful)
        avg_ttft = sum(m["ttft"] for m in successful) / len(successful)
        avg_tpot = sum(m["tpot"] for m in successful) / len(successful)
        success_rate = len(successful) / len(metrics)
        
        print(f"\nResults: {len(metrics)} requests, success rate: {success_rate:.2%}")
        print(f"Avg latency: {avg_latency:.3f}s, TTFT: {avg_ttft * 1e3:.3f}ms, TPOT: {avg_tpot * 1e3:.3f}ms")
        print(f"Total latency for successful requests: {total_latency:.2f}s")

if __name__ == "__main__":
    asyncio.run(run_inference())
    # python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  --host 0.0.0.