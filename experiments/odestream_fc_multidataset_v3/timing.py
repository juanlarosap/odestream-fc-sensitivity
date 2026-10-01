"""The frozen per-observation compute boundary (CPU wall seconds)."""
import time
from .datasets import batch


def timed_observation(step,model,gate,stream,index,updates):
    start=time.perf_counter()
    row=step(model,gate,batch(stream,index),index,updates)
    elapsed=time.perf_counter()-start
    return row,elapsed
