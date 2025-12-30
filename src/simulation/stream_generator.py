"""CLI Entry Point for Dynamic Noise Streaming.

Executes a memory lifetime experiment by stitching Stim circuits together 
based on drift parameters.
"""

from absl import app
from absl import flags
from absl import logging
import time

from src.simulation import stim_integration
from src.drift import orchestrator

FLAGS = flags.FLAGS
flags.DEFINE_float("total_duration_ms", 100.0, "Total simulation time in milliseconds.")
flags.DEFINE_string("output_format", "json", "Format for telemetry logs.")

def main(argv):
    del argv
    logging.info("Initializing Quantum Drift Control Stack...")
    
    # 1. Initialize Physics Engine
    drift_engine = orchestrator.DriftOrchestrator()
    
    # 2. Initialize Simulator
    streamer = stim_integration.StimStreamer(drift_model=drift_engine)
    
    # 3. Run Execution Loop
    start_time = time.time()
    final_error_rate = streamer.run_streaming_simulation(FLAGS.total_duration_ms)
    elapsed = time.time() - start_time
    
    logging.info("Experiment Complete.")
    logging.info("Wall Time: %.2fs | Logical Error Rate: %.4e", elapsed, final_error_rate)

if __name__ == "__main__":
    app.run(main)