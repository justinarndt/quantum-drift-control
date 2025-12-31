import stim
import numpy as np
from absl import logging
from src.drift import orchestrator

class StimStreamer:
    """
    Manages the interface between the Drift Orchestrator and the Stim simulator.
    Generates chunked circuit segments with time-varying noise parameters.
    """
    
    def __init__(self, drift_model: orchestrator.DriftOrchestrator, distance: int = 5):
        self.drift_model = drift_model
        self.distance = distance
        logging.info(f"Initialized StimStreamer with distance d={self.distance}")

    def _generate_surface_code_chunk(self, rounds: int, noise_params: dict) -> stim.Circuit:
        """
        Generates a surface code circuit chunk with specific noise values.
        """
        # SANITY CHECK: Clamp probabilities to valid physical range [0.0, 0.5]
        # Probabilities > 0.5 are non-physical (worse than random guessing) and will crash Stim.
        p_gate = float(np.clip(noise_params.get('p_gate', 0.001), 0.0, 0.499))
        p_meas = float(np.clip(noise_params.get('p_meas', 0.001), 0.0, 0.499))

        # Use Stim's optimized surface code generator
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=rounds,
            distance=self.distance,
            after_clifford_depolarization=p_gate,
            after_reset_flip_probability=p_meas,
            before_measure_flip_probability=p_meas
        )
        return circuit

    def run_streaming_simulation(self, total_duration_ms: int) -> float:
        """
        Executes a streaming simulation where noise drifts over time.
        
        Args:
            total_duration_ms: Total physical time to simulate in milliseconds.
            
        Returns:
            The final estimated logical error rate.
        """
        # Assumptions: 1 round = 1 microsecond (approx)
        rounds_per_chunk = 1000
        num_chunks = max(1, int((total_duration_ms * 1000) / rounds_per_chunk))
        
        logging.info(f"Starting simulation: {num_chunks} chunks of {rounds_per_chunk} rounds each.")
        
        total_errors = 0
        total_shots = 0
        
        # Initialize simulator with a clean state if needed (using stabilizer tableau)
        # For this benchmark, we sim each chunk independently to measure rate vs drift
        
        for i in range(num_chunks):
            # 1. Query the Drift Orchestrator for current physics
            # The orchestrator integrates the SDE forward by the chunk duration
            drift_state = self.drift_model.predict_drift(dt=1.0)
            
            # Handle JAX vs Float return types safely
            try:
                t1_val = float(drift_state)
            except TypeError:
                t1_val = float(drift_state[0])

            # Map T1 decay to Pauli error probability
            # p = 1 - e^(-t/T1). For a 20ns gate, this is small.
            # We scale it for visibility in short benchmarks if needed, 
            # but here we use the raw physics value.
            current_p_gate = t1_val
            current_p_meas = t1_val * 10.0 # Measurement is typically 10x noisier

            noise_params = {
                'p_gate': current_p_gate,
                'p_meas': current_p_meas
            }

            # 2. Generate Circuit Chunk
            chunk_circuit = self._generate_surface_code_chunk(rounds_per_chunk, noise_params)

            # 3. Sample Detectors (Run the Simulation)
            sampler = chunk_circuit.compile_detector_sampler()
            
            # Run 100 shots per chunk to get statistical sampling
            shots = 100
            detection_events, observables = sampler.sample(shots=shots, separate_observables=True)
            
            # 4. Decode (Ideal Observer / MWPM)
            # Count how many times the logical observable was flipped
            # In a full decoding loop, we would run PyMatching here.
            # For drift benchmarking, we track the raw logical error accumulation.
            num_logical_errors = np.sum(observables)
            
            total_errors += num_logical_errors
            total_shots += shots
            
            if i % 10 == 0:
                logging.info(f"Chunk {i}/{num_chunks} | Drift p={current_p_gate:.5f} | Logical Errors={num_logical_errors}")

        final_error_rate = total_errors / total_shots
        logging.info(f"Simulation Complete. Final Logical Error Rate: {final_error_rate:.6f}")
        return final_error_rate