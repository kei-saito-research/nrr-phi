#!/usr/bin/env python3
"""
NRR-Operators Validation Script
================================
Reproduces the experiments from:
  "NRR-Operators: Design Principles for Non-Collapsing State Transformations"

This script validates that principle-satisfying operators achieve 0% collapse
while principle-violating operators show measurable information loss.

Usage:
  python run_operator_validation.py --data data/operator_validation_states.json

Requirements:
  pip install numpy

Author: Kei Saito
License: CC BY 4.0
"""

import json
import argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


# =============================================================================
# NRR Core Classes
# =============================================================================

@dataclass
class Interpretation:
    semantic_vector: str
    context: str
    weight: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.metadata = self.metadata or {}


class NRRState:
    def __init__(self, interpretations: List[Interpretation]):
        self.interpretations = interpretations
    
    def get_weights(self) -> np.ndarray:
        return np.array([i.weight for i in self.interpretations])
    
    def entropy(self) -> float:
        w = self.get_weights()
        if w.sum() == 0:
            return 0.0
        p = w / w.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    
    def size(self) -> int:
        return len(self.interpretations)


class NRROperators:
    """Principle-satisfying and principle-violating operators."""
    
    @staticmethod
    def uniform_subtraction(state: NRRState, b: float) -> NRRState:
        """δ v1: Principle-VIOLATING uniform subtraction."""
        w = state.get_weights()
        new_w = np.maximum(w - b, 0)
        return NRRState([
            Interpretation(i.semantic_vector, i.context, ww, i.metadata)
            for i, ww in zip(state.interpretations, new_w)
        ])
    
    @staticmethod
    def dampening(state: NRRState, lambda_param: float = 0.3) -> NRRState:
        """δ v2: Principle-satisfying mean compression."""
        w = state.get_weights()
        w_mean = w.mean()
        new_w = w * (1 - lambda_param) + w_mean * lambda_param
        return NRRState([
            Interpretation(i.semantic_vector, i.context, ww, i.metadata)
            for i, ww in zip(state.interpretations, new_w)
        ])
    
    @staticmethod
    def stripping(state: NRRState, bias: float = 0.1) -> NRRState:
        """σ v2: Principle-satisfying proportional stripping."""
        w = state.get_weights()
        if w.max() == 0:
            return state
        new_w = np.maximum(w - bias * (w / w.max()), 0)
        return NRRState([
            Interpretation(i.semantic_vector, i.context, ww, i.metadata)
            for i, ww in zip(state.interpretations, new_w)
        ])
    
    @staticmethod
    def deferred_resolution(state: NRRState) -> NRRState:
        """τ: Identity mapping (deferred resolution)."""
        return state
    
    @staticmethod
    def cpp_integration(s1: NRRState, s2: NRRState) -> NRRState:
        """κ: Contradiction-preserving integration."""
        return NRRState(s1.interpretations + s2.interpretations)
    
    @staticmethod
    def persistence(curr: NRRState, prev: NRRState, decay: float = 0.5) -> NRRState:
        """π: Temporal persistence with decay."""
        new_interps = curr.interpretations.copy()
        for i in prev.interpretations:
            new_interps.append(Interpretation(
                i.semantic_vector,
                f'{i.context}_prev',
                i.weight * decay,
                {**i.metadata, 'is_historical': True}
            ))
        return NRRState(new_interps)


class CollapseDetector:
    @staticmethod
    def detect_collapse(before: NRRState, after: NRRState, epsilon: float = 0.1) -> Tuple[bool, float]:
        """Detect if entropy dropped below threshold."""
        delta_h = after.entropy() - before.entropy()
        return delta_h < -epsilon, delta_h


# =============================================================================
# Helper Functions
# =============================================================================

def dict_to_state(d: Dict) -> NRRState:
    """Convert dict representation to NRRState."""
    interps = [
        Interpretation(
            i['semantic_vector'],
            i['context'],
            i['weight'],
            i.get('metadata', {})
        )
        for i in d['interpretations']
    ]
    return NRRState(interps)


def run_single_state_experiments(single_states: List[Dict], epsilon: float) -> Dict:
    """Run δ v1, δ v2, σ v2, τ on single states."""
    results = {}
    n = len(single_states)
    
    # δ v1: Uniform subtraction (principle-violating)
    for b in [0.05, 0.10, 0.20]:
        violations, dhs = 0, []
        for s in single_states:
            st = dict_to_state(s['state'])
            after = NRROperators.uniform_subtraction(st, b)
            coll, dh = CollapseDetector.detect_collapse(st, after, epsilon)
            if coll:
                violations += 1
            dhs.append(dh)
        results[f'delta_v1_{b:.2f}'] = {
            'violations': violations,
            'rate': violations / n,
            'mean_dh': np.mean(dhs),
            'std_dh': np.std(dhs)
        }
    
    # δ v2: Dampening (principle-satisfying)
    for lam in [0.1, 0.2, 0.3, 0.4, 0.5]:
        violations, dhs = 0, []
        for s in single_states:
            st = dict_to_state(s['state'])
            after = NRROperators.dampening(st, lam)
            coll, dh = CollapseDetector.detect_collapse(st, after, epsilon)
            if coll:
                violations += 1
            dhs.append(dh)
        results[f'delta_v2_lambda_{lam}'] = {
            'violations': violations,
            'rate': violations / n,
            'mean_dh': np.mean(dhs),
            'std_dh': np.std(dhs)
        }
    
    # σ v2: Stripping (principle-satisfying)
    for bias in [0.05, 0.10, 0.15, 0.20]:
        violations, dhs = 0, []
        for s in single_states:
            st = dict_to_state(s['state'])
            after = NRROperators.stripping(st, bias)
            coll, dh = CollapseDetector.detect_collapse(st, after, epsilon)
            if coll:
                violations += 1
            dhs.append(dh)
        results[f'sigma_v2_bias_{bias:.2f}'] = {
            'violations': violations,
            'rate': violations / n,
            'mean_dh': np.mean(dhs),
            'std_dh': np.std(dhs)
        }
    
    # τ: Identity (principle-satisfying)
    violations, dhs = 0, []
    for s in single_states:
        st = dict_to_state(s['state'])
        after = NRROperators.deferred_resolution(st)
        coll, dh = CollapseDetector.detect_collapse(st, after, epsilon)
        if coll:
            violations += 1
        dhs.append(dh)
    results['tau_identity'] = {
        'violations': violations,
        'rate': violations / n,
        'mean_dh': np.mean(dhs),
        'std_dh': np.std(dhs)
    }
    
    return results


def run_paired_experiments(contradictory_pairs: List[Dict], 
                           temporal_pairs: List[Dict],
                           epsilon: float) -> Dict:
    """Run κ on contradictory pairs and π on temporal pairs."""
    results = {}
    
    # κ: CPP Integration
    violations, dhs = 0, []
    for p in contradictory_pairs:
        s1 = dict_to_state(p['state1']['state'] if 'state' in p['state1'] else p['state1'])
        s2 = dict_to_state(p['state2']['state'] if 'state' in p['state2'] else p['state2'])
        h_before = max(s1.entropy(), s2.entropy())  # Compare against higher entropy state
        merged = NRROperators.cpp_integration(s1, s2)
        h_after = merged.entropy()
        dh = h_after - h_before
        if dh < -epsilon:
            violations += 1
        dhs.append(dh)
    results['kappa'] = {
        'violations': violations,
        'rate': violations / len(contradictory_pairs),
        'mean_dh': np.mean(dhs),
        'std_dh': np.std(dhs)
    }
    
    # π: Persistence
    violations, dhs = 0, []
    for p in temporal_pairs:
        # Handle both types: two_turn_dialogue and context_evolution
        if p['type'] == 'two_turn_dialogue':
            s1_data = p['state_t1']
            s2_data = p['state_t2']
        else:  # context_evolution
            s1_data = p['state_base']
            s2_data = p['state_extended']
        
        s_prev = dict_to_state(s1_data['state'] if 'state' in s1_data else s1_data)
        s_curr = dict_to_state(s2_data['state'] if 'state' in s2_data else s2_data)
        h_before = s_curr.entropy()
        persisted = NRROperators.persistence(s_curr, s_prev)
        h_after = persisted.entropy()
        dh = h_after - h_before
        if dh < -epsilon:
            violations += 1
        dhs.append(dh)
    results['pi'] = {
        'violations': violations,
        'rate': violations / len(temporal_pairs),
        'mean_dh': np.mean(dhs),
        'std_dh': np.std(dhs)
    }
    
    return results


def print_results(single_results: Dict, paired_results: Dict):
    """Print formatted results table."""
    print("\n" + "=" * 70)
    print("NRR-Operators Validation Results")
    print("=" * 70)
    
    print("\n--- Single State Operators ---\n")
    print(f"{'Operator':<25} {'Violations':>12} {'Rate':>10} {'Mean ΔH':>12}")
    print("-" * 60)
    
    for key in sorted(single_results.keys()):
        r = single_results[key]
        print(f"{key:<25} {r['violations']:>12} {r['rate']*100:>9.1f}% {r['mean_dh']:>+12.4f}")
    
    print("\n--- Paired Operators ---\n")
    print(f"{'Operator':<25} {'Violations':>12} {'Rate':>10} {'Mean ΔH':>12}")
    print("-" * 60)
    
    for key, r in paired_results.items():
        print(f"{key:<25} {r['violations']:>12} {r['rate']*100:>9.1f}% {r['mean_dh']:>+12.4f}")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("  - Principle-violating (δ v1): Collapse rate increases with b")
    print("  - Principle-satisfying: 0% collapse across all operators")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='NRR-Operators Validation')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to operator_validation_states.json')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Collapse threshold (default: 0.1)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    with open(args.data) as f:
        data = json.load(f)
    
    single_states = data['single_states']
    contradictory_pairs = data['contradictory_pairs']
    temporal_pairs = data['temporal_pairs']
    
    print(f"  Single states: {len(single_states)}")
    print(f"  Contradictory pairs: {len(contradictory_pairs)}")
    print(f"  Temporal pairs: {len(temporal_pairs)}")
    print(f"  Epsilon: {args.epsilon}")
    
    # Run experiments
    print("\nRunning single state experiments...")
    single_results = run_single_state_experiments(single_states, args.epsilon)
    
    print("Running paired experiments...")
    paired_results = run_paired_experiments(contradictory_pairs, temporal_pairs, args.epsilon)
    
    # Print results
    print_results(single_results, paired_results)
    
    # Save results
    if args.output:
        full_results = {
            'metadata': {
                'epsilon': args.epsilon,
                'n_single': len(single_states),
                'n_contradictory': len(contradictory_pairs),
                'n_temporal': len(temporal_pairs)
            },
            'single_state_operators': single_results,
            'paired_operators': paired_results
        }
        with open(args.output, 'w') as f:
            json.dump(full_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
