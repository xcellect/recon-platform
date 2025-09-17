"""
Property-Based Tests for Section 3.1 Compact Implementation Compliance

Verifies that CompactReCoNNode strictly follows the arithmetic rules from
section 3.1 of "Request Confirmation Networks for Neuro-Symbolic Script Execution"

Section 3.1 defines f_node functions:
- f_node^gen: z^sur if (z^gen·z^sub=0) ∨ (∃link^por ∧ z^por=0), otherwise z^gen·z^sub  
- f_node^por: 0 if (z^sub≤0) ∨ (∃link^por ∧ z^por≤0), otherwise z^sur+z^gen
- f_node^ret: 1 if z^por<0, otherwise 0
- f_node^sub: 0 if z^gen≠0 ∨ (∃link^por ∧ z^por≤0), otherwise z^sub
- f_node^sur: 0 if (z^sub≤0) ∨ (∃link^por ∧ z^por≤0), 
               (z^sur+z^gen)·z^ret if ∃link^ret, otherwise z^sur+z^gen
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
import torch
from recon_engine.compact import CompactReCoNNode


class TestSection31CompactCompliance:
    """Property-based tests verifying Section 3.1 compact arithmetic rules."""
    
    def _evaluate_paper_equations(self, z_gen, z_por, z_ret, z_sub, z_sur, has_por_link=False, has_ret_link=False):
        """Reference implementation of paper equations for comparison."""
        
        # f_node^gen (equation from paper)
        if (z_gen * z_sub == 0) or (has_por_link and z_por == 0):
            f_gen = z_sur
        else:
            f_gen = z_gen * z_sub
        
        # f_node^por  
        if (z_sub <= 0) or (has_por_link and z_por <= 0):
            f_por = 0
        else:
            f_por = z_sur + z_gen
        
        # f_node^ret
        if z_por < 0:
            f_ret = 1
        else:
            f_ret = 0
        
        # f_node^sub
        if (z_gen != 0) or (has_por_link and z_por <= 0):
            f_sub = 0
        else:
            f_sub = z_sub
        
        # f_node^sur
        if (z_sub <= 0) or (has_por_link and z_por <= 0):
            f_sur = 0
        elif has_ret_link:
            f_sur = (z_sur + z_gen) * z_ret
        else:
            f_sur = z_sur + z_gen
        
        return {
            "gen": f_gen,
            "por": f_por, 
            "ret": f_ret,
            "sub": f_sub,
            "sur": f_sur
        }
    
    @given(
        z_gen=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
        z_por=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
        z_ret=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
        z_sub=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
        z_sur=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
        has_por_link=st.booleans(),
        has_ret_link=st.booleans()
    )
    @settings(max_examples=500)
    def test_compact_arithmetic_rules_property(self, z_gen, z_por, z_ret, z_sub, z_sur, has_por_link, has_ret_link):
        """Property test: CompactReCoNNode should exactly implement paper's f_node functions."""
        node = CompactReCoNNode("compact_test", "script")
        
        # Set link existence flags
        node.set_link_existence(has_por=has_por_link, has_ret=has_ret_link)
        
        # Prepare z values
        z = {
            "gen": z_gen,
            "por": z_por,
            "ret": z_ret, 
            "sub": z_sub,
            "sur": z_sur
        }
        
        # Get implementation result
        actual_gates = node.update_state_compact(z)
        
        # Get paper equation result
        expected_gates = self._evaluate_paper_equations(
            z_gen, z_por, z_ret, z_sub, z_sur, has_por_link, has_ret_link
        )
        
        # Verify each f_node function matches paper exactly
        for gate_type in ["gen", "por", "ret", "sub", "sur"]:
            actual = actual_gates[gate_type]
            expected = expected_gates[gate_type]
            
            # Handle floating point comparison
            if isinstance(actual, torch.Tensor):
                actual = actual.item()
            if isinstance(expected, torch.Tensor):
                expected = expected.item()
            
            assert abs(actual - expected) < 1e-6, \
                f"f_node^{gate_type} mismatch: expected {expected}, got {actual} " \
                f"with z={z}, has_por={has_por_link}, has_ret={has_ret_link}"
    
    @given(
        z_values=st.fixed_dictionaries({
            "gen": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
            "por": st.sampled_from([-1.0, 0.0, 1.0]),  # Paper says por/ret ∈ {-1, 0, 1}
            "ret": st.sampled_from([-1.0, 0.0, 1.0]),
            "sub": st.sampled_from([0.0, 1.0]),  # Paper says sub ∈ {0, 1}
            "sur": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)  # Paper says sur ∈ [0,1] or -1
        })
    )
    @settings(max_examples=200)
    def test_paper_activation_ranges(self, z_values):
        """Property test: Verify behavior with paper-specified activation ranges."""
        node = CompactReCoNNode("range_test", "script")
        
        # Test with both link configurations
        for has_por in [False, True]:
            for has_ret in [False, True]:
                node.set_link_existence(has_por=has_por, has_ret=has_ret)
                
                gates = node.update_state_compact(z_values)
                
                # Verify output constraints from paper
                # por/ret should be in {-1, 0, 1} range
                por_val = gates["por"]
                ret_val = gates["ret"]
                
                if isinstance(por_val, torch.Tensor):
                    por_val = por_val.item()
                if isinstance(ret_val, torch.Tensor):
                    ret_val = ret_val.item()
                
                # ret should be binary (0 or 1) per paper logic
                assert ret_val in [0.0, 1.0], \
                    f"f_node^ret should be binary, got {ret_val}"
                
                # sub should be binary or zero per paper logic  
                sub_val = gates["sub"]
                if isinstance(sub_val, torch.Tensor):
                    sub_val = sub_val.item()
                assert sub_val >= 0, f"f_node^sub should be non-negative, got {sub_val}"
    
    @given(
        step_count=st.integers(min_value=1, max_value=10),
        z_sequence=st.lists(
            st.fixed_dictionaries({
                "gen": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
                "por": st.sampled_from([-1.0, 0.0, 1.0]),
                "ret": st.sampled_from([-1.0, 0.0, 1.0]), 
                "sub": st.sampled_from([0.0, 1.0]),
                "sur": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)
            }),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100)
    def test_two_phase_propagation_property(self, step_count, z_sequence):
        """Property test: Two-phase propagation (z=W·a, f_gate(f_node(z))) should be consistent."""
        node = CompactReCoNNode("propagation_test", "script")
        
        # Test sequence of z inputs (simulating propagation steps)
        for step, z_values in enumerate(z_sequence[:step_count]):
            old_activation = float(node.activation) if not isinstance(node.activation, torch.Tensor) else node.activation.item()
            
            # Apply compact rules
            gates = node.update_state_compact(z_values)
            
            new_activation = float(node.activation) if not isinstance(node.activation, torch.Tensor) else node.activation.item()
            
            # Verify gen loop behavior (activation should be f_gen)
            expected_activation = gates["gen"]
            if isinstance(expected_activation, torch.Tensor):
                expected_activation = expected_activation.item()
            
            assert abs(new_activation - expected_activation) < 1e-6, \
                f"Step {step}: activation should equal f_gen, got {new_activation} vs {expected_activation}"
            
            # Verify all gates are computed
            for gate_type in ["gen", "por", "ret", "sub", "sur"]:
                assert gate_type in gates, f"Missing gate computation: {gate_type}"
                
                gate_val = gates[gate_type]
                if isinstance(gate_val, torch.Tensor):
                    gate_val = gate_val.item()
                
                # Gates should be finite numbers
                assert not (torch.isnan(torch.tensor(gate_val)) or torch.isinf(torch.tensor(gate_val))), \
                    f"Gate {gate_type} should be finite, got {gate_val}"
    
    def test_section31_reference_equations(self):
        """Reference test: Manually verify each f_node equation from section 3.1."""
        node = CompactReCoNNode("reference", "script")
        
        # Test cases covering key equation branches
        test_cases = [
            # (z_values, has_por, has_ret, expected_results, description)
            ({"gen": 0.0, "por": 0.0, "ret": 0.0, "sub": 1.0, "sur": 0.5}, False, False,
             {"gen": 0.5, "por": 0.5, "ret": 0.0, "sub": 1.0, "sur": 0.5}, 
             "Basic case: no links, z^gen·z^sub=0"),
            
            ({"gen": 0.5, "por": 0.0, "ret": 0.0, "sub": 1.0, "sur": 0.3}, False, False,
             {"gen": 0.5, "por": 0.8, "ret": 0.0, "sub": 0.0, "sur": 0.8},
             "Gen loop: z^gen·z^sub≠0 (z^gen≠0 → f_sub=0)"),
            
            ({"gen": 0.0, "por": -1.0, "ret": 0.0, "sub": 1.0, "sur": 0.5}, True, False,
             {"gen": 0.5, "por": 0.0, "ret": 1.0, "sub": 0.0, "sur": 0.0},
             "Por inhibition: z^por<0 with por link"),
            
            ({"gen": 0.0, "por": 0.0, "ret": 0.5, "sub": 1.0, "sur": 0.3}, False, True,
             {"gen": 0.3, "por": 0.3, "ret": 0.0, "sub": 1.0, "sur": 0.15},
             "Ret modulation: (z^sur+z^gen)·z^ret"),
        ]
        
        for z_values, has_por, has_ret, expected, description in test_cases:
            print(f"\nTesting: {description}")
            print(f"  Input z: {z_values}")
            print(f"  Links: por={has_por}, ret={has_ret}")
            
            node.set_link_existence(has_por=has_por, has_ret=has_ret)
            actual = node.update_state_compact(z_values)
            
            print(f"  Expected: {expected}")
            print(f"  Actual:   {actual}")
            
            for gate_type in ["gen", "por", "ret", "sub", "sur"]:
                actual_val = actual[gate_type]
                expected_val = expected[gate_type]
                
                if isinstance(actual_val, torch.Tensor):
                    actual_val = actual_val.item()
                
                assert abs(actual_val - expected_val) < 1e-6, \
                    f"{description}: f_node^{gate_type} expected {expected_val}, got {actual_val}"


class TestSection22NeuralDefinition:
    """Test what neural components exist and identify missing Section 2.2 implementation."""
    
    def test_neural_components_inventory(self):
        """Inventory of existing neural components vs Section 2.2 requirements."""
        from recon_engine.neural_terminal import NeuralTerminal
        from recon_engine.hybrid_node import HybridReCoNNode, NodeMode
        import torch.nn as nn
        
        # What we have: Neural terminals and hybrid neural modes
        neural_terminal = NeuralTerminal("nt", nn.Linear(1, 1))
        hybrid_neural = HybridReCoNNode("hn", "script", NodeMode.NEURAL)
        
        print("✅ Available neural components:")
        print(f"  - NeuralTerminal: {type(neural_terminal)}")
        print(f"  - HybridReCoNNode with NEURAL mode: {hybrid_neural.mode}")
        print(f"  - Neural model support: {hasattr(neural_terminal, 'model')}")
        
        # What Section 2.2 requires but we don't have:
        print("\n❌ Missing Section 2.2 components:")
        print("  - 10 threshold element ensemble (IC, IR, W neurons)")
        print("  - Specific activation function: α_j = Σ(w_ij·α_i) if all w_ij·α_i ≥ 0, else 0")
        print("  - Inhibitory link logic: any negative weight → complete inhibition")
        print("  - Neural unit boundary message passing")
        
        # Verify current neural components work
        assert hasattr(neural_terminal, 'model'), "NeuralTerminal should have model"
        assert hybrid_neural.mode == NodeMode.NEURAL, "Should support neural mode"
    
    def test_section22_activation_function_concept(self):
        """Test the concept of Section 2.2's activation function with mock implementation."""
        
        def section22_activation_function(weights, activations):
            """Mock implementation of paper's α_j activation function."""
            # α_j = Σ(w_ij·α_i) if w_ij·α_i ≥ 0 for all i, else 0
            
            weighted_activations = [w * a for w, a in zip(weights, activations)]
            
            # Check if any weighted activation is negative (inhibition)
            if any(wa < 0 for wa in weighted_activations):
                return 0.0  # Complete inhibition
            else:
                return sum(weighted_activations)  # Excitatory sum
        
        # Test cases from paper's neural definition
        test_cases = [
            # (weights, activations, expected, description)
            ([1.0, 1.0, 1.0], [0.5, 0.3, 0.2], 1.0, "All positive → sum"),
            ([1.0, -1.0, 1.0], [0.5, 0.3, 0.2], 0.0, "Any negative → inhibition"),
            ([2.0, 0.0, 1.0], [0.5, 0.0, 0.8], 1.8, "Zero weights allowed"),
            ([-1.0, -1.0], [0.1, 0.1], 0.0, "All negative → complete inhibition")
        ]
        
        for weights, activations, expected, description in test_cases:
            result = section22_activation_function(weights, activations)
            print(f"{description}: {weights} · {activations} = {result}")
            
            assert abs(result - expected) < 1e-6, \
                f"Section 2.2 activation function: {description} expected {expected}, got {result}"
        
        print("\n💡 This demonstrates the neural threshold logic from Section 2.2")
        print("   Our current implementation uses different neural approaches")


class TestSection31CompactEquations:
    """Test specific equation compliance from Section 3.1."""
    
    @given(
        z_gen=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        z_sub=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        z_sur=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=200)
    def test_f_node_gen_equation(self, z_gen, z_sub, z_sur):
        """Property test: f_node^gen equation compliance."""
        node = CompactReCoNNode("gen_test", "script")
        
        # Test without por link
        node.set_link_existence(has_por=False, has_ret=False)
        z = {"gen": z_gen, "por": 0.0, "ret": 0.0, "sub": z_sub, "sur": z_sur}
        gates = node.update_state_compact(z)
        
        # Paper equation: z^sur if z^gen·z^sub=0, otherwise z^gen·z^sub
        if z_gen * z_sub == 0:
            expected = z_sur
        else:
            expected = z_gen * z_sub
        
        actual = gates["gen"]
        if isinstance(actual, torch.Tensor):
            actual = actual.item()
        
        assert abs(actual - expected) < 1e-6, \
            f"f_node^gen: z_gen={z_gen}, z_sub={z_sub}, z_sur={z_sur} → expected {expected}, got {actual}"
    
    @given(
        z_por=st.sampled_from([-1.0, 0.0, 1.0])
    )
    @settings(max_examples=20)
    def test_f_node_ret_equation(self, z_por):
        """Property test: f_node^ret equation compliance."""
        node = CompactReCoNNode("ret_test", "script")
        
        z = {"gen": 0.0, "por": z_por, "ret": 0.0, "sub": 1.0, "sur": 0.5}
        gates = node.update_state_compact(z)
        
        # Paper equation: 1 if z^por<0, otherwise 0
        expected = 1.0 if z_por < 0 else 0.0
        
        actual = gates["ret"]
        if isinstance(actual, torch.Tensor):
            actual = actual.item()
        
        assert abs(actual - expected) < 1e-6, \
            f"f_node^ret: z_por={z_por} → expected {expected}, got {actual}"
    
    @given(
        z_sub=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        z_por=st.sampled_from([-1.0, 0.0, 1.0]),
        has_por_link=st.booleans()
    )
    @settings(max_examples=100)
    def test_f_node_sub_equation(self, z_sub, z_por, has_por_link):
        """Property test: f_node^sub equation compliance."""
        node = CompactReCoNNode("sub_test", "script")
        node.set_link_existence(has_por=has_por_link, has_ret=False)
        
        z = {"gen": 0.0, "por": z_por, "ret": 0.0, "sub": z_sub, "sur": 0.5}
        gates = node.update_state_compact(z)
        
        # Paper equation: 0 if z^gen≠0 ∨ (∃link^por ∧ z^por≤0), otherwise z^sub
        if (0.0 != 0) or (has_por_link and z_por <= 0):  # z_gen is 0 in our test
            expected = 0.0
        else:
            expected = z_sub
        
        actual = gates["sub"]
        if isinstance(actual, torch.Tensor):
            actual = actual.item()
        
        assert abs(actual - expected) < 1e-6, \
            f"f_node^sub: z_sub={z_sub}, z_por={z_por}, has_por={has_por_link} → expected {expected}, got {actual}"
    
    @given(
        z_sur=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        z_gen=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        z_ret=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        has_ret_link=st.booleans()
    )
    @settings(max_examples=150)
    def test_f_node_sur_equation(self, z_sur, z_gen, z_ret, has_ret_link):
        """Property test: f_node^sur equation compliance."""
        node = CompactReCoNNode("sur_test", "script")
        node.set_link_existence(has_por=False, has_ret=has_ret_link)
        
        z = {"gen": z_gen, "por": 0.0, "ret": z_ret, "sub": 1.0, "sur": z_sur}  # z_sub > 0
        gates = node.update_state_compact(z)
        
        # Paper equation: (z^sur+z^gen)·z^ret if ∃link^ret, otherwise z^sur+z^gen
        if has_ret_link:
            expected = (z_sur + z_gen) * z_ret
        else:
            expected = z_sur + z_gen
        
        actual = gates["sur"]
        if isinstance(actual, torch.Tensor):
            actual = actual.item()
        
        assert abs(actual - expected) < 1e-6, \
            f"f_node^sur: z_sur={z_sur}, z_gen={z_gen}, z_ret={z_ret}, has_ret={has_ret_link} → expected {expected}, got {actual}"


class TestCompactImplementationInvariants:
    """Test invariants that should hold for the compact implementation."""
    
    @given(
        z_values=st.fixed_dictionaries({
            "gen": st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
            "por": st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
            "ret": st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
            "sub": st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
            "sur": st.floats(min_value=-2.0, max_value=2.0, allow_nan=False)
        }),
        link_config=st.tuples(st.booleans(), st.booleans())  # (has_por, has_ret)
    )
    @settings(max_examples=300)
    def test_compact_implementation_invariants(self, z_values, link_config):
        """Property test: Compact implementation invariants should always hold."""
        node = CompactReCoNNode("invariant_test", "script")
        has_por, has_ret = link_config
        
        node.set_link_existence(has_por=has_por, has_ret=has_ret)
        gates = node.update_state_compact(z_values)
        
        # Invariant 1: All gates should be computed
        required_gates = ["gen", "por", "ret", "sub", "sur"]
        for gate in required_gates:
            assert gate in gates, f"Missing gate: {gate}"
        
        # Invariant 2: Gates should be finite numbers
        for gate, value in gates.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            assert not (torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value))), \
                f"Gate {gate} should be finite, got {value}"
        
        # Invariant 3: Node activation should equal f_gen (gen loop)
        node_activation = node.activation
        if isinstance(node_activation, torch.Tensor):
            node_activation = node_activation.item()
        
        f_gen = gates["gen"]
        if isinstance(f_gen, torch.Tensor):
            f_gen = f_gen.item()
        
        assert abs(node_activation - f_gen) < 1e-6, \
            f"Node activation should equal f_gen: {node_activation} vs {f_gen}"
        
        # Invariant 4: f_ret should be binary (0 or 1)
        f_ret = gates["ret"]
        if isinstance(f_ret, torch.Tensor):
            f_ret = f_ret.item()
        assert f_ret in [0.0, 1.0], f"f_ret should be binary, got {f_ret}"
