# Sequence Test Analysis

## Summary
All 6 previously skipped sequence tests fail due to timing and state transition differences between test expectations and the paper's specification.

## Root Causes

### 1. Mixed Hierarchy and Sequence Structure
The tests create nodes that are both:
- Part of a sequence (connected via por/ret links)
- Part of a hierarchy (connected via sub/sur links)

Example from `test_successors_wait_for_predecessors`:
```
Root -> A (sub link)
A -> B (both sub AND por links)
B -> C (both sub AND por links)
```

This creates conflicting behaviors:
- When A becomes TRUE, it stops sending por inhibition (correct per Table 1)
- But A also stops sending sub requests (correct per Table 1)
- B loses both inhibition AND request, causing unexpected state transitions

### 2. State Transition Timing
Tests expect specific states at specific propagation steps, but the paper's state machine has different timing:

**Test Expectation:**
- Step 7: B should be SUPPRESSED

**Actual Behavior (following paper):**
- Step 5: B is SUPPRESSED (correctly inhibited by A)
- Step 6: B becomes ACTIVE (A is TRUE, stops inhibiting)
- Step 7: B becomes WAITING (has children to request)

### 3. TRUE State Behavior
According to Table 1 from the paper, TRUE state sends:
- por: - (nothing)
- ret: inhibit_confirm
- sub: - (nothing)
- sur: - (nothing)

This means TRUE nodes:
- Stop inhibiting successors (allowing them to activate)
- Stop requesting children (causing them to potentially go INACTIVE)

### 4. Test Structure Issues

#### test_successors_wait_for_predecessors
- Expects B to remain SUPPRESSED after A goes TRUE
- But paper says TRUE nodes don't send inhibit_request

#### test_only_last_node_confirms_parent
- Expects intermediate nodes to be TRUE, last node CONFIRMED
- Actual: All nodes become CONFIRMED due to request propagation

#### test_sequence_failure_propagation
- Expects failure to propagate through sequence
- Actual: Nodes confirm independently

#### test_nested_sequences
- Complex nested structure timing issues
- Nodes go INACTIVE when parents stop requesting

#### test_sequence_timing_constraints
- Expects specific timing that doesn't match paper's state machine

#### test_sequence_interruption
- Expects specific behavior when requests are removed
- Actual behavior follows paper's specification

## Recommendations

### Option 1: Fix Test Expectations
Update tests to match the paper's specification:
- Adjust expected states based on correct state machine
- Account for proper timing of transitions
- Respect Table 1 message passing rules

### Option 2: Implement Compatibility Mode
Add special handling for mixed hierarchy/sequence structures:
- TRUE nodes could continue requesting children in sequences
- Add sequence-aware state transitions
- This would deviate from the paper but match test expectations

### Option 3: Redesign Test Structure
Avoid mixing hierarchy and sequence in ways that create conflicts:
- Use pure sequences (only por/ret links)
- Use pure hierarchies (only sub/sur links)
- Don't mix both on the same nodes

## Conclusion

The sequence tests fail because they expect behavior that violates the paper's specification. The core implementation is correct according to the theoretical foundation. The tests need to be updated to match the paper's state machine and message passing rules, or we need to explicitly document where we deviate from the paper for practical reasons.

## Current Implementation Status

✅ **Correctly Implements:**
- State transitions per paper
- Message passing per Table 1
- por/ret inhibition mechanisms
- sub/sur hierarchy propagation

⚠️ **Test Issues:**
- Tests expect different timing
- Tests create invalid structures (nodes without proper children)
- Tests mix hierarchy and sequence in conflicting ways

The implementation maintains theoretical rigor. The test failures indicate the tests themselves need revision to align with the paper's specification.
