Based on our deep analyses, here are concrete, actionable extension plans organized by feasibility and impact:

  ---
  CONCRETE EXTENSION PLANS

  TIER 1: SAFE DEMO EXTENSIONS (2-4 hours each)

  Can fully implement and test before interview, low demo risk

  ---
  Extension 1A: Compact Implementation with Gen Loops

  Motivation: Section 3.2 uses compact formulation with gen loops for state persistence

  What we learned:
  - Message-passing (current): Explicit states, no temporal integration
  - Compact (Section 3.1): Gen loops store persistent states
  - Section 3.2 (MicroPsi2): Uses compact with 0.95 gen loops

  Implementation Plan:

  Step 1: Switch to CompactReCoNGraph (30 minutes)

  # File: recon_agents/recon_arc_angel/improved_hierarchy_manager.py

  # Current:
  from recon_engine.graph import ReCoNGraph

  # Change to:
  from recon_engine.compact import CompactReCoNGraph

  class ImprovedHierarchicalHypothesisManager:
      def __init__(self, cnn_threshold: float = 0.1, max_objects: int = 50):
          # Change this line:
          self.graph = CompactReCoNGraph()  # Instead of ReCoNGraph()
          # Rest stays same

  Step 2: Add Gen Loops to Object Terminals (1 hour)

  def _add_gen_loops_to_objects(self):
      """Add gen loops for object hypothesis persistence"""
      for obj_idx in range(len(self.current_objects)):
          obj_id = f"object_{obj_idx}"

          # Add gen loop with 0.95 persistence (Section 3.2 style)
          self.graph.add_link(obj_id, obj_id, "gen", weight=0.95)

          # Initialize gen activation from object confidence
          obj_node = self.graph.get_node(obj_id)
          obj = self.current_objects[obj_idx]
          obj_node.gates["gen"] = obj["confidence"]

  Step 3: Update Object Confidence via Gen Accumulation (1 hour)

  def update_objects_with_persistence(self, frame):
      """Update object hypotheses with temporal integration"""
      new_objects = self.extract_objects_from_frame(frame)

      for new_obj in new_objects:
          # Find matching existing object (spatial overlap)
          matched_old_obj = self.find_matching_object(new_obj)

          if matched_old_obj:
              # Accumulate via gen loop: 95% old + 5% new
              old_confidence = matched_old_obj.gates["gen"]
              new_confidence = new_obj["confidence"]

              # Gen loop formula from Section 3.1
              accumulated = 0.95 * old_confidence + 0.05 * new_confidence
              matched_old_obj.gates["gen"] = accumulated

              # Use accumulated confidence for selection
              new_obj["confidence"] = accumulated
          else:
              # New object, no history
              new_obj["confidence"] = new_obj["regularity"]

      return new_objects

  Step 4: Test and Validate (1 hour)

  # Test script: test_gen_loops.py
  def test_gen_loop_persistence():
      manager = ImprovedHierarchicalHypothesisManager()
      manager.build_improved_structure()

      # Frame 1: Object appears
      frame1 = create_test_frame_with_object(x=10, y=10)
      manager.update_objects_with_persistence(frame1)
      obj1_confidence = manager.current_objects[0]["confidence"]

      # Frame 2: Same object, update confidence
      frame2 = create_test_frame_with_object(x=10, y=10)
      manager.update_objects_with_persistence(frame2)
      obj2_confidence = manager.current_objects[0]["confidence"]

      # Verify accumulation: should be 0.95 * old + 0.05 * new
      assert obj2_confidence > obj1_confidence, "Confidence should accumulate"
      print(f"✅ Gen loop persistence working: {obj1_confidence:.3f} → {obj2_confidence:.3f}")

  Expected Outcome:
  - Object confidences smooth over time (less noisy)
  - Persistent objects get stronger scores
  - Faithful to Section 3.2 compact implementation

  Demo Strategy:
  [Show current implementation]
  "Currently using message-passing - stateless per frame."

  [Switch to compact - ONE LINE CHANGE]
  self.graph = CompactReCoNGraph()

  [Add gen loops - 10 lines]
  [Show smoothed confidence over frames]
  "See how object confidence accumulates via gen loops -
  this is Section 3.2's compact implementation."

  Time Investment: 3.5 hoursDemo Risk: Low (can test thoroughly)Impact: High (shows understanding of both formulations)

  ---
  Extension 1B: Link Weight Learning from Outcomes

  Motivation: True neuro-symbolic integration requires bidirectional feedback

  What we learned:
  - Current: CNN sets link weights, never updated
  - Paper mentions: "weights of connecting links" for learning
  - Neither MicroPsi2 nor current impl learns link weights through ReCoN

  Implementation Plan:

  Step 1: Track Link Weight History (30 minutes)

  class LinkWeightLearner:
      """Learn link weights from action outcomes"""

      def __init__(self, learning_rate: float = 0.1):
          self.lr = learning_rate
          # Track: (source, target, link_type) -> accumulated weight
          self.learned_weights = {}
          # Track success rates for logging
          self.link_success_history = {}

      def get_learned_weight(self, source: str, target: str, link_type: str, 
                            cnn_prior: float) -> float:
          """Get blended weight: learned + CNN prior"""
          key = (source, target, link_type)
          learned = self.learned_weights.get(key, 0.5)  # Default neutral

          # Blend: 70% CNN prior, 30% learned
          return 0.7 * cnn_prior + 0.3 * learned

  Step 2: Update Weights from Outcomes (1 hour)

  def update_from_outcome(self, action: str, obj_idx: int, 
                         outcome: float):
      """
      Update link weights based on action outcome
      
      Args:
          action: Selected action (e.g., "action_click")
          obj_idx: Selected object index
          outcome: Reward signal (1.0 if frame changed, 0.0 if no change)
      """
      # Identify the links involved in this decision
      source = action
      target = f"object_{obj_idx}"
      link_type = "sub"

      key = (source, target, link_type)

      # Current weight
      current = self.learned_weights.get(key, 0.5)

      # Update rule: Move toward 1.0 if success, toward 0.0 if failure
      if outcome > 0:
          # Success: strengthen link
          delta = self.lr * (1.0 - current)
      else:
          # Failure: weaken link
          delta = -self.lr * current

      new_weight = current + delta
      self.learned_weights[key] = np.clip(new_weight, 0.0, 1.0)

      # Log for analysis
      if key not in self.link_success_history:
          self.link_success_history[key] = []
      self.link_success_history[key].append(outcome)

  Step 3: Integrate into Agent (1 hour)

  class ImprovedProductionReCoNArcAngel:
      def __init__(self, ...):
          # Add weight learner
          self.link_learner = LinkWeightLearner(learning_rate=0.1)

      def choose_action_internal(self, frame, available_actions):
          # Extract objects
          objects = self.hypothesis_manager.extract_objects_from_frame(frame)

          # Set link weights using learned values
          for obj_idx, obj in enumerate(objects):
              source = "action_click"
              target = f"object_{obj_idx}"

              # Get CNN prior
              cnn_prior = obj["cnn_confidence"]

              # Blend with learned weight
              final_weight = self.link_learner.get_learned_weight(
                  source, target, "sub", cnn_prior
              )

              # Set link in graph
              self.hypothesis_manager.graph.add_link(
                  source, target, "sub", weight=final_weight
              )

          # Execute ReCoN
          action, coords, obj_idx = self.execute_and_select()

          return action, coords, obj_idx

      def update_after_action(self, prev_action, prev_obj_idx, 
                             frame_changed: bool):
          """Update weights after observing outcome"""
          outcome = 1.0 if frame_changed else 0.0
          self.link_learner.update_from_outcome(
              prev_action, prev_obj_idx, outcome
          )

  Step 4: Visualization for Demo (30 minutes)

  def get_learning_stats(self):
      """Get statistics for demo visualization"""
      stats = {
          "total_links_learned": len(self.link_learner.learned_weights),
          "top_learned_links": [],
          "improvement_over_time": []
      }

      # Find most-learned links
      for key, weight in sorted(
          self.link_learner.learned_weights.items(),
          key=lambda x: abs(x[1] - 0.5),
          reverse=True
      )[:5]:
          source, target, link_type = key
          success_rate = np.mean(self.link_learner.link_success_history.get(key, [0]))
          stats["top_learned_links"].append({
              "link": f"{source} → {target}",
              "learned_weight": weight,
              "success_rate": success_rate
          })

      return stats

  Expected Outcome:
  - Links to successful objects strengthened over time
  - Links to unsuccessful objects weakened
  - Demonstration of learning through ReCoN graph

  Demo Strategy:
  [Show initial weights from CNN only]
  "Currently, link weights come from CNN probabilities only."

  [Enable link learning]
  link_learner = LinkWeightLearner(learning_rate=0.1)

  [Run for 50 actions, show learned weights]
  print(agent.get_learning_stats())

  Output:
  Top Learned Links:
    action_click → object_3: weight=0.85, success_rate=0.9
    action_click → object_7: weight=0.23, success_rate=0.1

  "See how successful actions are reinforced, unsuccessful ones suppressed.
  This is learning THROUGH the ReCoN graph, not just parameters."

  Time Investment: 3 hoursDemo Risk: Low (clear metrics to show)Impact: Very High (addresses neuro-symbolic integration gap)

  ---
  Extension 1C: Hybrid Timing Modes

  Motivation: Paper shows both discrete and continuous formulations

  What we learned:
  - Current: Fixed discrete timing (3 steps wait)
  - MicroPsi2: Activation-based decay
  - Your code already has this! Just expose it

  Implementation Plan:

  Step 1: Expose Timing Configuration (30 minutes)

  class ImprovedHierarchicalHypothesisManager:
      def __init__(self, cnn_threshold: float = 0.1, 
                   max_objects: int = 50,
                   timing_mode: str = "discrete"):  # NEW PARAMETER
          self.graph = ReCoNGraph()
          self.timing_mode = timing_mode

      def build_improved_structure(self):
          # ... existing code ...

          # Configure timing for each action type
          self.configure_timing_modes()

      def configure_timing_modes(self):
          """Set timing modes based on action type"""
          if self.timing_mode == "discrete":
              # Fast simple actions
              for i in range(1, 6):
                  node = self.graph.get_node(f"action_{i}")
                  node.configure_timing(mode="discrete", discrete_wait_steps=2)

              # Slower complex action
              action_click = self.graph.get_node("action_click")
              action_click.configure_timing(mode="discrete", discrete_wait_steps=6)

          elif self.timing_mode == "activation":
              # Activation-based for all (MicroPsi2 style)
              for node_id in self.graph.nodes:
                  node = self.graph.get_node(node_id)
                  node.configure_timing(
                      mode="activation",
                      activation_decay_rate=0.8,
                      activation_failure_threshold=0.1
                  )

          elif self.timing_mode == "hybrid":
              # Discrete for simple, activation for complex
              for i in range(1, 6):
                  node = self.graph.get_node(f"action_{i}")
                  node.configure_timing(mode="discrete", discrete_wait_steps=2)

              action_click = self.graph.get_node("action_click")
              action_click.configure_timing(
                  mode="activation",
                  activation_decay_rate=0.9,
                  activation_failure_threshold=0.15
              )

  Step 2: Live Switching Demo (30 minutes)

  def demo_timing_modes():
      """Show different timing behaviors"""

      # Mode 1: Discrete (current)
      manager_discrete = ImprovedHierarchicalHypothesisManager(
          timing_mode="discrete"
      )
      steps_discrete = run_until_decision(manager_discrete)

      # Mode 2: Activation-based (MicroPsi2)
      manager_activation = ImprovedHierarchicalHypothesisManager(
          timing_mode="activation"
      )
      steps_activation = run_until_decision(manager_activation)

      # Mode 3: Hybrid
      manager_hybrid = ImprovedHierarchicalHypothesisManager(
          timing_mode="hybrid"
      )
      steps_hybrid = run_until_decision(manager_hybrid)

      print(f"""
      Timing Mode Comparison:
      Discrete:    {steps_discrete} steps to decision
      Activation:  {steps_activation} steps to decision
      Hybrid:      {steps_hybrid} steps to decision
      """)

  Demo Strategy:
  "The paper presents both discrete state machines and continuous activation.
  I implemented both. Let me show you the difference..."

  [Switch live between modes]
  [Show step count and confidence curves]

  "Discrete is faster but brittle.
  Activation is smoother but slower.
  Hybrid balances both - this is my innovation."

  Time Investment: 1 hourDemo Risk: Very Low (already implemented!)Impact: Medium (shows understanding, less novel)

  ---
  TIER 2: PROTOTYPE & DISCUSS (4-8 hours)

  Can sketch core, prepare detailed narrative

  ---
  Extension 2A: Sequential Hypothesis Testing with Hidden States

  Motivation: Section 3.2's active perception uses autoencoder hidden states

  What we learned:
  - Section 3.2: Autoencoder hidden states as terminal measurements
  - Sequential foveal scanning builds feature hypotheses
  - Current: One-shot frame processing

  Conceptual Architecture:

  class ActivePerceptionManager:
      """
      Section 3.2 style active perception:
      Sequential scanning with neural feature detection
      """

      def __init__(self):
          self.graph = CompactReCoNGraph()  # With gen loops
          self.cnn_model = CNNValidActionTerminal()

          # Define scanning regions (like MicroPsi2's 7×3 fovea grid)
          self.scan_regions = self.create_scan_grid(grid_size=8)

      def create_scan_grid(self, grid_size: int):
          """Create 8×8 grid of attention regions"""
          regions = []
          for y in range(grid_size):
              for x in range(grid_size):
                  regions.append({
                      'id': f"region_{y}_{x}",
                      'bounds': (x*8, y*8, (x+1)*8, (y+1)*8),
                      'scanned': False
                  })
          return regions

      def build_scanning_hypothesis(self):
          """
          Build sequential scanning structure:
          hypothesis → region_0 → region_1 → ... → region_N
          with por/ret constraints ensuring sequential order
          """
          # Root hypothesis
          self.graph.add_node("frame_hypothesis", "script")

          # Create region scan nodes
          prev_region = None
          for region in self.scan_regions:
              # Scan action (move attention)
              scan_node = self.graph.add_node(
                  f"scan_{region['id']}", "script"
              )

              # Feature detection (CNN hidden state)
              feature_node = CNNHiddenStateTerminal(
                  f"feature_{region['id']}",
                  layer_name="layer3",  # Mid-level features
                  region_bounds=region['bounds']
              )
              self.graph.add_node(feature_node)

              # Scan → Feature
              self.graph.add_link(scan_node, feature_node, "sub")

              # Sequential constraint
              if prev_region:
                  # Can't scan region_i until region_{i-1} done
                  self.graph.add_link(
                      f"scan_{prev_region['id']}",
                      f"scan_{region['id']}",
                      "por"
                  )

              prev_region = region

          # Add gen loops for hypothesis persistence
          for region in self.scan_regions:
              feature_id = f"feature_{region['id']}"
              self.graph.add_link(feature_id, feature_id, "gen", weight=0.95)

  CNNHiddenStateTerminal Implementation:

  class CNNHiddenStateTerminal(ReCoNNode):
      """Terminal that measures CNN hidden layer activation"""

      def __init__(self, node_id, layer_name, region_bounds):
          super().__init__(node_id, "terminal")
          self.layer_name = layer_name
          self.region_bounds = region_bounds  # (x1, y1, x2, y2)

      def measure(self, env):
          """Extract hidden layer activation for this region"""
          if env is None:
              return 0.9  # Default for ReCoN gates

          frame = env['frame']

          # Extract region from frame
          x1, y1, x2, y2 = self.region_bounds
          region_frame = frame[:, y1:y2, x1:x2]

          # Get CNN hidden layer activation
          with torch.no_grad():
              # Forward pass up to target layer
              hidden_activation = self.get_layer_output(
                  self.cnn_model,
                  self.layer_name,
                  region_frame
              )

          # Aggregate over spatial dimensions
          # (Similar to autoencoder hidden node in MicroPsi2)
          feature_strength = hidden_activation.max().item()

          return feature_strength

      def get_layer_output(self, model, layer_name, input_tensor):
          """Hook to extract intermediate layer output"""
          activation = {}
          def hook(module, input, output):
              activation['output'] = output

          # Register hook
          handle = getattr(model, layer_name).register_forward_hook(hook)

          # Forward pass
          model(input_tensor)

          # Clean up
          handle.remove()

          return activation['output']

  Sequential Scanning Execution:

  def execute_sequential_scan(self, frame):
      """
      Execute sequential hypothesis testing:
      Scan regions in order, accumulate features via gen loops
      """
      self.reset_scan_state()

      history = []
      for step in range(100):  # Max steps
          # Propagate ReCoN
          self.graph.propagate_step()

          # Check which region is currently being scanned
          active_scan = self.get_active_scan_node()

          if active_scan:
              region = self.get_region_for_scan(active_scan)
              feature_node = self.graph.get_node(f"feature_{region['id']}")

              # Measure feature (CNN hidden state)
              feature_strength = feature_node.measure({'frame': frame})

              # Accumulate in gen loop
              old_gen = feature_node.gates['gen']
              feature_node.gates['gen'] = 0.95 * old_gen + 0.05 * feature_strength

              history.append({
                  'step': step,
                  'region': region['id'],
                  'feature_strength': feature_strength,
                  'accumulated': feature_node.gates['gen']
              })

              region['scanned'] = True

          # Check if hypothesis confirmed
          if self.all_regions_scanned():
              hypothesis_strength = self.compute_hypothesis_strength()
              return {
                  'confirmed': hypothesis_strength > 0.7,
                  'strength': hypothesis_strength,
                  'scan_history': history
              }

      return {'confirmed': False, 'scan_history': history}

  Why This is Powerful:

  1. Biologically Plausible: Foveal attention, sequential scanning
  2. Feature Learning: CNN hidden states are learned features
  3. Temporal Integration: Gen loops accumulate evidence
  4. Hypothesis Testing: ReCoN confirms/rejects based on features

  Demo Narrative:

  "Section 3.2 describes active perception: sequential scanning with learned features. Let me show you the architecture..."

  [Show diagram of sequential scan nodes with por/ret chains]

  "Each scan extracts CNN hidden layer activation for a region - like MicroPsi2's autoencoder features. Gen loops accumulate evidence across scans."

  [Run simulation, show scan history]

  "See how it builds hypothesis incrementally? This is closer to how vision actually works - not one-shot classification, but active interrogation of the scene."

  Time Investment: 6-8 hoursDemo Risk: Medium (complex, but can show concept)Impact: Very High (aligns with consciousness research)

  ---
  Extension 2B: Hierarchical Object Patterns

  Motivation: Paper emphasizes hierarchical script composition

  Conceptual Architecture:

  class PatternDetector:
      """Detect recurring object configurations"""

      def find_spatial_patterns(self, objects):
          """Find objects in spatial relationships"""
          patterns = []

          for i, obj_i in enumerate(objects):
              for j, obj_j in enumerate(objects[i+1:], i+1):
                  # Check spatial relationship
                  relation = self.get_spatial_relation(obj_i, obj_j)

                  if relation in ['adjacent', 'aligned', 'symmetric']:
                      pattern = {
                          'type': relation,
                          'objects': [i, j],
                          'strength': self.compute_pattern_strength(
                              obj_i, obj_j, relation
                          )
                      }
                      patterns.append(pattern)

          return patterns

      def build_pattern_hypotheses(self, patterns):
          """Create ReCoN nodes for patterns"""
          for pattern in patterns:
              # Meta-object node
              pattern_id = f"pattern_{pattern['type']}_{len(self.patterns)}"
              pattern_node = self.graph.add_node(pattern_id, "script")

              # Link to constituent objects
              for obj_idx in pattern['objects']:
                  obj_id = f"object_{obj_idx}"
                  self.graph.add_link(pattern_node, obj_id, "sub")

              # Spatial constraint between objects
              if len(pattern['objects']) == 2:
                  obj1_id = f"object_{pattern['objects'][0]}"
                  obj2_id = f"object_{pattern['objects'][1]}"
                  # obj1 must confirm before obj2
                  self.graph.add_link(obj1_id, obj2_id, "por")

  Time Investment: 5 hoursImpact: High (ARC-relevant)

  ---
  TIER 3: RESEARCH DIRECTIONS (Discuss Only)

  Extension 3A: Differentiable ReCoN Graphs

  Key Idea: Make ReCoN states differentiable for end-to-end learning

  # Soft state approximations
  state_logits = {
      'inactive': 0.0,
      'requested': soft_request_activation,
      'active': soft_active_activation,
      ...
  }

  # Softmax over states
  state_probs = softmax(state_logits)

  # Differentiable transitions
  next_state_probs = transition_function(state_probs, messages)

  # Loss from outcomes
  loss = -log(P(success | state_probs))

  # Backprop through graph
  ∂loss/∂link_weights

  Challenge: Discreteness of statesImpact: Huge (true neuro-symbolic learning)

  ---
  Extension 3B: Global Workspace Integration

  Key Idea: Winning hypothesis broadcasts to entire graph

  class GlobalWorkspace:
      def broadcast_winner(self, recon_graph):
          # Find confirmed hypothesis
          winner = max(
              [n for n in recon_graph.nodes.values()
               if n.state == CONFIRMED],
              key=lambda n: n.activation
          )

          # Broadcast modulates all other nodes
          for node in recon_graph.nodes.values():
              if node != winner:
                  node.activation *= (1 - 0.3 * winner.activation)

  CIMC Alignment: Direct implementation of consciousness theoryImpact: Extreme (unique research angle)

  ---
  RECOMMENDED PREP PLAN

  Before Interview (Total: 8-10 hours over 2-3 days):

  Day 1 (3-4 hours):
  - ✅ Extension 1A: Compact implementation (3.5h)
  - ✅ Test thoroughly, prepare demo

  Day 2 (3-4 hours):
  - ✅ Extension 1B: Link weight learning (3h)
  - ✅ Extension 1C: Expose timing modes (1h)
  - ✅ Test all, prepare metrics

  Day 3 (2-3 hours):
  - 📝 Sketch Extension 2A architecture (2h)
  - 📝 Prepare Global Workspace narrative (1h)
  - 🎯 Practice demo flow

  Demo Priority in Interview:

  1. Quick wins (5 min): Extensions 1A, 1C
  2. Deep demo (5 min): Extension 1B with learning curves
  3. Discussion (5 min): Extension 2A architecture + Global Workspace

  ---
  FINAL ASSESSMENT

  These extensions collectively demonstrate:

  ✅ Theoretical mastery: Understand all paper formulations✅ Engineering skill: Can implement cleanly and quickly✅ Research thinking: See beyond current implementation✅ CIMC alignment: Connect to consciousness theories

  You'll walk out with them thinking:
  "This person can execute, innovate, and push research forward. Perfect fit."

  Ready to start implementing? Which extension should we tackle first?