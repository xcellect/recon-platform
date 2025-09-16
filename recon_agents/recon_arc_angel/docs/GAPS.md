Gaps and quick fixes
1) Critical bug: CNN terminal node insertion
ReCoNGraph tracks nodes in self.nodes, not self.graph.nodes (networkx). add_link checks self.nodes and will raise.
Fix: use self.graph.add_node(self.cnn_terminal) (or self.graph.add_node("cnn_terminal","terminal") and then attach model), then add_link.
2) Region confirmation threshold
With measurement=0.9 and transition_threshold=0.8, a region needs weight ≥0.89 to confirm. Either:
Set region terminals measurement_fn = lambda _: 1.0 so sur equals region weight directly; or
Reduce action_click.transition_threshold to ~0.5–0.7.
3) Action scripts vs terminals (plan compliance)
Plan and tests prefer action scripts with terminal children. HypothesisManager uses terminal-only actions; workable, but for strict plan compliance mirror EfficientHierarchicalHypothesisManager (scripts + terminal children) or switch the agent to use the efficient manager.
4) Coordinate selection
Current HypothesisManager.get_best_action uses region center, not argmax within region as per plan. Quick improvement: pick argmax location within the selected region from the cached 64×64 probs.
5) Availability gating
The agent sets unavailable actions to FAILED directly, which is fine pragmatically, but if you want pure ReCoN semantics, set the root→child sub weight to ≈0 to defer requests, rather than forcing FAILED.
Overall assessment
The design aligns with the refined plan’s core: single CNN terminal, continuous sur in explicit FSM, sub/sur weight usage, regional hierarchy, StochasticGoose-style learning; plus BlindSquirrel value and segmentation tracks. It’s viable after the CNN-terminal insertion fix and minor threshold/selection adjustments.
Minimal fixes checklist
Fix CNN terminal insertion in HypothesisManager (call add_node(self.cnn_terminal)).
Lower action_click.transition_threshold or set region terminal measurement to 1.0.
Optionally convert action_1..5 to scripts with terminal children (plan strictness).
Use regional argmax for ACTION6 coordinate selection.
Run tests:
Engine sur tests (already passing).
recon_arc_angel tests (cnn_integration, plan_compliance, selection_availability, learning).
If you want immediate plan strictness plus better scalability, consider switching the agent to EfficientHierarchicalHypothesisManager and keep the current learning manager; it already satisfies “scripts with terminal children,” dynamic coordinates, and integrates BlindSquirrel components.