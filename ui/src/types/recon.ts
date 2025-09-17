// ReCoN Network Types for UI

export type ReCoNNodeType = 'script' | 'terminal' | 'hybrid';
export type ReCoNLinkType = 'sub' | 'sur' | 'por' | 'ret' | 'gen';
export type ReCoNState = 'inactive' | 'requested' | 'active' | 'suppressed' | 'waiting' | 'true' | 'confirmed' | 'failed';
export type ExecutionMode = 'explicit' | 'neural' | 'implicit';

export interface ReCoNNode {
  id: string;
  type: ReCoNNodeType;
  state: ReCoNState;
  activation: number;
  mode?: ExecutionMode;
  position: { x: number; y: number };
}

export interface ReCoNLink {
  id: string;
  source: string;
  target: string;
  type: ReCoNLinkType;
  weight: number;
}

export interface ReCoNNetwork {
  id: string;
  nodes: ReCoNNode[];
  links: ReCoNLink[];
  stepCount: number;
  requestedRoots: string[];
}

export interface ExecutionState {
  isExecuting: boolean;
  currentStep: number;
  executionResult?: 'confirmed' | 'failed' | 'timeout';
  animatingMessages: boolean;
}

export interface NetworkState {
  currentNetwork: ReCoNNetwork | null;
  selectedNode: ReCoNNode | null;
  selectedLink: ReCoNLink | null;
  isDirty: boolean;
}