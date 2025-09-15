// Zustand store for ReCoN network state management

import { create } from 'zustand';
import { ReCoNNetwork, ReCoNNode, ReCoNLink, NetworkState, ExecutionState } from '../types/recon';
import { reconAPI } from '../services/api';

interface NetworkStore extends NetworkState {
  // Network operations
  createNetwork: (id?: string) => Promise<void>;
  loadNetwork: (id: string) => Promise<void>;
  saveNetwork: () => Promise<void>;
  deleteNetwork: (id: string) => Promise<void>;

  // Node operations
  addNode: (node: Omit<ReCoNNode, 'id'>) => Promise<void>;
  updateNode: (id: string, updates: Partial<ReCoNNode>) => void;
  deleteNode: (id: string) => void;
  selectNode: (node: ReCoNNode | null) => void;

  // Link operations
  addLink: (link: Omit<ReCoNLink, 'id'>) => Promise<void>;
  updateLink: (id: string, updates: Partial<ReCoNLink>) => void;
  deleteLink: (id: string) => void;
  selectLink: (link: ReCoNLink | null) => void;

  // Execution operations
  requestNode: (nodeId: string) => Promise<void>;
  executeScript: (rootNodeId: string, maxSteps?: number) => Promise<void>;
  propagateStep: () => Promise<void>;
  resetNetwork: () => Promise<void>;

  // UI state
  setDirty: (dirty: boolean) => void;
  clearSelection: () => void;
}

interface ExecutionStore extends ExecutionState {
  setExecuting: (executing: boolean) => void;
  setCurrentStep: (step: number) => void;
  setExecutionResult: (result: 'confirmed' | 'failed' | 'timeout' | undefined) => void;
  setAnimatingMessages: (animating: boolean) => void;
  reset: () => void;
}

export const useNetworkStore = create<NetworkStore>((set, get) => ({
  // Initial state
  currentNetwork: null,
  selectedNode: null,
  selectedLink: null,
  isDirty: false,

  // Network operations
  createNetwork: async (id?: string) => {
    // Create a local network without requiring backend
    const networkId = id || `network_${Date.now()}`;
    const network: ReCoNNetwork = {
      id: networkId,
      nodes: [],
      links: [],
      stepCount: 0,
      requestedRoots: [],
    };

    set({ currentNetwork: network, isDirty: false });
  },

  loadNetwork: async (id: string) => {
    try {
      const response = await reconAPI.getNetwork(id);
      const network: ReCoNNetwork = {
        id: response.network_id,
        nodes: response.nodes.map(node => ({
          id: node.node_id,
          type: node.node_type as any,
          state: node.state as any,
          activation: node.activation,
          position: { x: 0, y: 0 }, // Will be set by layout
        })),
        links: response.links.map(link => ({
          id: `${link.source}-${link.target}-${link.link_type}`,
          source: link.source,
          target: link.target,
          type: link.link_type as any,
          weight: link.weight,
        })),
        stepCount: response.step_count,
        requestedRoots: [],
      };

      set({ currentNetwork: network, isDirty: false });
    } catch (error) {
      console.error('Failed to load network:', error);
      throw error;
    }
  },

  saveNetwork: async () => {
    const { currentNetwork } = get();
    if (!currentNetwork) return;

    // Network is automatically saved via API calls
    set({ isDirty: false });
  },

  deleteNetwork: async (id: string) => {
    try {
      await reconAPI.deleteNetwork(id);
      const { currentNetwork } = get();
      if (currentNetwork?.id === id) {
        set({ currentNetwork: null, selectedNode: null, selectedLink: null, isDirty: false });
      }
    } catch (error) {
      console.error('Failed to delete network:', error);
      throw error;
    }
  },

  // Node operations
  addNode: async (nodeData: any) => {
    const { currentNetwork } = get();

    // If no network exists, create a demo network first
    if (!currentNetwork) {
      const demoNetwork: ReCoNNetwork = {
        id: 'demo-network',
        nodes: [],
        links: [],
        stepCount: 0,
        requestedRoots: [],
      };
      set({ currentNetwork: demoNetwork });
    }

    const newNode: ReCoNNode = {
      id: nodeData.id,
      type: nodeData.type,
      state: nodeData.state || 'inactive',
      activation: nodeData.activation || 0,
      mode: nodeData.mode,
      position: nodeData.position,
    };

    set(state => ({
      currentNetwork: state.currentNetwork ? {
        ...state.currentNetwork,
        nodes: [...state.currentNetwork.nodes, newNode],
      } : null,
      isDirty: true,
    }));
  },

  updateNode: (id: string, updates: Partial<ReCoNNode>) => {
    set(state => ({
      currentNetwork: state.currentNetwork ? {
        ...state.currentNetwork,
        nodes: state.currentNetwork.nodes.map(node =>
          node.id === id ? { ...node, ...updates } : node
        ),
      } : null,
      selectedNode: state.selectedNode?.id === id ? { ...state.selectedNode, ...updates } : state.selectedNode,
      isDirty: true,
    }));
  },

  deleteNode: (id: string) => {
    set(state => ({
      currentNetwork: state.currentNetwork ? {
        ...state.currentNetwork,
        nodes: state.currentNetwork.nodes.filter(node => node.id !== id),
        links: state.currentNetwork.links.filter(link => link.source !== id && link.target !== id),
      } : null,
      selectedNode: state.selectedNode?.id === id ? null : state.selectedNode,
      isDirty: true,
    }));
  },

  selectNode: (node: ReCoNNode | null) => {
    set({ selectedNode: node, selectedLink: null });
  },

  // Link operations
  addLink: async (linkData: Omit<ReCoNLink, 'id'>) => {
    const { currentNetwork } = get();
    if (!currentNetwork) return;

    const newLink: ReCoNLink = {
      ...linkData,
      id: `${linkData.source}-${linkData.target}-${linkData.type}`,
    };

    set(state => ({
      currentNetwork: state.currentNetwork ? {
        ...state.currentNetwork,
        links: [...state.currentNetwork.links, newLink],
      } : null,
      isDirty: true,
    }));
  },

  updateLink: (id: string, updates: Partial<ReCoNLink>) => {
    set(state => ({
      currentNetwork: state.currentNetwork ? {
        ...state.currentNetwork,
        links: state.currentNetwork.links.map(link =>
          link.id === id ? { ...link, ...updates } : link
        ),
      } : null,
      selectedLink: state.selectedLink?.id === id ? { ...state.selectedLink, ...updates } : state.selectedLink,
      isDirty: true,
    }));
  },

  deleteLink: (id: string) => {
    set(state => ({
      currentNetwork: state.currentNetwork ? {
        ...state.currentNetwork,
        links: state.currentNetwork.links.filter(link => link.id !== id),
      } : null,
      selectedLink: state.selectedLink?.id === id ? null : state.selectedLink,
      isDirty: true,
    }));
  },

  selectLink: (link: ReCoNLink | null) => {
    set({ selectedLink: link, selectedNode: null });
  },

  // Execution operations
  requestNode: async (nodeId: string) => {
    const { currentNetwork } = get();
    if (!currentNetwork) return;

    try {
      await reconAPI.requestNode(currentNetwork.id, nodeId);
      set(state => ({
        currentNetwork: state.currentNetwork ? {
          ...state.currentNetwork,
          requestedRoots: [...state.currentNetwork.requestedRoots, nodeId],
        } : null,
      }));
    } catch (error) {
      console.error('Failed to request node:', error);
      throw error;
    }
  },

  executeScript: async (rootNodeId: string, maxSteps = 100) => {
    const { currentNetwork } = get();
    if (!currentNetwork) return;

    try {
      const result = await reconAPI.executeScript(currentNetwork.id, {
        root_node: rootNodeId,
        max_steps: maxSteps,
      });

      // Update node states based on execution result
      set(state => ({
        currentNetwork: state.currentNetwork ? {
          ...state.currentNetwork,
          nodes: state.currentNetwork.nodes.map(node => ({
            ...node,
            state: (result.final_states[node.id] as any) || node.state,
          })),
          stepCount: result.steps_taken,
        } : null,
      }));

      return result;
    } catch (error) {
      console.error('Failed to execute script:', error);
      throw error;
    }
  },

  propagateStep: async () => {
    const { currentNetwork } = get();
    if (!currentNetwork) return;

    try {
      const result = await reconAPI.propagateStep(currentNetwork.id);
      set(state => ({
        currentNetwork: state.currentNetwork ? {
          ...state.currentNetwork,
          stepCount: result.step,
        } : null,
      }));
    } catch (error) {
      console.error('Failed to propagate step:', error);
      throw error;
    }
  },

  resetNetwork: async () => {
    const { currentNetwork } = get();
    if (!currentNetwork) return;

    try {
      await reconAPI.resetNetwork(currentNetwork.id);
      set(state => ({
        currentNetwork: state.currentNetwork ? {
          ...state.currentNetwork,
          stepCount: 0,
          requestedRoots: [],
          nodes: state.currentNetwork.nodes.map(node => ({
            ...node,
            state: 'inactive' as any,
            activation: 0,
          })),
        } : null,
      }));
    } catch (error) {
      console.error('Failed to reset network:', error);
      throw error;
    }
  },

  // UI state
  setDirty: (dirty: boolean) => {
    set({ isDirty: dirty });
  },

  clearSelection: () => {
    set({ selectedNode: null, selectedLink: null });
  },
}));

export const useExecutionStore = create<ExecutionStore>((set) => ({
  // Initial state
  isExecuting: false,
  currentStep: 0,
  executionResult: undefined,
  animatingMessages: false,

  setExecuting: (executing: boolean) => set({ isExecuting: executing }),
  setCurrentStep: (step: number) => set({ currentStep: step }),
  setExecutionResult: (result: 'confirmed' | 'failed' | 'timeout' | undefined) => set({ executionResult: result }),
  setAnimatingMessages: (animating: boolean) => set({ animatingMessages: animating }),

  reset: () => set({
    isExecuting: false,
    currentStep: 0,
    executionResult: undefined,
    animatingMessages: false,
  }),
}));