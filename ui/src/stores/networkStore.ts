// Zustand store for ReCoN network state management

import { create } from 'zustand';
import type { ReCoNNetwork, ReCoNNode, ReCoNLink, NetworkState, ExecutionState } from '../types/recon';
import { reconAPI } from '../services/api';

// Local storage utilities for node positions
const POSITIONS_STORAGE_KEY = 'recon-node-positions';

interface StoredPositions {
  [networkId: string]: {
    [nodeId: string]: { x: number; y: number };
  };
}

const getStoredPositions = (): StoredPositions => {
  try {
    const stored = localStorage.getItem(POSITIONS_STORAGE_KEY);
    return stored ? JSON.parse(stored) : {};
  } catch {
    return {};
  }
};

const savePositions = (networkId: string, positions: Record<string, { x: number; y: number }>) => {
  try {
    const stored = getStoredPositions();
    stored[networkId] = positions;
    localStorage.setItem(POSITIONS_STORAGE_KEY, JSON.stringify(stored));
  } catch (error) {
    console.warn('Failed to save positions to localStorage:', error);
  }
};

const getNetworkPositions = (networkId: string): Record<string, { x: number; y: number }> => {
  const stored = getStoredPositions();
  return stored[networkId] || {};
};

interface NetworkStore extends NetworkState {
  shouldRelayout?: boolean;
  // Network operations
  loadNetwork: (id: string) => Promise<void>; // Only for demo/initial load
  exportLocalGraph: () => any;

  // Node operations
  addNode: (node: Omit<ReCoNNode, 'id'>) => void;
  updateNode: (id: string, updates: Partial<ReCoNNode>) => void;
  deleteNode: (id: string) => void;
  selectNode: (node: ReCoNNode | null) => void;

  // Link operations
  addLink: (link: Omit<ReCoNLink, 'id'>) => void;
  updateLink: (id: string, updates: Partial<ReCoNLink>) => void;
  deleteLink: (id: string) => void;
  selectLink: (link: ReCoNLink | null) => void;

  // UI operations only - no server execution calls

  // UI state
  setDirty: (dirty: boolean) => void;
  clearSelection: () => void;
  setShouldRelayout: (should: boolean) => void;
  saveCurrentPositions: () => void;
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
  shouldRelayout: false,

  // Network operations - simplified to local only

  loadNetwork: async (id: string) => {
    try {
      const response = await reconAPI.getNetwork(id);
      const storedPositions = getNetworkPositions(response.network_id);
      
      console.log(`Loading network ${response.network_id}, found ${Object.keys(storedPositions).length} stored positions`);
      
      const network: ReCoNNetwork = {
        id: response.network_id,
        nodes: response.nodes.map(node => {
          const position = storedPositions[node.node_id] || { x: 0, y: 0 };
          console.log(`Node ${node.node_id}: position ${position.x}, ${position.y}`);
          
          return {
            id: node.node_id,
            type: node.node_type as any,
            state: node.state as any,
            activation: node.activation,
            // Restore position from localStorage or default to (0,0) for layout
            position,
          };
        }),
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

  // saveNetwork and deleteNetwork removed - no server persistence needed

  // Node operations
  addNode: (nodeData: any) => {
    const { currentNetwork } = get();
    if (!currentNetwork) return;

    const newNode: ReCoNNode = {
      id: nodeData.id,
      type: nodeData.type,
      state: 'inactive' as any,
      activation: 0,
      mode: nodeData.mode,
      position: nodeData.position,
    };

    // Add node locally - no server calls
    set(state => ({
      currentNetwork: state.currentNetwork ? {
        ...state.currentNetwork,
        nodes: [...state.currentNetwork.nodes, newNode],
      } : null,
      isDirty: true,
    }));
  },

  updateNode: (id: string, updates: Partial<ReCoNNode>) => {
    set(state => {
      if (!state.currentNetwork) return state;
      
      const existingNode = state.currentNetwork.nodes.find(node => node.id === id);
      if (!existingNode) return state;
      
      // Force update for debugging - remove early return
      console.log('updateNode:', { id, updates, existingNode });
      
      const updatedNetwork = {
        ...state.currentNetwork,
        nodes: state.currentNetwork.nodes.map(node =>
          node.id === id ? { ...node, ...updates } : node
        ),
      };
      
      // Save positions to localStorage when position is updated
      if (updates.position) {
        const positions = updatedNetwork.nodes.reduce((acc, node) => {
          acc[node.id] = node.position;
          return acc;
        }, {} as Record<string, { x: number; y: number }>);
        
        console.log(`Saving positions for network ${updatedNetwork.id}:`, positions);
        savePositions(updatedNetwork.id, positions);
      }
      
      return {
        currentNetwork: updatedNetwork,
        selectedNode: state.selectedNode?.id === id ? { ...state.selectedNode, ...updates } : state.selectedNode,
        isDirty: updates.position ? false : true, // Don't mark dirty for position updates (UI-only)
      };
    });
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
  addLink: (linkData: Omit<ReCoNLink, 'id'>) => {
    const { currentNetwork } = get();
    if (!currentNetwork) return;

    const newLink: ReCoNLink = {
      id: `${linkData.source}-${linkData.target}-${linkData.type}`,
      source: linkData.source,
      target: linkData.target,
      type: linkData.type as any,
      weight: linkData.weight,
    };

    // Determine reciprocal for bidirectional pairs
    let reciprocal: ReCoNLink | null = null;
    if (linkData.type === 'sub') {
      reciprocal = {
        id: `${linkData.target}-${linkData.source}-sur`,
        source: linkData.target,
        target: linkData.source,
        type: 'sur',
        weight: linkData.weight,
      } as ReCoNLink;
    } else if (linkData.type === 'sur') {
      reciprocal = {
        id: `${linkData.target}-${linkData.source}-sub`,
        source: linkData.target,
        target: linkData.source,
        type: 'sub',
        weight: linkData.weight,
      } as ReCoNLink;
    } else if (linkData.type === 'por') {
      reciprocal = {
        id: `${linkData.target}-${linkData.source}-ret`,
        source: linkData.target,
        target: linkData.source,
        type: 'ret',
        weight: linkData.weight,
      } as ReCoNLink;
    } else if (linkData.type === 'ret') {
      reciprocal = {
        id: `${linkData.target}-${linkData.source}-por`,
        source: linkData.target,
        target: linkData.source,
        type: 'por',
        weight: linkData.weight,
      } as ReCoNLink;
    }

    // Add both links locally - no server calls
    set(state => {
      const existing = new Set(state.currentNetwork?.links.map(l => l.id));
      const newLinks: ReCoNLink[] = [newLink];
      if (reciprocal && !existing.has(reciprocal.id)) newLinks.push(reciprocal);
      return {
        currentNetwork: state.currentNetwork ? {
          ...state.currentNetwork,
          links: [...state.currentNetwork.links, ...newLinks],
        } : null,
        isDirty: true,
      };
    });
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

  // Execution operations removed - handled by ControlPanel directly

  // All execution functions removed - ControlPanel handles execution directly

  exportLocalGraph: () => {
    const { currentNetwork } = get();
    if (!currentNetwork) return null;
    return {
      nodes: currentNetwork.nodes.map(n => ({
        id: n.id,
        type: n.type,
        state: n.state,
        activation: n.activation,
        measurementValue: (n as any).measurementValue,
        gates: { sub: 0, sur: 0, por: 0, ret: 0, gen: 0 },
        timing_config: {
          timing_mode: 'discrete',
          discrete_wait_steps: 3,
          sequence_wait_steps: 6,
          activation_decay_rate: 0.8,
          activation_failure_threshold: 0.1,
          activation_initial_level: 0.8,
          current_waiting_activation: 0,
        },
      })),
      links: currentNetwork.links.map(l => ({
        source: l.source,
        target: l.target,
        type: l.type,
        weight: l.weight,
      })),
      requested_roots: [],
      step_count: 0,
    } as any;
  },

  // UI state
  setDirty: (dirty: boolean) => {
    set({ isDirty: dirty });
  },

  clearSelection: () => {
    set({ selectedNode: null, selectedLink: null });
  },

  setShouldRelayout: (should: boolean) => {
    set({ shouldRelayout: should });
  },

  saveCurrentPositions: () => {
    const { currentNetwork } = get();
    if (!currentNetwork) return;
    
    const positions = currentNetwork.nodes.reduce((acc, node) => {
      acc[node.id] = node.position;
      return acc;
    }, {} as Record<string, { x: number; y: number }>);
    
    savePositions(currentNetwork.id, positions);
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