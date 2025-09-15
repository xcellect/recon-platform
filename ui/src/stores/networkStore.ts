// Zustand store for ReCoN network state management

import { create } from 'zustand';
import { ReCoNNetwork, ReCoNNode, ReCoNLink, NetworkState, ExecutionState } from '../types/recon';
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

  // Network operations
  createNetwork: async (id?: string) => {
    try {
      const response = await reconAPI.createNetwork(id);
      const storedPositions = getNetworkPositions(response.network_id);
      
      const network: ReCoNNetwork = {
        id: response.network_id,
        nodes: response.nodes.map(node => ({
          id: node.node_id,
          type: node.node_type as any,
          state: node.state as any,
          activation: node.activation,
          // Restore position from localStorage or default to (0,0) for layout
          position: storedPositions[node.node_id] || { x: 0, y: 0 },
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
      console.error('Failed to create network:', error);
      throw error;
    }
  },

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
    if (!currentNetwork) return;

    try {
      const response = await reconAPI.createNode(currentNetwork.id, {
        node_id: nodeData.id,
        node_type: nodeData.type,
      });

      const newNode: ReCoNNode = {
        id: response.node_id,
        type: response.node_type,
        state: response.state as any,
        activation: response.activation,
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
    } catch (error) {
      console.error('Failed to add node:', error);
      throw error;
    }
  },

  updateNode: (id: string, updates: Partial<ReCoNNode>) => {
    set(state => {
      if (!state.currentNetwork) return state;
      
      const existingNode = state.currentNetwork.nodes.find(node => node.id === id);
      if (!existingNode) return state;
      
      // Check if the update actually changes anything
      const hasChanges = Object.keys(updates).some(key => {
        const updateKey = key as keyof ReCoNNode;
        if (updateKey === 'position') {
          return existingNode.position.x !== updates.position?.x || 
                 existingNode.position.y !== updates.position?.y;
        }
        return existingNode[updateKey] !== updates[updateKey];
      });
      
      if (!hasChanges) return state; // No changes, return current state
      
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
  addLink: async (linkData: Omit<ReCoNLink, 'id'>) => {
    const { currentNetwork } = get();
    if (!currentNetwork) return;

    try {
      const response = await reconAPI.createLink(currentNetwork.id, {
        source: linkData.source,
        target: linkData.target,
        link_type: linkData.type,
        weight: linkData.weight,
      });

      const newLink: ReCoNLink = {
        id: `${response.source}-${response.target}-${response.link_type}`,
        source: response.source,
        target: response.target,
        type: response.link_type as any,
        weight: response.weight,
      };

      set(state => ({
        currentNetwork: state.currentNetwork ? {
          ...state.currentNetwork,
          links: [...state.currentNetwork.links, newLink],
        } : null,
        isDirty: true,
      }));
    } catch (error) {
      console.error('Failed to add link:', error);
      throw error;
    }
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
        shouldRelayout: true, // Trigger relayout after reset
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