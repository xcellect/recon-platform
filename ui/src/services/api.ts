// API client for ReCoN Platform backend

const API_BASE_URL = 'http://localhost:8000';

export interface NodeCreateRequest {
  node_id: string;
  node_type: 'script' | 'terminal';
}

export interface LinkCreateRequest {
  source: string;
  target: string;
  link_type: 'sub' | 'sur' | 'por' | 'ret' | 'gen';
  weight: number;
}

export interface ExecuteRequest {
  root_node: string;
  max_steps: number;
}

export interface NetworkResponse {
  network_id: string;
  nodes: Array<{
    node_id: string;
    node_type: string;
    state: string;
    activation: number;
  }>;
  links: Array<{
    source: string;
    target: string;
    link_type: string;
    weight: number;
  }>;
  step_count: number;
}

export interface ExecuteResponse {
  network_id: string;
  root_node: string;
  result: string;
  final_states: Record<string, string>;
  steps_taken: number;
}

export interface ExecutionHistoryResponse {
  network_id: string;
  root_node: string;
  result: string;
  steps: Array<{
    step: number;
    states: Record<string, string>;
    messages: Array<{
      type: string;
      from: string;
      to: string;
      link: string;
    }>;
  }>;
  final_state: string;
  total_steps: number;
}

export interface VisualizationResponse {
  network_id: string;
  snapshot: {
    nodes: Record<string, {
      id: string;
      type: string;
      state: string;
      activation: number;
      gates: Record<string, number>;
    }>;
    step: number;
    requested_roots: string[];
    messages: number;
  };
}

class ReCoNAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async createNetwork(networkId?: string): Promise<NetworkResponse> {
    const response = await fetch(`${this.baseUrl}/networks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ network_id: networkId }),
    });

    if (!response.ok) {
      throw new Error(`Failed to create network: ${response.statusText}`);
    }

    return response.json();
  }

  async getNetwork(networkId: string): Promise<NetworkResponse> {
    const response = await fetch(`${this.baseUrl}/networks/${networkId}`);

    if (!response.ok) {
      throw new Error(`Failed to get network: ${response.statusText}`);
    }

    return response.json();
  }

  async listNetworks(): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/networks`);

    if (!response.ok) {
      throw new Error(`Failed to list networks: ${response.statusText}`);
    }

    return response.json();
  }

  async deleteNetwork(networkId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/networks/${networkId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      throw new Error(`Failed to delete network: ${response.statusText}`);
    }
  }

  async createNode(networkId: string, nodeData: NodeCreateRequest) {
    const response = await fetch(`${this.baseUrl}/networks/${networkId}/nodes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(nodeData),
    });

    if (!response.ok) {
      throw new Error(`Failed to create node: ${response.statusText}`);
    }

    return response.json();
  }

  async createLink(networkId: string, linkData: LinkCreateRequest) {
    const response = await fetch(`${this.baseUrl}/networks/${networkId}/links`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(linkData),
    });

    if (!response.ok) {
      throw new Error(`Failed to create link: ${response.statusText}`);
    }

    return response.json();
  }

  async executeScript(networkId: string, executeData: ExecuteRequest): Promise<ExecuteResponse> {
    const response = await fetch(`${this.baseUrl}/networks/${networkId}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(executeData),
    });

    if (!response.ok) {
      throw new Error(`Failed to execute script: ${response.statusText}`);
    }

    return response.json();
  }

  async executeNetworkDirect(networkData: any, rootNode: string, maxSteps: number = 100, terminalConfigs: any[] = []): Promise<ExecutionHistoryResponse> {
    const requestBody = {
      network_data: networkData,
      root_node: rootNode,
      max_steps: maxSteps,
      terminal_configs: terminalConfigs
    };
    
    console.log('Sending direct execution request:', requestBody);
    
    const response = await fetch(`${this.baseUrl}/execute-network`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Execute network failed:', response.status, errorText);
      throw new Error(`Failed to execute network: ${response.statusText}`);
    }

    return response.json();
  }

  async executeScriptWithHistory(networkId: string, executeData: ExecuteRequest): Promise<ExecutionHistoryResponse> {
    const response = await fetch(`${this.baseUrl}/networks/${networkId}/execute-with-history`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(executeData),
    });

    if (!response.ok) {
      throw new Error(`Failed to execute script with history: ${response.statusText}`);
    }

    return response.json();
  }

  async requestNode(networkId: string, nodeId: string) {
    const response = await fetch(`${this.baseUrl}/networks/${networkId}/request/${nodeId}`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to request node: ${response.statusText}`);
    }

    return response.json();
  }

  async propagateStep(networkId: string) {
    const response = await fetch(`${this.baseUrl}/networks/${networkId}/step`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to propagate step: ${response.statusText}`);
    }

    return response.json();
  }

  async resetNetwork(networkId: string) {
    const response = await fetch(`${this.baseUrl}/networks/${networkId}/reset`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to reset network: ${response.statusText}`);
    }

    return response.json();
  }

  // exportNetwork removed - all exports are client-side now

  async importNetwork(data: any) {
    const response = await fetch(`${this.baseUrl}/networks/import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`Failed to import network: ${response.statusText}`);
    }

    return response.json();
  }

  async getVisualizationData(networkId: string): Promise<VisualizationResponse> {
    const response = await fetch(`${this.baseUrl}/networks/${networkId}/visualize`);

    if (!response.ok) {
      throw new Error(`Failed to get visualization data: ${response.statusText}`);
    }

    return response.json();
  }

  async confirmNode(networkId: string, nodeId: string) {
    const response = await fetch(`${this.baseUrl}/networks/${networkId}/nodes/${nodeId}/confirm`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to confirm node: ${response.statusText}`);
    }

    return response.json();
  }

  async failNode(networkId: string, nodeId: string) {
    const response = await fetch(`${this.baseUrl}/networks/${networkId}/nodes/${nodeId}/fail`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to fail node: ${response.statusText}`);
    }

    return response.json();
  }
}

export const reconAPI = new ReCoNAPI();