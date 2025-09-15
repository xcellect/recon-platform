// Auto-layout algorithms for ReCoN networks

import { ReCoNNode, ReCoNLink } from '../types/recon';

export interface LayoutOptions {
  nodeWidth: number;
  nodeHeight: number;
  horizontalSpacing: number;
  verticalSpacing: number;
}

const defaultLayoutOptions: LayoutOptions = {
  nodeWidth: 120,
  nodeHeight: 60,
  horizontalSpacing: 180,
  verticalSpacing: 100,
};

export function hierarchicalLayout(
  nodes: ReCoNNode[],
  links: ReCoNLink[],
  options: Partial<LayoutOptions> = {}
): ReCoNNode[] {
  const opts = { ...defaultLayoutOptions, ...options };

  // Build adjacency lists for hierarchy detection
  const children: Record<string, string[]> = {};
  const parents: Record<string, string[]> = {};

  links.forEach(link => {
    if (link.type === 'sub') {
      if (!children[link.source]) children[link.source] = [];
      children[link.source].push(link.target);

      if (!parents[link.target]) parents[link.target] = [];
      parents[link.target].push(link.source);
    }
  });

  // Find root nodes (nodes with no parents)
  const roots = nodes.filter(node => !parents[node.id] || parents[node.id].length === 0);

  // Assign levels using BFS
  const levels: Record<string, number> = {};
  const nodesByLevel: Record<number, string[]> = {};

  const queue: Array<{ id: string; level: number }> = roots.map(node => ({ id: node.id, level: 0 }));

  while (queue.length > 0) {
    const { id, level } = queue.shift()!;

    if (levels[id] === undefined || level > levels[id]) {
      levels[id] = level;

      if (!nodesByLevel[level]) nodesByLevel[level] = [];
      nodesByLevel[level].push(id);

      // Add children to queue
      if (children[id]) {
        children[id].forEach(childId => {
          queue.push({ id: childId, level: level + 1 });
        });
      }
    }
  }

  // Handle nodes without hierarchical connections
  nodes.forEach(node => {
    if (levels[node.id] === undefined) {
      levels[node.id] = 0;
      if (!nodesByLevel[0]) nodesByLevel[0] = [];
      nodesByLevel[0].push(node.id);
    }
  });

  // Position nodes
  const positionedNodes = nodes.map(node => {
    const level = levels[node.id];
    const nodesAtLevel = nodesByLevel[level];
    const indexAtLevel = nodesAtLevel.indexOf(node.id);

    // Center nodes horizontally at each level
    const totalWidth = (nodesAtLevel.length - 1) * opts.horizontalSpacing;
    const startX = -totalWidth / 2;

    return {
      ...node,
      position: {
        x: startX + indexAtLevel * opts.horizontalSpacing,
        y: level * opts.verticalSpacing,
      },
    };
  });

  return positionedNodes;
}

export function sequenceLayout(
  nodes: ReCoNNode[],
  links: ReCoNLink[],
  options: Partial<LayoutOptions> = {}
): ReCoNNode[] {
  const opts = { ...defaultLayoutOptions, ...options };

  // Build sequence chains from por/ret links
  const nextInSequence: Record<string, string> = {};
  const prevInSequence: Record<string, string> = {};

  links.forEach(link => {
    if (link.type === 'por') {
      nextInSequence[link.source] = link.target;
      prevInSequence[link.target] = link.source;
    }
  });

  // Find sequence starts (nodes with no predecessor)
  const sequenceStarts = nodes.filter(node => !prevInSequence[node.id]);

  // Position nodes in sequences
  const positioned = new Set<string>();
  const positionedNodes = [...nodes];

  sequenceStarts.forEach((startNode, sequenceIndex) => {
    let currentNode = startNode.id;
    let positionInSequence = 0;

    while (currentNode && !positioned.has(currentNode)) {
      const nodeIndex = nodes.findIndex(n => n.id === currentNode);
      if (nodeIndex >= 0) {
        positionedNodes[nodeIndex] = {
          ...positionedNodes[nodeIndex],
          position: {
            x: positionInSequence * opts.horizontalSpacing,
            y: sequenceIndex * opts.verticalSpacing,
          },
        };
        positioned.add(currentNode);
      }

      currentNode = nextInSequence[currentNode];
      positionInSequence++;
    }
  });

  // Position remaining nodes
  let unpositionedCount = 0;
  positionedNodes.forEach((node, index) => {
    if (!positioned.has(node.id)) {
      positionedNodes[index] = {
        ...node,
        position: {
          x: unpositionedCount * opts.horizontalSpacing,
          y: (sequenceStarts.length + 1) * opts.verticalSpacing,
        },
      };
      unpositionedCount++;
    }
  });

  return positionedNodes;
}

export function autoLayout(
  nodes: ReCoNNode[],
  links: ReCoNLink[],
  options: Partial<LayoutOptions> = {}
): ReCoNNode[] {
  // Detect if network is primarily hierarchical or sequential
  const hierarchicalLinks = links.filter(link => link.type === 'sub' || link.type === 'sur');
  const sequentialLinks = links.filter(link => link.type === 'por' || link.type === 'ret');

  // Use hierarchical layout if we have more hierarchical links, otherwise use sequence layout
  if (hierarchicalLinks.length >= sequentialLinks.length) {
    return hierarchicalLayout(nodes, links, options);
  } else {
    return sequenceLayout(nodes, links, options);
  }
}

export function forceDirectedLayout(
  nodes: ReCoNNode[],
  links: ReCoNLink[],
  options: Partial<LayoutOptions> = {}
): ReCoNNode[] {
  // Simple force-directed layout for complex mixed networks
  const opts = { ...defaultLayoutOptions, ...options };

  // Start with random positions if nodes don't have positions
  let positionedNodes = nodes.map((node, index) => ({
    ...node,
    position: node.position.x === 0 && node.position.y === 0 ? {
      x: (Math.random() - 0.5) * 1000,
      y: (Math.random() - 0.5) * 1000,
    } : node.position,
  }));

  // Simple spring forces
  const iterations = 50;
  const repulsionStrength = 10000;
  const attractionStrength = 0.01;
  const damping = 0.9;

  for (let iter = 0; iter < iterations; iter++) {
    const forces: Record<string, { x: number; y: number }> = {};

    // Initialize forces
    positionedNodes.forEach(node => {
      forces[node.id] = { x: 0, y: 0 };
    });

    // Repulsion between all nodes
    for (let i = 0; i < positionedNodes.length; i++) {
      for (let j = i + 1; j < positionedNodes.length; j++) {
        const nodeA = positionedNodes[i];
        const nodeB = positionedNodes[j];

        const dx = nodeA.position.x - nodeB.position.x;
        const dy = nodeA.position.y - nodeB.position.y;
        const distance = Math.sqrt(dx * dx + dy * dy) || 1;

        const force = repulsionStrength / (distance * distance);
        const forceX = (dx / distance) * force;
        const forceY = (dy / distance) * force;

        forces[nodeA.id].x += forceX;
        forces[nodeA.id].y += forceY;
        forces[nodeB.id].x -= forceX;
        forces[nodeB.id].y -= forceY;
      }
    }

    // Attraction along links
    links.forEach(link => {
      const sourceNode = positionedNodes.find(n => n.id === link.source);
      const targetNode = positionedNodes.find(n => n.id === link.target);

      if (sourceNode && targetNode) {
        const dx = targetNode.position.x - sourceNode.position.x;
        const dy = targetNode.position.y - sourceNode.position.y;
        const distance = Math.sqrt(dx * dx + dy * dy) || 1;

        const force = distance * attractionStrength;
        const forceX = (dx / distance) * force;
        const forceY = (dy / distance) * force;

        forces[sourceNode.id].x += forceX;
        forces[sourceNode.id].y += forceY;
        forces[targetNode.id].x -= forceX;
        forces[targetNode.id].y -= forceY;
      }
    });

    // Apply forces
    positionedNodes = positionedNodes.map(node => ({
      ...node,
      position: {
        x: node.position.x + forces[node.id].x * damping,
        y: node.position.y + forces[node.id].y * damping,
      },
    }));
  }

  return positionedNodes;
}