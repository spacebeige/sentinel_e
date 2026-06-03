import React, { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';
import { Activity, BarChart3, Zap, Brain, X } from 'lucide-react';
import api from '../services/api';

const NerdMode = ({ sessionId, onClose }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const graphRef = useRef(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await api.get(`/api/session/${sessionId}/debug`);
        setData(res.data);
      } catch (err) {
        console.error('Failed to fetch debug data:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [sessionId]);

  useEffect(() => {
    if (!data || !data.call_graph || !graphRef.current) return;

    // D3 Graph
    const width = 600;
    const height = 400;
    const svg = d3.select(graphRef.current)
      .attr('width', width)
      .attr('height', height);
    
    svg.selectAll('*').remove();

    const nodes = data.call_graph.map(n => ({ id: n.call_id, ...n }));
    const links = [];
    data.call_graph.forEach(n => {
      n.depends_on.forEach(dep => {
        links.push({ source: dep, target: n.call_id });
      });
    });

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2));

    const link = svg.append('g')
      .selectAll('line')
      .data(links)
      .enter().append('line')
      .attr('stroke', '#4b5563')
      .attr('stroke-width', 2);

    const node = svg.append('g')
      .selectAll('circle')
      .data(nodes)
      .enter().append('circle')
      .attr('r', 8)
      .attr('fill', d => d.status === 'resolved' ? '#10b981' : '#3b82f6');

    node.append('title').text(d => `${d.endpoint} (${d.latency_ms}ms)`);

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);
    });

    return () => simulation.stop();
  }, [data]);

  if (loading) return <div className="p-8 text-white/50">Loading diagnostics...</div>;
  if (!data) return <div className="p-8 text-red-400">Failed to load diagnostics.</div>;

  return (
    <div className="fixed inset-y-0 right-0 w-[640px] bg-[#0a0f1a] border-l border-white/10 shadow-2xl z-[100] flex flex-col">
      <div className="p-4 border-b border-white/10 flex items-center justify-between bg-white/5">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-cyan-400" />
          <h2 className="text-lg font-semibold text-white">Nerd Mode Diagnostics</h2>
        </div>
        <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-lg text-white/50">
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-8">
        {/* Token & Latency Summary */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white/5 p-4 rounded-xl border border-white/10">
            <div className="flex items-center gap-2 text-white/50 text-sm mb-1">
              <Zap className="w-4 h-4" />
              Token Budget
            </div>
            <div className="text-2xl font-mono text-white">{data.token_usage} <span className="text-sm text-white/30">tokens</span></div>
          </div>
          <div className="bg-white/5 p-4 rounded-xl border border-white/10">
            <div className="flex items-center gap-2 text-white/50 text-sm mb-1">
              <BarChart3 className="w-4 h-4" />
              Total Latency
            </div>
            <div className="text-2xl font-mono text-white">{data.total_latency.toFixed(0)} <span className="text-sm text-white/30">ms</span></div>
          </div>
        </div>

        {/* GraphRAG Explorer */}
        <div>
          <h3 className="text-sm font-medium text-white/40 uppercase tracking-wider mb-4">GraphRAG Explorer</h3>
          <div className="bg-black/40 rounded-xl border border-white/10 overflow-hidden">
            <svg ref={graphRef}></svg>
          </div>
        </div>

        {/* Memory Browser */}
        <div>
          <h3 className="text-sm font-medium text-white/40 uppercase tracking-wider mb-4 flex items-center gap-2">
            <Brain className="w-4 h-4" />
            Learned Memory ({data.memory.length})
          </h3>
          <div className="space-y-2">
            {data.memory.map((mem, i) => (
              <div key={i} className="bg-white/5 p-3 rounded-lg border border-white/5 flex items-center justify-between">
                <div>
                  <div className="text-xs font-mono text-cyan-400 mb-1">{mem.key}</div>
                  <div className="text-sm text-white">{mem.value}</div>
                </div>
                <div className="text-xs text-white/30 font-mono">{mem.confidence}% conf</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NerdMode;
