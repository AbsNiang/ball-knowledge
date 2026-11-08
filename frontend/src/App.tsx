import { useEffect, useMemo, useRef, useState } from 'react';
import { ReactFlow, Background, Controls, MiniMap, Handle, Position, type NodeProps, type Node as FlowNode } from '@xyflow/react';
import dagre from '@dagrejs/dagre';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ResponsiveContainer, BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { ArrowUpRight, Plus, PanelLeftClose, PanelLeftOpen, FlaskConical, Pause, Play, GitBranch, X, ArrowUp, Check, Circle, RefreshCw, Radio, ChevronRight } from 'lucide-react';
import type { Node as ResearchNode } from '../../contracts/generated';
import { newSession, isActive, type Session } from './state';
import { RunStream, type ConnectionState } from './stream';

function ResearchCard({data}: NodeProps<FlowNode<{record:ResearchNode; paused:boolean}>>) {
  const n=data.record;
  const state=n.flags?.includes('pruned')?'interrupted':n.flags?.includes('tainted')?'tainted':n.lifecycle==='running'?(data.paused?'paused':'running'):n.check_outcome;
  return <article className={`research-node ${state}`}><Handle type="target" position={Position.Top}/><div className="node-eyebrow"><FlaskConical size={13}/>{n.type.replaceAll('_',' ')}<span className="node-id">{n.id.slice(0,7)}</span></div><h3>{n.title}</h3><div className="node-bottom"><span className={`status ${state}`}>{state==='passed'?<Check size={12}/>:<Circle size={10}/>} {state}</span><span>Gate <b>{n.check_outcome === 'pending'?'pending':n.check_outcome}</b></span></div><Handle type="source" position={Position.Bottom}/></article>;
}
const nodeTypes={research:ResearchCard};
function Graph({session,onSelect}: {session:Session;onSelect:(id:string)=>void}) {
  const graph = useMemo(()=>{
    const g=new dagre.graphlib.Graph().setGraph({rankdir:'TB',nodesep:65,ranksep:110}).setDefaultEdgeLabel(()=>({}));
    Object.values(session.nodes).forEach(n=>g.setNode(n.id,{width:288,height:158}));
    const edges=Object.values(session.edges).filter(e=>session.nodes[e.source]&&session.nodes[e.target]).map(e=>({id:e.id,source:['derives_from','revises'].includes(e.type)?e.target:e.source,target:['derives_from','revises'].includes(e.type)?e.source:e.target,label:e.label||e.type.replaceAll('_',' '),type:'smoothstep',style:{stroke:'#4b595d',strokeWidth:1.3},labelStyle:{fill:'#aebbbd',fontSize:10},labelBgStyle:{fill:'#151d20'},labelBgPadding:[8,5] as [number,number]}));
    edges.forEach(e=>g.setEdge(e.source,e.target)); dagre.layout(g);
    return {nodes:Object.values(session.nodes).map(n=>({id:n.id,type:'research',position:{x:g.node(n.id).x-144,y:g.node(n.id).y-79},data:{record:n,paused:session.status==='paused'}})),edges};
  },[session.nodes,session.edges,session.status]);
  return <ReactFlow nodes={graph.nodes} edges={graph.edges} nodeTypes={nodeTypes} onNodeClick={(_,n)=>onSelect(n.id)} fitView minZoom={0.2} maxZoom={1.6} nodesDraggable={false} nodesConnectable={false} deleteKeyCode={null} colorMode="dark"><Background color="#324044" gap={24} size={1}/><Controls showInteractive={false}/><MiniMap pannable zoomable nodeColor="#506863" maskColor="#0c1219aa"/></ReactFlow>;
}
interface ChartSpec {type:'bar'|'line';title:string;xKey:string;yKey:string;data:Record<string,string|number>[]}
function chartsFrom(n:ResearchNode):ChartSpec[] {
  const raw=n.tool?.output?.charts;
  if(!Array.isArray(raw)) return [];
  return raw.filter((c):c is ChartSpec=>!!c && ['bar','line'].includes(c.type) && typeof c.title==='string' && typeof c.xKey==='string' && typeof c.yKey==='string' && Array.isArray(c.data) && c.data.every((r:Record<string,unknown>)=>r && ['string','number'].includes(typeof r[c.xKey]) && typeof r[c.yKey]==='number' && Number.isFinite(r[c.yKey])));
}
function Inspector({node,onClose}:{node:ResearchNode;onClose:()=>void}) {
  const [tab,setTab]=useState('Research');
  return <aside className="inspector"><div className="panel-heading"><span>STEP INSPECTOR</span><button aria-label="Close inspector" onClick={onClose}><X size={18}/></button></div><span className="eyebrow">{node.type.replaceAll('_',' ')}</span><h2>{node.title}</h2><div className="tabs">{['Research','Evidence','Checks'].map(t=><button key={t} className={tab===t?'selected':''} onClick={()=>setTab(t)}>{t}</button>)}</div><div className="inspector-content">{tab==='Research'&&<div className="markdown"><Markdown remarkPlugins={[remarkGfm]}>{node.llm?.rationale||'Research details will appear as the agent publishes this step.'}</Markdown>{node.verdict&&<blockquote>{node.verdict.label}</blockquote>}{node.human?.map((h,i)=><blockquote key={i}>{h.note}</blockquote>)}</div>}{tab==='Checks'&&<>{!(node.checks?.length)&&<p className="muted">Waiting for backend gate results.</p>}{node.checks?.map(c=><div className="check-card" key={c.id}><span className={`status ${c.status==='pass'?'passed':'failed'}`}>{c.gate_id||c.checker} · {c.status}</span><p>{c.rationale}</p></div>)}</>}{tab==='Evidence'&&<>{chartsFrom(node).map((c,i)=><div className="chart" key={i}><h3>{c.title}</h3><ResponsiveContainer width="100%" height={220}>{c.type==='bar'?<BarChart data={c.data}><CartesianGrid stroke="#303d40" vertical={false}/><XAxis dataKey={c.xKey}/><YAxis/><Tooltip contentStyle={{background:'#202b2e',border:'1px solid #435459'}}/><Bar dataKey={c.yKey} fill="#94d6b9" radius={[3,3,0,0]}/></BarChart>:<LineChart data={c.data}><XAxis dataKey={c.xKey}/><YAxis/><Tooltip/><Line dataKey={c.yKey} stroke="#94d6b9"/></LineChart>}</ResponsiveContainer></div>)}{!node.tool?<p className="muted">Evidence arrives after a tool completes.</p>:<details><summary>Tool payload · {node.tool.tool_name}</summary><pre>{JSON.stringify(node.tool.output,null,2)}</pre></details>}</>}</div></aside>;
}
const storageKey='whack.sessions.v1';
function savedSessions():Session[] {try {return JSON.parse(localStorage.getItem(storageKey)||'[]').filter((s:Session)=>typeof s.id==='string'&&typeof s.question==='string').map((s:Session)=>newSession(s.id,s.question,s.demo));}catch{return [];}}
export default function App(){
  const [sessions,setSessions]=useState<Session[]>(savedSessions);
  const [selected,setSelected]=useState<string|null>(()=>sessions[0]?.id??null);
  const [collapsed,setCollapsed]=useState(false),[inspect,setInspect]=useState<string|null>(null),[question,setQuestion]=useState(''),[demo,setDemo]=useState(true),[busy,setBusy]=useState(false),[error,setError]=useState(''),[redirect,setRedirect]=useState(false);
  const [connections,setConnections]=useState<Record<string,ConnectionState>>({});
  const sockets=useRef(new Map<string,RunStream>());
  const pending=useRef(new Map<string,ReturnType<typeof setTimeout>>());
  const current=sessions.find(s=>s.id===selected);
  const activeNode=current&&Object.values(current.nodes).find(n=>n.lifecycle==='running'&&!n.flags?.includes('pruned'));
  const activeSession=sessions.find(isActive);
  const disconnected=sessions.some(s=>connections[s.id]==='failed');
  const connected=!!current&&connections[current.id]==='connected';
  const reconnecting=sessions.some(s=>connections[s.id]==='reconnecting');
  const ids=sessions.map(s=>s.id).join('|');
  useEffect(()=>{localStorage.setItem(storageKey,JSON.stringify(sessions.map(({id,question,demo})=>({id,question,demo}))));},[sessions]);
  useEffect(()=>{
    const wanted=new Set(sessions.map(s=>s.id));
    sockets.current.forEach((stream,id)=>{if(!wanted.has(id)){stream.close();sockets.current.delete(id);}});
    sessions.forEach(session=>{
      const id=session.id;
      if(sockets.current.has(id))return;
      sockets.current.set(id,new RunStream(session,{
        session:next=>setSessions(all=>all.map(s=>s.id===id?next:s)),
        connection:state=>setConnections(all=>({...all,[id]:state})),
        missing:()=>{
          setSessions(all=>all.filter(s=>s.id!==id));
          setSelected(value=>value===id?null:value);
          setError('A saved session is no longer on the backend, which may have restarted. You can start a new research session.');
          setBusy(false);
        },
        error:setError,
        command:message=>{
          if(!pending.current.has(message.request_id))return;
          clearTimeout(pending.current.get(message.request_id));pending.current.delete(message.request_id);setBusy(false);
          if(message.type==='command_error')setError(message.detail?.message||'Command rejected.');
        },
      }));
    });
  },[ids]);
  useEffect(()=>()=>{sockets.current.forEach(stream=>stream.close());sockets.current.clear();pending.current.forEach(clearTimeout);pending.current.clear();},[]);
  useEffect(()=>{document.querySelectorAll<HTMLElement>('.sidebar,.workspace,.inspector').forEach(el=>{el.inert=disconnected;});},[disconnected]);
  async function start(){
    if(!question.trim()||busy||activeSession)return;
    setBusy(true);setError('');
    try {const response=await fetch(`/api/${demo?'demo/':''}runs`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:question.trim(),mode:'live'})});const body=await response.json();if(!response.ok)throw new Error(body.detail?.message||'Unable to start research');setSessions(s=>[...s,newSession(body.run_id,question.trim(),demo)]);setSelected(body.run_id);setQuestion('');setInspect(null);}catch(e){setError(e instanceof Error?e.message:'Unable to start');}finally{setBusy(false);}
  }
  function command(type:'pause'|'resume'|'redirect'){
    if(!current||busy||(type==='redirect'&&(!question.trim()||!activeNode)))return;
    const ws=sockets.current.get(current.id);if(!connected||!ws){setError('Waiting for the session connection. Try again once reconnected.');return;}
    const request_id=crypto.randomUUID();setBusy(true);setError('');
    pending.current.set(request_id,setTimeout(()=>{pending.current.delete(request_id);setBusy(false);setError('No acknowledgement received. Check the session state before retrying.');},10000));
    if(!ws.send({request_id,command:{type,node_id:type==='redirect'?activeNode?.id:null,payload:type==='redirect'?{statement:question.trim()}:{}}})){
      clearTimeout(pending.current.get(request_id));pending.current.delete(request_id);setBusy(false);setError('Connection lost before the command was sent.');return;
    }
    if(type==='redirect'){setRedirect(false);setQuestion('');}
  }
  return <main className="flex h-dvh overflow-hidden bg-[#11191c] text-[#e1e8e5]">
    <aside className={`sidebar ${collapsed?'collapsed':''}`}><div className="brand"><div className="brand-icon"><GitBranch size={23}/></div>{!collapsed&&<span>whack<span className="brand-dot">.</span></span>}<button onClick={()=>setCollapsed(!collapsed)} aria-label="Toggle session sidebar">{collapsed?<PanelLeftOpen size={17}/>:<PanelLeftClose size={17}/>}</button></div>{!collapsed&&<><button className="new-session" onClick={()=>{setSelected(null);setInspect(null);setRedirect(false);setQuestion('');}}><Plus size={16}/> New research <span>?</span></button><div className="section-label">SESSION TABS <span>{sessions.length.toString().padStart(2,'0')}</span></div><nav aria-label="Research sessions">{sessions.map(s=><button className={`session ${selected===s.id?'selected':''}`} key={s.id} onClick={()=>{setSelected(s.id);setInspect(null);setRedirect(false);setQuestion('');}}><div><span className={`session-dot ${s.status}`}/><span>{s.question}</span></div><small>{s.demo?'DEMO':'LIVE'} <span>· {s.status}</span></small></button>)}{!sessions.length&&<p className="sidebar-hint">Your research sessions<br/>will appear here.</p>}</nav><div className="sidebar-bottom"><div className="avatar">W</div><div>Research workspace<small>Every path leaves evidence.</small></div></div></>}</aside>
    <section className="workspace"><header className="topbar"><div className="breadcrumb">Workspace <ChevronRight size={14}/><span>{current?'Research graph':'New research'}</span></div><span className="connection"><span className={connected?'online':''}/>{reconnecting?'Reconnecting...':connected?'Stream connected':current?'Connecting...':'Ready to explore'}</span></header><div className="canvas-header"><div><div className="eyebrow">AUTONOMOUS RESEARCH / {current?.demo?'SCRIPTED DEMO':'WORKSPACE'}</div><h1>{current?.question||'Follow the question.'}</h1><p>{current?'':'Ask a question. Watch the evidence take shape.'}</p></div>{current&&<div className="run-controls"><span className={`status ${current.status}`}><Radio size={13}/>{current.status}</span>{['running','paused'].includes(current.status)&&<button className="control-button" disabled={busy||!connected||(current.status==='paused'&&!!activeSession)} onClick={()=>command(current.status==='paused'?'resume':'pause')}>{current.status==='paused'?<Play size={15}/>:<Pause size={15}/>} {current.status==='paused'?'Resume':'Pause'}</button>}</div>}</div>
    <div className="canvas">{current?<Graph key={current.id} session={current} onSelect={setInspect}/>:<div className="empty-state"><div className="empty-orbit"><GitBranch size={40}/></div><span className="eyebrow">A QUESTION IS JUST THE BEGINNING</span><h2>Make the reasoning visible.</h2><p>Explore competing ideas, challenge assumptions,<br/>and see which paths survive the evidence.</p><button className="suggestion" onClick={()=>setQuestion('Why do NBA teams underperform on back-to-back games?')}><FlaskConical size={18}/><span>Why do teams underperform<br/>on back-to-back games?</span><ArrowUpRight size={18}/></button></div>}<div className="canvas-legend"><span><i className="running"/>Researching</span><span><i className="passed"/>Passed</span><span><i className="failed"/>Failed / retained</span></div></div>
    <div className="composer-area">{(error||current?.error)&&<div className="error-message" role="alert">{error||current?.error}<button onClick={()=>setError('')} aria-label="Dismiss error"><X size={14}/></button></div>}{!current||redirect?<form className="composer" onSubmit={e=>{e.preventDefault();redirect?command('redirect'):start();}}><div className="composer-label">{redirect?<><GitBranch size={14}/> Branch from the active step <button type="button" onClick={()=>setRedirect(false)} aria-label="Cancel redirect"><X size={14}/></button></>:<><FlaskConical size={14}/> WHAT WOULD YOU LIKE TO INVESTIGATE?</>}</div><div className="input-row"><textarea aria-label={redirect?'New research direction':'Research question'} placeholder={redirect?'Ask a new question to redirect this path…':'Ask a research question…'} value={question} maxLength={4000} onChange={e=>setQuestion(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();redirect?command('redirect'):start();}}}/><button className="send" aria-label={redirect?'Redirect research':'Start research'} disabled={busy||!question.trim()||(!redirect&&!!activeSession)}><ArrowUp size={20}/></button></div><div className="composer-footer">{!redirect&&<label><input type="checkbox" checked={demo} onChange={e=>setDemo(e.target.checked)}/> Scripted demo</label>}<span>{activeSession&&!redirect?'Pause the active session to start another.':demo?'Synthetic evidence · no LLM calls':'Experimental agent: synthetic tools and gates'}</span></div></form>:<div className="watch-bar"><span><span className="live-dot"/>{current.status==='completed'?'Research complete. Select a node to inspect its evidence.':current.status==='paused'?'Research paused. Resume or explore a new direction.':'Follow the evidence. You can change direction at any time.'}</span><button disabled={!current.demo||!activeNode||!connected||busy||!!(activeSession&&activeSession.id!==current.id)} onClick={()=>{setRedirect(true);setQuestion('');}}><GitBranch size={15}/> Redirect</button></div>}<div className="bottom-note">{current?.demo?'DEMO · All results are synthetic':current?'EXPERIMENTAL AGENT: synthetic tools and gates':'WHACK · An inspectable research workspace'}<span>Failed paths are always preserved.</span></div></div></section>{current&&inspect&&current.nodes[inspect]&&<Inspector key={inspect} node={current.nodes[inspect]} onClose={()=>setInspect(null)}/>}
    {disconnected&&<div className="disconnect-backdrop"><section role="alertdialog" aria-modal="true" aria-labelledby="disconnect-title" className="disconnect-modal"><div className="disconnect-icon"><Radio size={26}/></div><h2 id="disconnect-title">Connection interrupted</h2><p>{error||'Unable to reach the event stream. Check that both servers are running, then retry.'}</p><button autoFocus className="primary-button" onClick={()=>{setError('');sockets.current.forEach(stream=>stream.retry());}}><RefreshCw size={16}/> Retry connection</button><small>Existing evidence is retained while reconnecting.</small><button className="reset-history" onClick={()=>{setSessions([]);setSelected(null);setConnections({});setError('');setBusy(false);}}>Clear local session list</button></section></div>}
  </main>;
}
