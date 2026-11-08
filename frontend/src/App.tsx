import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowUpRight, Plus, PanelLeftClose, PanelLeftOpen, FlaskConical, Pause, Play, GitBranch, X, ArrowUp, RefreshCw, Radio, ChevronRight } from 'lucide-react';
import { newSession, isActive, type Session } from './state';
import { RunStream, type ConnectionState } from './stream';
import { graphContext } from './describe';
import { Graph } from './Graph';
import { Inspector } from './Inspector';
import { Conclusion } from './Conclusion';

const storageKey='whack.sessions.v1';
function savedSessions():Session[] {try {return JSON.parse(localStorage.getItem(storageKey)||'[]').filter((s:Session)=>typeof s.id==='string'&&typeof s.question==='string').map((s:Session)=>newSession(s.id,s.question,s.demo));}catch{return [];}}
export default function App(){
  const [sessions,setSessions]=useState<Session[]>(savedSessions);
  const [selected,setSelected]=useState<string|null>(()=>sessions[0]?.id??null);
  const [collapsed,setCollapsed]=useState(false),[inspect,setInspect]=useState<string|null>(null),[question,setQuestion]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState(''),[redirect,setRedirect]=useState(false);
  const [connections,setConnections]=useState<Record<string,ConnectionState>>({});
  const sockets=useRef(new Map<string,RunStream>());
  const pending=useRef(new Map<string,ReturnType<typeof setTimeout>>());
  const current=sessions.find(s=>s.id===selected);
  const ctx=useMemo(()=>graphContext(current),[current]);
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
  const [runtime,setRuntime]=useState<import('../../contracts/generated').RuntimeInfo|null>(null);
  useEffect(()=>{fetch('/api/runtime').then(r=>{if(!r.ok)throw new Error();return r.json();}).then(setRuntime).catch(()=>setError('Unable to check research availability.'));},[]);
  const runtimeDescription=runtime?.ready?`Gemini ${runtime.model_id} · NBA data through 2021–22`:(runtime?.unavailable_reason||'Checking research availability…');
  async function start(){
    if(!question.trim()||busy||activeSession||(!demo&&!runtime?.ready))return;
    setBusy(true);setError('');
    try {const response=await fetch('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:question.trim(),mode:'live'})});const body=await response.json();if(!response.ok)throw new Error(body.detail?.message||'Unable to start research');setSessions(s=>[...s,newSession(body.run_id,question.trim(),false)]);setSelected(body.run_id);setQuestion('');setInspect(null);}catch(e){setError(e instanceof Error?e.message:'Unable to start');}finally{setBusy(false);}
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
    <aside className={`sidebar ${collapsed?'collapsed':''}`}><div className="brand"><div className="brand-icon"><GitBranch size={23}/></div>{!collapsed&&<span className="brand-name">ball knowledge<span className="brand-dot">.</span></span>}<button onClick={()=>setCollapsed(!collapsed)} aria-label="Toggle session sidebar">{collapsed?<PanelLeftOpen size={17}/>:<PanelLeftClose size={17}/>}</button></div>{!collapsed&&<><button className="new-session" onClick={()=>{setSelected(null);setInspect(null);setRedirect(false);setQuestion('');}}><Plus size={16}/> New research <span>?</span></button><div className="section-label">SESSION TABS <span>{sessions.length.toString().padStart(2,'0')}</span></div><nav aria-label="Research sessions">{sessions.map(s=><button className={`session ${selected===s.id?'selected':''}`} key={s.id} onClick={()=>{setSelected(s.id);setInspect(null);setRedirect(false);setQuestion('');}}><div><span className={`session-dot ${s.status}`}/><span>{s.question}</span></div><small>{s.demo?'DEMO':'LIVE'} <span>· {s.status}</span></small></button>)}{!sessions.length&&<p className="sidebar-hint">Your research sessions<br/>will appear here.</p>}</nav><div className="sidebar-bottom"><div className="avatar">W</div><div>Research workspace<small>Every path leaves evidence.</small></div></div></>}</aside>
    <section className="workspace"><header className="topbar"><div className="breadcrumb">Workspace <ChevronRight size={14}/><span>{current?'Research graph':'New research'}</span></div><span className="connection"><span className={connected?'online':''}/>{reconnecting?'Reconnecting...':connected?'Stream connected':current?'Connecting...':'Ready to explore'}</span></header><div className="canvas-header"><div><div className="eyebrow">AUTONOMOUS RESEARCH / {current?.demo?'SCRIPTED DEMO':'WORKSPACE'}</div><h1>{current?.question||'Follow the question.'}</h1></div>{current&&<div className="run-controls"><span className={`status ${current.status}`}><Radio size={13}/>{current.status}</span>{['running','paused'].includes(current.status)&&<button className="control-button" disabled={busy||!connected||(current.status==='paused'&&!!activeSession)} onClick={()=>command(current.status==='paused'?'resume':'pause')}>{current.status==='paused'?<Play size={15}/>:<Pause size={15}/>} {current.status==='paused'?'Resume':'Pause'}</button>}</div>}</div>
    <div className="canvas">{current?<Graph key={current.id} session={current} ctx={ctx} selected={inspect} onSelect={setInspect}/>:<div className="empty-state"><div className="empty-orbit"><GitBranch size={40}/></div><span className="eyebrow">A QUESTION IS JUST THE BEGINNING</span><h2>Make the reasoning visible.</h2><p>Explore competing ideas, challenge assumptions,<br/>and see which paths survive the evidence.</p><button className="suggestion" onClick={()=>setQuestion('Do NBA home teams win by a larger margin than away teams?')}><FlaskConical size={18}/><span>Do home teams win by<br/>a larger margin?</span><ArrowUpRight size={18}/></button></div>}{current&&<div className="canvas-legend" aria-label="Graph legend"><span><i className="dot running"/>Researching</span><span><i className="dot passed"/>Passed</span><span><i className="dot flagged"/>Flagged</span><span><i className="dot failed"/>Failed · retained</span><span><i className="line supports"/>Supports</span><span><i className="line refutes"/>Refutes</span><span><i className="line revises"/>Revises</span></div>}</div>
    <div className="composer-area">{(error||current?.error)&&<div className="error-message" role="alert">{error||current?.error}<button onClick={()=>setError('')} aria-label="Dismiss error"><X size={14}/></button></div>}{!current||redirect?<form className="composer" onSubmit={e=>{e.preventDefault();redirect?command('redirect'):start();}}><div className="composer-label">{redirect?<><GitBranch size={14}/> Branch from the active step <button type="button" onClick={()=>setRedirect(false)} aria-label="Cancel redirect"><X size={14}/></button></>:<><FlaskConical size={14}/> WHAT WOULD YOU LIKE TO INVESTIGATE?</>}</div><div className="input-row"><textarea aria-label={redirect?'New research direction':'Research question'} placeholder={redirect?'Ask a new question to redirect this path…':'Ask a research question…'} value={question} maxLength={4000} onChange={e=>setQuestion(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();redirect?command('redirect'):start();}}}/><button className="send" aria-label={redirect?'Redirect research':'Start research'} disabled={busy||!question.trim()||(!redirect&&!!activeSession)}><ArrowUp size={20}/></button></div><div className="composer-footer"><span>{activeSession&&!redirect?'Pause the active session to start another.':'Experimental agent: synthetic tools and gates'}</span></div></form>:current.status==='completed'?<Conclusion ctx={ctx} onSelect={setInspect}/>:<div className="watch-bar"><span><span className="live-dot"/>{current.status==='paused'?'Research paused. Resume or explore a new direction.':current.status==='failed'?'Research stopped with an error. Completed steps remain inspectable.':'Follow the evidence. You can change direction at any time.'}</span><button disabled={!current.demo||!activeNode||!connected||busy||!!(activeSession&&activeSession.id!==current.id)} onClick={()=>{setRedirect(true);setQuestion('');}}><GitBranch size={15}/> Redirect</button></div>}<div className="bottom-note">{current?.demo?'DEMO · All results are synthetic':current?'EXPERIMENTAL AGENT: synthetic tools and gates':'WHACK · An inspectable research workspace'}<span>Failed paths are always preserved.</span></div></div></section>{current&&inspect&&current.nodes[inspect]&&<Inspector key={inspect} node={current.nodes[inspect]} ctx={ctx} paused={current.status==='paused'} onClose={()=>setInspect(null)} onSelect={setInspect}/>}
    {disconnected&&<div className="disconnect-backdrop"><section role="alertdialog" aria-modal="true" aria-labelledby="disconnect-title" className="disconnect-modal"><div className="disconnect-icon"><Radio size={26}/></div><h2 id="disconnect-title">Connection interrupted</h2><p>{error||'Unable to reach the event stream. Check that both servers are running, then retry.'}</p><button autoFocus className="primary-button" onClick={()=>{setError('');sockets.current.forEach(stream=>stream.retry());}}><RefreshCw size={16}/> Retry connection</button><small>Existing evidence is retained while reconnecting.</small><button className="reset-history" onClick={()=>{setSessions([]);setSelected(null);setConnections({});setError('');setBusy(false);}}>Clear local session list</button></section></div>}
  </main>;
}
