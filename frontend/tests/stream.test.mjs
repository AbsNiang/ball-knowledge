import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildSync } from 'esbuild';
import { fileURLToPath } from 'node:url';

const {outputFiles} = buildSync({entryPoints:[fileURLToPath(new URL('../src/stream.ts',import.meta.url))],bundle:true,write:false,format:'esm'});
const source=outputFiles[0].text+'\n//# sourceURL=stream.bundle.mjs';
const {RunStream} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

class Socket {
  static CONNECTING = 0;
  static OPEN = 1;
  static instances = [];
  readyState = 0;
  sent = [];
  constructor(url) { this.url=url; Socket.instances.push(this); }
  open() { this.readyState=1; this.onopen?.(); }
  message(value) { this.onmessage?.({data:JSON.stringify(value)}); }
  close() { this.readyState=3; this.onclose?.(); }
  send(data) { this.sent.push(JSON.parse(data)); }
}

function setup(t) {
  Socket.instances=[];
  let nextTimer=0;
  const timers=new Map();
  t.mock.method(globalThis, 'setTimeout', (fn,ms) => { const id=++nextTimer;timers.set(id,{fn,ms});return id; });
  t.mock.method(globalThis, 'clearTimeout', (id) => timers.delete(id));
  const oldSocket=globalThis.WebSocket, oldLocation=globalThis.location;
  globalThis.WebSocket=Socket;
  globalThis.location={protocol:'http:',host:'127.0.0.1:5173'};
  const states=[], sessions=[], errors=[], commands=[];
  let missing=0;
  const stream=new RunStream({id:'run',question:'Why?',demo:true,status:'queued',seq:0,nodes:{},edges:{}},{
    connection:s=>states.push(s), session:s=>sessions.push(s), error:e=>errors.push(e),
    missing:()=>missing++, command:m=>commands.push(m),
  });
  t.after(()=>{stream.close();globalThis.WebSocket=oldSocket;globalThis.location=oldLocation;});
  const tick=()=>{const [id,{fn}]=[...timers].sort((a,b)=>a[1].ms-b[1].ms)[0];timers.delete(id);fn();};
  return {stream,timers,tick,states,sessions,errors,commands,get missing(){return missing;}};
}
const event=(seq,status)=>({run_id:'run',seq,type:'run_status',payload:{type:'run_status',status}});

test('missing saved run is removed without retrying or marking the connection failed',t=>{
  const env=setup(t), socket=Socket.instances[0];
  socket.open();
  socket.message({detail:{code:'run_not_found',message:'Unknown run'}});
  assert.equal(env.missing,1);
  assert.equal(socket.readyState,3);
  assert.equal(env.timers.size,0);
  assert.ok(!env.states.includes('failed'));
});

test('temporary disconnect resumes at the next event and never repeats a command',t=>{
  const env=setup(t), socket=Socket.instances[0];
  socket.open(); socket.message(event(1,'running'));
  assert.equal(env.stream.send({request_id:'pause',command:{type:'pause'}}),true);
  socket.close();
  assert.equal(env.states.at(-1),'reconnecting');
  env.tick();
  const next=Socket.instances[1];
  assert.ok(next.url.endsWith('from_seq=2'));
  next.open(); next.message(event(2,'paused'));
  assert.equal(env.sessions.at(-1).status,'paused');
  assert.deepEqual(next.sent,[]);
});

test('strict-mode cleanup waits for the handshake and ignores late events',t=>{
  const env=setup(t), socket=Socket.instances[0];
  env.stream.close();
  assert.equal(socket.readyState,0);
  socket.open(); socket.message(event(1,'running'));
  assert.equal(socket.readyState,3);
  assert.deepEqual(env.sessions,[]);
  assert.equal(env.timers.size,0);
});

test('unavailable server has bounded retries and can be retried manually',t=>{
  const env=setup(t);
  for(let i=0;i<5;i++){Socket.instances.at(-1).close();env.tick();}
  Socket.instances.at(-1).close();
  assert.equal(env.states.at(-1),'failed');
  env.stream.retry();
  Socket.instances.at(-1).open();
  assert.equal(env.states.at(-1),'connected');
});

test('invalid events fail visibly instead of silently corrupting the graph',t=>{
  const env=setup(t), socket=Socket.instances[0];
  socket.open(); socket.message(event(2,'running'));
  assert.match(env.errors[0],/gap/);
  assert.equal(env.states.at(-1),'failed');
  assert.deepEqual(env.sessions,[]);
});

test('heartbeat detects a proxy connection left open after backend shutdown',t=>{
  const env=setup(t), socket=Socket.instances[0];
  socket.open();env.tick();
  assert.deepEqual(socket.sent,[{type:'ping'}]);
  env.tick();
  assert.equal(env.states.at(-1),'reconnecting');
  env.tick();
  assert.equal(Socket.instances.length,2);
});

test('pong keeps an idle connection alive without changing graph state',t=>{
  const env=setup(t), socket=Socket.instances[0];
  socket.open();env.tick();socket.message({type:'pong'});
  assert.equal(env.states.at(-1),'connected');
  assert.deepEqual(env.sessions,[]);
  assert.deepEqual(env.errors,[]);
  assert.equal(env.timers.size,1);
});

test('a stalled handshake times out and retries',t=>{
  const env=setup(t);
  env.tick();
  assert.equal(env.states.at(-1),'reconnecting');
  env.tick();
  assert.equal(Socket.instances.length,2);
});
