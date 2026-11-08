import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { fileURLToPath } from 'node:url';
import { buildSync } from 'esbuild';

const root=fileURLToPath(new URL('../../..',import.meta.url));
const frontend=fileURLToPath(new URL('../..',import.meta.url));
const {outputFiles}=buildSync({entryPoints:[`${frontend}/src/stream.ts`],bundle:true,write:false,format:'esm'});
const source=outputFiles[0].text+'\n//# sourceURL=stream.integration.mjs';
const {RunStream}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));

async function freePort(){
  const server=createServer();
  await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});
  const port=server.address().port;
  await new Promise(resolve=>server.close(resolve));
  return port;
}
async function waitFor(predicate,diagnostic,timeout=30000){
  const deadline=Date.now()+timeout;
  while(Date.now()<deadline){if(await predicate())return;await delay(30);}
  assert.fail(diagnostic());
}
function launch(command,args,cwd,env){
  const child=spawn(command,args,{cwd,env:{...process.env,...env},windowsHide:true,stdio:['ignore','pipe','pipe']});
  child.log='';
  child.stdout.on('data',data=>child.log+=data);
  child.stderr.on('data',data=>child.log+=data);
  child.on('error',error=>child.log+=error.message);
  return child;
}
async function stop(child){
  if(child.exitCode!==null)return;
  const exited=new Promise(resolve=>child.once('close',resolve));
  child.kill();await exited;
}

test('Vite proxies HTTP, live agent events, controls, reconnects and backend restarts', {timeout:120000}, async t=>{
  const apiPort=await freePort(), uiPort=await freePort();
  const python=process.env.WHACK_TEST_PYTHON || `${root}/.venv/${process.platform==='win32'?'Scripts/python.exe':'bin/python'}`;
  const startApi=()=>launch(python,['-m','uvicorn','backend.tests.fake_server:app','--port',String(apiPort)],root,{LLM_MODE:'fake',LANGSMITH_TRACING:'false'});
  let api=startApi();
  const vite=launch(process.execPath,[`${frontend}/node_modules/vite/bin/vite.js`,'--host','127.0.0.1','--port',String(uiPort)],frontend,{WHACK_API_TARGET:`http://127.0.0.1:${apiPort}`});
  const streams=[];
  t.after(async()=>{streams.forEach(s=>s.close());await stop(vite);await stop(api);});
  const base=`http://127.0.0.1:${uiPort}`;
  globalThis.location={protocol:'http:',host:`127.0.0.1:${uiPort}`};
  await waitFor(async()=>{try{return (await fetch(`${base}/api/health`)).ok;}catch{return false;}},()=>api.log+'\n'+vite.log);
  assert.equal((await fetch(base)).status,200);
  async function create(path){
    const response=await fetch(`${base}/api/${path}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:'Why do back-to-back teams underperform?'})});
    const body=await response.json();
    assert.equal(response.status,202,JSON.stringify(body));
    return body.run_id;
  }
  function connect(id,demo){
    const observed={state:null,connection:null,missing:false,commands:[],errors:[]};
    observed.stream=new RunStream({id,question:'Why?',demo,status:'queued',seq:0,nodes:{},edges:{}},{
      session:s=>observed.state=s,connection:s=>observed.connection=s,
      missing:()=>observed.missing=true,error:e=>observed.errors.push(e),command:m=>observed.commands.push(m),
    });
    streams.push(observed.stream);return observed;
  }
  async function command(view,type,payload={},node_id=null){
    const request_id=crypto.randomUUID();
    assert.equal(view.stream.send({request_id,command:{type,payload,node_id}}),true);
    await waitFor(()=>view.commands.some(m=>m.request_id===request_id),()=>JSON.stringify(view.errors));
    assert.equal(view.commands.find(m=>m.request_id===request_id).type,'command_ack');
  }
  const demo=connect(await create('demo/runs'),true);
  await waitFor(()=>Object.keys(demo.state?.nodes||{}).length>0,()=>JSON.stringify(demo.errors));
  await command(demo,'pause');
  await waitFor(()=>demo.state.status==='paused',()=>JSON.stringify(demo.state));
  const seq=demo.state.seq;
  demo.stream.retry();
  await waitFor(()=>demo.connection==='connected',()=>JSON.stringify(demo.errors));
  await command(demo,'resume');
  await waitFor(()=>demo.state.seq>seq,()=>JSON.stringify(demo.errors));
  const active=Object.values(demo.state.nodes).find(n=>n.lifecycle==='running');
  await command(demo,'redirect',{statement:'Explore travel instead'},active.id);
  await waitFor(()=>Object.values(demo.state.edges).some(e=>e.label==='Explore travel instead'),()=>JSON.stringify(demo.errors));
  assert.ok(demo.state.nodes[active.id].flags.includes('pruned'));
  await command(demo,'pause');
  const live=connect(await create('runs'),false);
  await waitFor(()=>['completed','failed'].includes(live.state?.status),()=>api.log,60000);
  assert.equal(live.state.status,'completed',JSON.stringify(live.state));
  assert.ok(Object.values(live.state.nodes).some(n=>n.type==='verdict'));
  const report=await fetch(`${base}/api/runs/${live.state.id}/report`);
  assert.equal(report.status,200);
  assert.ok((await report.json()).verdicts.length>0);
  await stop(api);api=startApi();
  await waitFor(()=>demo.missing&&live.missing,()=>api.log+'\n'+vite.log+'\n'+JSON.stringify([demo.connection,live.connection,demo.errors,live.errors]),45000);
  assert.deepEqual(demo.errors,[]);
  assert.deepEqual(live.errors,[]);
});
