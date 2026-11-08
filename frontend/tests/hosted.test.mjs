import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildSync } from 'esbuild';
import { fileURLToPath } from 'node:url';

const target=process.env.WHACK_SMOKE_URL;
const entry=fileURLToPath(new URL('../src/stream.ts',import.meta.url));
const {outputFiles}=buildSync({entryPoints:[entry],bundle:true,write:false,format:'esm'});
const {RunStream}=await import(`data:text/javascript;base64,${Buffer.from(outputFiles[0].text).toString('base64')}`);
async function waitFor(predicate,diagnostic,timeout=30000){
  const deadline=Date.now()+timeout;
  while(Date.now()<deadline){if(await predicate())return;await new Promise(r=>setTimeout(r,50));}
  assert.fail(diagnostic());
}

test('production HTTP, WS, controls, reconnect and fake-agent report', {skip:!target,timeout:120000}, async t=>{
  const url=new URL(target);
  const base=url.origin;
  globalThis.location={protocol:url.protocol,host:url.host};
  const streams=[];
  t.after(()=>streams.forEach(stream=>stream.close()));
  assert.equal((await fetch(`${base}/api/health`)).status,200);
  const html=await (await fetch(base)).text();
  assert.ok(html.includes('<html'));
  for(const asset of html.matchAll(/(?:src|href)="(\/assets\/[^" ]+)"/g)){
    assert.equal((await fetch(base+asset[1])).status,200);
  }
  for(const path of ['/api/missing','/assets/missing.js','/assets/missing']){
    assert.equal((await fetch(base+path)).status,404);
  }
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
  await waitFor(()=>['completed','failed'].includes(live.state?.status),()=>JSON.stringify(live.errors),60000);
  assert.equal(live.state.status,'completed',JSON.stringify(live.state));
  assert.ok(Object.values(live.state.nodes).some(n=>n.type==='verdict'));
  const report=await fetch(`${base}/api/runs/${live.state.id}/report`);
  assert.equal(report.status,200);
  assert.ok((await report.json()).verdicts.length>0);
  assert.deepEqual(demo.errors,[]);
  assert.deepEqual(live.errors,[]);
  // Finish the demo so the smoke check does not retain a running background task.
  await command(demo,'resume');
  await waitFor(()=>demo.state.status==='completed',()=>JSON.stringify(demo.errors),60000);
});
