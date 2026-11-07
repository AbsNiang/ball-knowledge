import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { transformSync } from 'esbuild';
const {code}=transformSync(readFileSync(new URL('../src/state.ts',import.meta.url),'utf8'),{loader:'ts',format:'esm'});
const {newSession,reduceEvent}=await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const event=(seq,payload)=>({run_id:'run',seq,payload});
test('replay is deterministic, deduplicated, and preserves failed nodes',()=>{
 const initial=newSession('run','Why?',true);
 const events=[event(1,{type:'node_created',node:{id:'root',lifecycle:'running'}}),event(2,{type:'node_updated',node:{id:'root',lifecycle:'complete',check_outcome:'failed'}}),event(3,{type:'node_created',node:{id:'branch',lifecycle:'running'}}),event(4,{type:'edge_created',edge:{id:'edge',source:'branch',target:'root'}})];
 const result=events.reduce(reduceEvent,initial);
 assert.equal(result.nodes.root.check_outcome,'failed');
 assert.equal(Object.keys(result.nodes).length,2);
 assert.deepEqual(events.reduce(reduceEvent,result),result);
 assert.deepEqual(events.reduce(reduceEvent,initial),result);
 assert.equal(Object.keys(initial.nodes).length,0);
});
test('reject sequence gaps and ignore events for another session',()=>{
 const initial=newSession('run','Why?',true);
 assert.throws(()=>reduceEvent(initial,event(2,{type:'run_status',status:'paused'})),/gap/);
 assert.equal(reduceEvent(initial,{...event(1,{}),run_id:'other'}),initial);
});
test('gate records update by id without deriving a verdict',()=>{
 let state=reduceEvent(newSession('run','Why?',true),event(1,{type:'node_created',node:{id:'root',checks:[],check_outcome:'pending'}}));
 state=reduceEvent(state,event(2,{type:'check_result',check:{id:'g1',node_id:'root',status:'pass'}}));
 state=reduceEvent(state,event(3,{type:'check_result',check:{id:'g1',node_id:'root',status:'fail'}}));
 assert.equal(state.nodes.root.checks.length,1);
 assert.equal(state.nodes.root.checks[0].status,'fail');
 assert.equal(state.nodes.root.check_outcome,'pending');
});
