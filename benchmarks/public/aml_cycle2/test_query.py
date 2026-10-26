# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import ipaddress
import json
import os
import random
import socket
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
import weakref

import httpx
from fastapi import HTTPException
import adapter

_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex


def loopback_only(method):
    def checked(sock, address):
        if not isinstance(address, tuple) or not ipaddress.ip_address(address[0]).is_loopback:
            raise AssertionError('query repair tests forbid external network I/O')
        return method(sock, address)
    return checked


socket.socket.connect = loopback_only(_connect)
socket.socket.connect_ex = loopback_only(_connect_ex)


def payload(query, top_k=100, user_id='alice'):
    return {'query':query, 'top_k':top_k, 'user_id':user_id}


class PartitionTests(unittest.TestCase):
    def test_ascii_and_multibyte_8191_8192_8193_boundaries(self):
        for size in (8191,8192,8193):
            for query in ('a'*size, '𐍈'*(size//4)+'a'*(size%4)):
                parts=adapter.query_parts(query)
                self.assertEqual(''.join(parts),query)
                self.assertTrue(all(0<len(part.encode())<=8192 for part in parts))
                if size<=8192:self.assertEqual(parts,[query])

    def test_whitespace_is_preserved_and_work_bound_covers_worst_case(self):
        for query in ['a'*4095+' '+'b'*9000, ('é '*2000+'漢\n'*1500)*3,
                      ('a'*4095+' ')*256, '𐍈'*(adapter.MAX_QUERY_BYTES//4)]:
            parts=adapter.query_parts(query)
            self.assertEqual(''.join(parts).encode(),query.encode())
            self.assertLessEqual(len(parts),256)
            self.assertTrue(all(len(part.encode())>=4096 for part in parts[:-1]))

    def test_decoded_limit_minus_one_equal_plus_one(self):
        for size in (adapter.MAX_QUERY_BYTES-1,adapter.MAX_QUERY_BYTES):
            adapter.search_input(payload('x'*size))
            self.assertLessEqual(len(adapter.query_parts('x'*size)),256)
        with self.assertRaises(ValueError):adapter.search_input(payload('x'*(adapter.MAX_QUERY_BYTES+1)))


class FusionTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_partitions_and_duplicate_ids_contribute_once(self):
        calls=[]
        async def recall(part):
            calls.append(part)
            return {'results':[{'id':'a','text':'first'},{'id':'a','text':'duplicate'}, {'id':'b','text':'second'}]}
        result=await adapter.fused_recall('x'*16384,3,recall)
        self.assertEqual(calls,['x'*8192])
        self.assertEqual([row['id'] for row in result],['a','b'])
        self.assertEqual(result[0]['text'],'first')
        self.assertEqual(result[0]['score'],1/61)
        self.assertEqual(result[1]['score'],1/62)

    async def test_stable_ties_and_top_k(self):
        lists=[[{'id':'b','text':'B'},{'id':'a','text':'A'}], [{'id':'a','text':'A'},{'id':'b','text':'B'}]]
        async def recall(part):return {'results':lists.pop(0)}
        result=await adapter.fused_recall('a'*8192+'b',2,recall)
        self.assertEqual([row['id'] for row in result],['a','b'])
        self.assertEqual(result[0]['score'],result[1]['score'])

    async def test_streaming_cache_releases_full_records_outside_winners(self):
        class Text(str):pass
        discarded=[];calls=0
        async def recall(part):
            nonlocal calls
            gc.collect()
            self.assertTrue(all(ref() is None for ref in discarded))
            calls+=1
            leader={'id':'winner','text':Text('retained')}
            loser={'id':'b' if calls==1 else f'z{calls}','text':Text('discarded')}
            if calls==2:discarded.append(weakref.ref(loser['text']))
            return {'results':[leader,loser]}
        result=await adapter.fused_recall('a'*8192+'b'*8192+'c',2,recall)
        self.assertEqual(result[0]['id'],'winner')

    async def test_internal_utf8_id_and_text_limits_minus_one_equal_plus_one(self):
        for key,limit in [('id',256),('text',65536)]:
            for size in (limit-1,limit,limit+1):
                value='é'*(size//2)+'x'*(size%2)
                hit={'id':'a','text':'memory'};hit[key]=value
                async def recall(part):return {'results':[hit]}
                if size<=limit:await adapter.fused_recall('x'*8193,1,recall)
                else:
                    with self.assertRaises(HTTPException):await adapter.fused_recall('x'*8193,1,recall)

    async def test_reentry_after_eviction_gets_content_from_current_batch(self):
        lists=[[{'id':'b','text':'B-first'}],[{'id':'a','text':'A'}],[{'id':'b','text':'B-current'}]]
        async def recall(part):return {'results':lists.pop(0)}
        result=await adapter.fused_recall('a'*8192+'b'*8192+'c',1,recall)
        self.assertEqual(result,[{'id':'b','text':'B-current','score':2/61}])

    async def test_streaming_fusion_matches_full_reference_with_changing_winners(self):
        rng=random.Random(0);scores={};records={};lists=[]
        for step in range(20):
            hits=[{'id':f'id-{rng.randrange(60):02}','text':f'content-{step}-{rank}'} for rank in range(20)]
            lists.append(hits);seen=set()
            for hit in hits:
                key=hit['id']
                if key in seen:continue
                seen.add(key);scores[key]=scores.get(key,0)+1/(60+len(seen));records[key]=hit
        expected=[dict(records[key],score=scores[key]) for key in sorted(scores,key=lambda key:(-scores[key],key))[:20]]
        async def recall(part):return {'results':lists.pop(0)}
        query=''.join(chr(65+i)*8192 for i in range(20))
        self.assertEqual(await adapter.fused_recall(query,20,recall),expected)

    async def test_invalid_native_shapes_and_bounds_fail_closed(self):
        invalid=[None,{}, {'results':{}},{'results':[{}]}, {'results':[{'id':'x'*257,'text':'x'}]},
                 {'results':[{'id':'a','text':'x'*65537}]}, {'results':[{'id':'a','text':'\ud800'}]},
                 {'results':[{'id':'a','text':'x','eventTime':253402300800000}]},
                 {'results':[{'id':'a','text':'x'},{'id':'b','text':'y'}]}]
        for result in invalid:
            async def recall(part):return result
            with self.assertRaises(HTTPException) as error:await adapter.fused_recall('a'*8193,1,recall)
            self.assertEqual(error.exception.status_code,503)


class RouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.native_requests=[]
        self.respond=lambda request:httpx.Response(200,json={'results':[{'id':'a','text':'memory','score':.75,'eventTime':0}]})
        async def native(request):
            self.native_requests.append(request)
            response=self.respond(request)
            return await response if hasattr(response,'__await__') else response
        real=httpx.AsyncClient
        with patch.object(adapter.httpx,'AsyncClient',side_effect=lambda **kw:real(transport=httpx.MockTransport(native),**kw)):
            self.app=adapter.create_app('http://native','synthetic-key')
        self.context=self.app.router.lifespan_context(self.app)
        await self.context.__aenter__()
        self.client=real(transport=httpx.ASGITransport(app=self.app),base_url='http://adapter',headers={'Authorization':'Bearer synthetic-key'})

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.context.__aexit__(None,None,None)

    async def test_exact_short_query_call_and_response_for_utf8_boundaries(self):
        for size in (8191,8192):
            for query in ('x'*size, '漢'*(size//3)+'x'*(size%3)):
                result=await self.client.post('/search',json=payload(query,1))
                self.assertEqual(result.json(),{'data':[{'id':'a','content':'memory','score':.75,'created_at':'1970-01-01T00:00:00Z'}]})
                request=self.native_requests[-1]
                self.assertEqual(json.loads(request.content),{'query':query,'k':1})
                self.assertEqual(request.headers['X-Tenant-ID'],adapter.tenant_for('alice'))
        self.assertEqual(len(self.native_requests),4)

    async def test_long_queries_preserve_text_and_user_isolation(self):
        query='𐍈'*2048+'tail'
        for user in ('alice','bob'):
            start=len(self.native_requests)
            response=await self.client.post('/search',json=payload(query,2,user))
            self.assertEqual(response.status_code,200)
            requests=self.native_requests[start:]
            self.assertEqual(''.join(json.loads(req.content)['query'] for req in requests),query)
            self.assertTrue(all(req.headers['X-Tenant-ID']==adapter.tenant_for(user) for req in requests))
            self.assertTrue(all(json.loads(req.content)['k']==2 for req in requests))

    async def test_chunk_error_returns_no_partial_results_and_stops_fanout(self):
        calls=0
        def response(request):
            nonlocal calls
            calls+=1
            return httpx.Response(200,json={'results':[{'id':'a','text':'memory'}]}) if calls==1 else httpx.Response(500)
        self.respond=response
        result=await self.client.post('/search',json=payload('a'*8192+'b'*8192+'c'))
        self.assertEqual(result.status_code,503)
        self.assertNotIn('data',result.json())
        self.assertEqual(calls,2)

    async def test_native_transport_invalid_json_and_shape_fail_closed(self):
        for failure in ('transport','json','shape'):
            def response(request):
                if failure=='transport':raise httpx.ConnectError('synthetic unavailable',request=request)
                return httpx.Response(200,content=b'{' if failure=='json' else b'[]')
            self.respond=response
            response=await self.client.post('/search',json=payload('x'*8193))
            self.assertEqual(response.status_code,503)
            self.assertNotIn('data',response.json())

    async def test_body_limit_minus_one_equal_plus_one_before_parse(self):
        body=json.dumps(payload('x',1)).encode()
        for size,expected in [(adapter.MAX_SEARCH_BODY-1,200),(adapter.MAX_SEARCH_BODY,200),(adapter.MAX_SEARCH_BODY+1,413)]:
            response=await self.client.post('/search',content=body+b' '*(size-len(body)))
            self.assertEqual(response.status_code,expected)
        self.assertEqual(len(self.native_requests),2)

    async def test_queue_is_bounded_and_cancellation_releases_capacity(self):
        entered=asyncio.Event();release=asyncio.Event();count=0
        async def response(request):
            nonlocal count
            count+=1
            if count==16:entered.set()
            await release.wait()
            return httpx.Response(200,json={'results':[]})
        self.respond=response
        tasks=[asyncio.create_task(self.client.post('/search',json=payload('x'))) for _ in range(32)]
        await asyncio.wait_for(entered.wait(),1)
        body_read=False
        async def unread_body():
            nonlocal body_read
            body_read=True
            yield json.dumps(payload('x')).encode()
        rejected=await self.client.post('/search',content=unread_body())
        self.assertEqual(rejected.status_code,503)
        self.assertFalse(body_read)
        self.assertEqual(count,16)
        tasks[0].cancel()
        with self.assertRaises(asyncio.CancelledError):await tasks[0]
        release.set()
        self.assertTrue(all(row.status_code==200 for row in await asyncio.gather(*tasks[1:])))
        self.assertEqual((await self.client.post('/search',json=payload('x'))).status_code,200)

    async def test_whole_deadline_and_invalid_json_release_capacity(self):
        blocked=asyncio.Event()
        async def response(request):
            await blocked.wait()
            return httpx.Response(200,json={'results':[]})
        self.respond=response
        with patch.object(adapter,'SEARCH_TIMEOUT',.01):
            result=await self.client.post('/search',json=payload('x'))
        self.assertEqual(result.status_code,503)
        blocked.set()
        self.assertEqual((await self.client.post('/search',json=payload('x'))).status_code,200)
        for body in (b'{',b'\xff',b'['*1100+b']'*1100):
            self.assertEqual((await self.client.post('/search',content=body)).status_code,422)

    async def test_deadline_includes_body_read_and_capacity_wait(self):
        block=asyncio.Event()
        async def body():
            await block.wait()
            yield json.dumps(payload('x')).encode()
        with patch.object(adapter,'SEARCH_TIMEOUT',.01):
            response=await self.client.post('/search',content=body())
        self.assertEqual(response.status_code,503)
        self.assertEqual(self.native_requests,[])
        entered=asyncio.Event();count=0
        async def native(request):
            nonlocal count
            count+=1
            if count==16:entered.set()
            await block.wait()
            return httpx.Response(200,json={'results':[]})
        self.respond=native
        active=[asyncio.create_task(self.client.post('/search',json=payload('x'))) for _ in range(16)]
        await asyncio.wait_for(entered.wait(),1)
        with patch.object(adapter,'SEARCH_TIMEOUT',.01):
            queued=await self.client.post('/search',json=payload('x'))
        self.assertEqual(queued.status_code,503)
        self.assertEqual(count,16)
        block.set()
        self.assertTrue(all(row.status_code==200 for row in await asyncio.gather(*active)))
        self.assertEqual((await self.client.post('/search',json=payload('x'))).status_code,200)


class NativeSyntheticTests(unittest.TestCase):
    def test_exact_s_long_query_and_user_isolation(self):
        binary=os.environ.get('HMS_QUERY_TEST_BINARY')
        if not binary:self.skipTest('HMS_QUERY_TEST_BINARY enables bounded synthetic exact-S integration')
        with tempfile.TemporaryDirectory() as data:
            process=subprocess.Popen([binary,'--no-auth','--bind','127.0.0.1:0','--data-dir',data,
                '--max-top-k','100','--max-tenants','4','--input-dim','384','--log-level','error'],
                stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
            timer=threading.Timer(120,process.terminate);timer.daemon=True;timer.start()
            try:
                line=process.stdout.readline();url='http://'+line.split('listening on ',1)[1].strip()
                async def exercise():
                    async with httpx.AsyncClient(base_url=url) as native:
                        rejected=await native.post('/api/v1/memory/recall',json={'query':'x'*8193,'k':100},headers={'X-Tenant-ID':adapter.tenant_for('alice')})
                        self.assertEqual(rejected.status_code,422)
                    app=adapter.create_app(url,'synthetic-key')
                    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',headers={'Authorization':'Bearer synthetic-key'}) as client:
                        for user,text in [('alice','The sapphire key is in the attic.'),('bob','The cobalt key is in the cellar.')]:
                            add={'request_id':'synthetic','user_id':user,'session_id':'s','messages':[{'role':'user','content':text}]}
                            unauthorized=await client.post('/add',json=add,headers={'Authorization':'Bearer wrong'})
                            self.assertEqual(unauthorized.status_code,401)
                            for _ in range(2):
                                response=await client.post('/add',json=add)
                                self.assertEqual(response.status_code,200)
                        response=await client.post('/search',json=payload('padding '*1200+'sapphire key'))
                        self.assertEqual(response.status_code,200)
                        self.assertTrue(any('sapphire' in item['content'] for item in response.json()['data']))
                        self.assertTrue(all('cobalt' not in item['content'] for item in response.json()['data']))
                        response=await client.post('/search',json=payload('padding '*1200+'sapphire key',user_id='unseen'))
                        self.assertEqual(response.json(),{'data':[]})
                asyncio.run(asyncio.wait_for(exercise(),120))
            finally:
                timer.cancel()
                process.terminate();process.wait(timeout=10)
                process.stdout.close()


if __name__=='__main__':unittest.main()
