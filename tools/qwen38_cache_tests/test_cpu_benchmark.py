"""CPU protocol checks are not GPU correctness/performance evidence."""
import copy
import json
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .benchmark import VERSION
from .benchmark.core import (StreamMeasurement, check_effective, digest, file_hash,
                             read, resume_guard, schedule, summarize, validate, write)
from .benchmark.__main__ import freeze_answer, prepare, seal_trace, summarize_run
from .benchmark.replay import numeric_delta, validate_trace
from .benchmark.runner import generate, run_unit, stop_owned

HERE = Path(__file__).parent / "benchmark"


def fixture():
    suite = read(HERE / "suite.example.json")
    machine = read(HERE / "machine.example.json")
    machine["container_image"] = "local@sha256:test"
    return suite, machine


def stream_result(ids=(1, 2, 3), chunks=(1, 3), finish="length", output=3):
    m = StreamMeasurement()
    for index, n in enumerate(chunks):
        m.feed({"output_ids": list(ids[:n]), "meta_info": {
            "prompt_tokens": 4, "completion_tokens": n,
            "finish_reason": {"type": finish} if index == len(chunks)-1 else None}}, 0.2 + index * 0.4)
    return m.finish(0.8, {"length_policy": "fixed", "sampling": {"max_new_tokens": output}}, 4)


class BenchmarkProtocol(unittest.TestCase):
    def test_chunk_tokens_and_first_last_timing(self):
        result = stream_result()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["ttft_s"], 0.2)
        self.assertAlmostEqual(result["tpot_s"], 0.2)
        self.assertEqual(result["coalesced_chunks"], 1)
        self.assertEqual(result["output_tokens"], 3)

    def test_one_token_and_empty_are_not_zero(self):
        result = stream_result(ids=(4,), chunks=(1,), output=1)
        self.assertIsNone(result["tpot_s"])
        self.assertEqual(result["status"], "ok")
        result = stream_result(ids=(), chunks=(0,), output=1)
        self.assertIsNone(result["ttft_s"])
        self.assertEqual(result["status"], "failed")

    def test_incomplete_truncated_and_changed_ids(self):
        self.assertEqual(stream_result(finish=None)["status"], "failed")
        self.assertEqual(stream_result(output=4)["status"], "failed")
        m = StreamMeasurement()
        m.feed({"output_ids": [1]}, 0.1)
        with self.assertRaisesRegex(ValueError, "non-cumulative"):
            m.feed({"output_ids": [2]}, 0.2)

    def test_eos_policy_and_count_mismatch(self):
        m = StreamMeasurement()
        m.feed({"output_ids": [1], "meta_info": {"prompt_tokens": 4,
            "completion_tokens": 1, "finish_reason": {"type": "stop"}}}, 0.1)
        self.assertEqual(m.finish(0.2, {"length_policy": "eos", "sampling": {"max_new_tokens": 8}}, 4)["status"], "ok")
        self.assertIn("prompt_count_mismatch", m.finish(0.2, {"length_policy": "eos", "sampling": {"max_new_tokens": 8}}, 5)["errors"])

    def test_matrix_seed_pairs_and_isolation_rejects(self):
        suite, machine = fixture()
        suite["candidates"].append({**copy.deepcopy(suite["candidates"][0]), "id": "control"})
        rows = validate(suite, machine)
        self.assertEqual(len(rows), 70)
        self.assertEqual(rows, schedule(suite))
        for r in range(5):
            for case in suite["cases"]:
                group = [u for u in rows if u["pair"] == f"{r}:{case['id']}"]
                self.assertEqual({u["candidate"] for u in group}, {"baseline04", "control"})
        suite["candidates"][1]["loader"]["adaptive_cache"] = True
        with self.assertRaisesRegex(ValueError, "adaptive"):
            validate(suite, machine)

    def test_layout_effective_and_unsafe_names(self):
        suite, machine = fixture()
        candidate = suite["candidates"][0]
        info = {**candidate["expected_effective"], "model_loader_extra_config": json.dumps(candidate["loader"])}
        check_effective(info, candidate, candidate["loader"])
        info["chunked_prefill_size"] = 2048
        with self.assertRaisesRegex(ValueError, "effective"):
            check_effective(info, candidate, candidate["loader"])
        candidate["loader"]["layer_cache_slots"] = [96]
        with self.assertRaisesRegex(ValueError, "slot"):
            validate(suite, machine)
        suite, machine = fixture()
        suite["cases"][0]["id"] = "../../oops"
        with self.assertRaisesRegex(ValueError, "unsafe"):
            validate(suite, machine)

    def test_input_length_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, machine = fixture()
            suite["cases"] = [suite["cases"][0]]
            write(root / "short.json", [1])
            with self.assertRaisesRegex(ValueError, "length mismatch"):
                validate(suite, machine, root)
            path = root / "run.json"
            resume_guard(path, {"code": "abc", "input": "a"})
            resume_guard(path, {"input": "a", "code": "abc"})
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                resume_guard(path, {"code": "abc", "input": "b"})

    def test_paired_summary_retains_failures_and_nulls(self):
        suite, _ = fixture()
        suite["cases"] = [suite["cases"][0]]
        suite["candidates"].append({**copy.deepcopy(suite["candidates"][0]), "id": "control"})
        samples = []
        for r in range(5):
            for candidate, tpot in (("baseline04", 0.2), ("control", 0.18)):
                req = stream_result()
                req["tpot_s"] = tpot
                samples.append(dict(round=r, case="short-decode", candidate=candidate, status="ok", requests=[req]))
        report = summarize(samples, suite)
        pair = report["paired"][0]
        self.assertEqual(len(pair["tpot_ratios"]), 5)
        self.assertAlmostEqual(pair["mean_ratio_bootstrap_95"][0], 0.9)
        samples[-1]["status"] = "failed"
        report = summarize(samples, suite)
        self.assertEqual(len(report["failed_samples"]), 1)
        self.assertIsNone(report["paired"][0]["mean_ratio_bootstrap_95"])
        self.assertEqual(report["rows"][1]["failed_units"], 1)
        self.assertTrue(report["rows"][0]["metrics"]["tpot_s"]["tail_unstable"])

    def test_trace_hash_dimensions_order_and_deltas(self):
        suite, _ = fixture()
        suite.update(num_layers=2, num_experts=4, top_k=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            events = [{"layer": l, "token": t, "routes": [[0, 1]]} for t in range(2) for l in range(2)]
            path.write_text("".join(json.dumps(e)+"\n" for e in events))
            meta = dict(trace_sha256=file_hash(path), manifest_sha256="pack", kind="synthetic", events=4)
            self.assertEqual(len(validate_trace(path, meta, suite, "pack")), 4)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                validate_trace(path, meta, suite, "other")
            events[1]["token"] = 1
            path.write_text("".join(json.dumps(e)+"\n" for e in events))
            meta["trace_sha256"] = file_hash(path)
            with self.assertRaisesRegex(ValueError, "boundary"):
                validate_trace(path, meta, suite, "pack")
        self.assertEqual(numeric_delta({"hits": 4, "cache_hit_rate": 0.5}, {"hits": 8, "cache_hit_rate": 0.6}), {"hits": 4})

    def test_generate_http_stream_and_disconnect(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200); self.end_headers()
                for n in (1, 3):
                    value = {"output_ids": [1, 2, 3][:n], "meta_info": {
                        "prompt_tokens": 4, "completion_tokens": n,
                        "finish_reason": {"type": "length"} if n == 3 else None}}
                    self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode()); self.wfile.flush()
                    time.sleep(0.01)
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = generate(f"http://127.0.0.1:{server.server_port}", [5]*4,
                    {"length_policy": "fixed", "sampling": {"max_new_tokens": 3}}, 2, Path(directory)/"result.json")
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["output_tokens"], 3)
                self.assertEqual(result["input_ids"], [5]*4)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_owned_process_stop(self):
        import subprocess
        import sys
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        stop_owned(process)
        self.assertIsNotNone(process.poll())

    def test_port_collision_preserves_listener_and_failure(self):
        suite, machine = fixture()
        unit = schedule(suite)[0]
        with socket.socket() as listener, tempfile.TemporaryDirectory() as directory:
            listener.bind(("127.0.0.1", 0)); listener.listen()
            machine["port"] = listener.getsockname()[1]
            import_paths = json.dumps({"sglang": str(HERE.parents[2]/"python/sglang/__init__.py"),
                                       "runtime": str(HERE.parents[2]/"python/sglang/srt/layers/moe/qwen38_cache_threaded.py")})
            with patch("tools.qwen38_cache_tests.benchmark.runner.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=import_paths, stderr="")), \
                 patch("tools.qwen38_cache_tests.benchmark.runner.telemetry", return_value={}), \
                 patch("tools.qwen38_cache_tests.benchmark.runner.subprocess.Popen") as launch:
                result = run_unit(unit, suite, machine, directory, directory, "benchmark")
                self.assertEqual(result["status"], "failed")
                launch.assert_not_called()
                self.assertTrue((Path(directory)/"unit.json").exists())
            with socket.create_connection(("127.0.0.1", machine["port"]), timeout=1):
                pass

    def test_freeze_answer_is_immutable_and_keeps_provenance(self):
        suite, _ = fixture()
        suite["session_strip_tail_ids"] = [99]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root/"suite.json", suite); write(root/"arbiter-suffix.json", [7,8])
            write(root/"unit.json", dict(status="ok", case="raft-session", round=0, candidate="baseline04",
                  requests=[dict(status="ok",input_ids=[1,2],output_ids=[3,99],output_sha256=digest([3,99]),finish_reason={"type":"stop"})]))
            args=SimpleNamespace(suite=root/"suite.json",source_unit=root/"unit.json",inputs=root,output_suite=root/"frozen-suite.json")
            freeze_answer(args)
            self.assertEqual(read(root/"raft-answer-prefill.json"), [1,2,3,7,8])
            self.assertEqual(read(root/"frozen-suite.json")["cases"][-1]["turns"][0]["input_tokens"], 5)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                freeze_answer(args)

    def test_mixed_summary_fingerprints_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root/"run.json", {"fingerprint":"a"})
            write(root/"units/u/unit.json", {"fingerprint":"b"})
            with self.assertRaisesRegex(ValueError, "mixed"):
                summarize_run(SimpleNamespace(results=root))

    def test_prepare_local_tokenizer_and_exact_lengths(self):
        # CPU tokenizer fixture verifies template assembly, not model behavior.
        import sys
        class Tokenizer:
            eos_token_id = 99
            def encode(self, text, add_special_tokens=False):
                return [ord(c)+100 for c in text]
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
                text = ''.join(f"<{m['role']}>"+m['content']+"</end>" for m in messages)+"<assistant>"
                return self.encode(text) if tokenize else text
        fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, machine = fixture()
            write(root/'suite.json', suite)
            args = SimpleNamespace(suite=root/'suite.json',checkpoint=root,output=root/'inputs')
            with patch.dict(sys.modules, {'transformers': fake}):
                prepare(args)
            prepared = read(root/'inputs/suite.json')
            self.assertEqual(len(read(root/'inputs/28k.json')), 28672)
            self.assertEqual(len(read(root/'inputs/50k.json')), 51200)
            validate(prepared, machine, root/'inputs')
            with patch.dict(sys.modules, {'transformers': fake}), self.assertRaisesRegex(ValueError, 'empty directory'):
                prepare(args)

    def test_failure_before_launch_is_saved(self):
        suite, machine = fixture()
        with tempfile.TemporaryDirectory() as directory:
            with patch('tools.qwen38_cache_tests.benchmark.runner.subprocess.run', side_effect=TimeoutError('probe deadline')), \
                 patch('tools.qwen38_cache_tests.benchmark.runner.telemetry', return_value={}):
                result = run_unit(schedule(suite)[0], suite, machine, directory, directory, 'benchmark', 'identity')
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(read(Path(directory)/'unit.json')['fingerprint'], 'identity')
            self.assertIn('probe deadline', result['error'])

    def test_output_deviation_and_missing_pair(self):
        suite, _ = fixture()
        suite['cases'] = [suite['cases'][0]]
        suite['candidates'].append({**copy.deepcopy(suite['candidates'][0]), 'id':'control'})
        base = dict(round=0,case='short-decode',candidate='baseline04',status='ok',requests=[stream_result()])
        other = copy.deepcopy(base); other['candidate']='control'; other['requests'][0]['output_ids']=[1,2,4]
        report = summarize([base,other], suite)
        self.assertTrue(report['paired'][0]['correctness_review_required'])
        self.assertEqual(report['rows'][0]['missing_units'],4)

    def test_http_disconnect_marks_failed(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200); self.end_headers()
                self.wfile.write(b'data: {"output_ids":[1],"meta_info":{"completion_tokens":1,"prompt_tokens":4}}\n\n')
                self.wfile.flush()
            def log_message(self,*args): pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                result=generate(f'http://127.0.0.1:{server.server_port}',[1]*4,
                    {'length_policy':'fixed','sampling':{'max_new_tokens':3}},2,Path(directory)/'result.json')
                self.assertEqual(result['status'],'failed')
                self.assertEqual(result['output_ids'],[1])
                self.assertIn('incomplete_or_failed_finish', result['errors'])
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_trace_sealing_refuses_partial_sweep(self):
        suite,machine=fixture()
        suite.update(num_layers=2,num_experts=4,top_k=2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            machine['manifest']=str(root/'manifest.json');write(root/'manifest.json',{})
            write(root/'suite.json',suite);write(root/'machine.json',machine)
            write(root/'unit.json',{'mode':'trace','status':'ok','fingerprint':'source'})
            (root/'routes.raw.jsonl').write_text('{"layer":0,"routes":[[1,2]]}\n')
            with self.assertRaisesRegex(ValueError,'incomplete trace'):
                seal_trace(SimpleNamespace(suite=root/'suite.json',machine=root/'machine.json',
                    source_unit=root/'unit.json',output=root/'sealed.jsonl'))

    def test_http_total_deadline_keeps_partial_output(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200); self.end_headers()
                self.wfile.write(b'data: {"output_ids":[1],"meta_info":{"completion_tokens":1,"prompt_tokens":4}}\n\n')
                self.wfile.flush(); time.sleep(0.3)
            def log_message(self,*args): pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                result=generate(f'http://127.0.0.1:{server.server_port}',[1]*4,
                    {'length_policy':'fixed','sampling':{'max_new_tokens':3}},0.1,Path(directory)/'result.json')
                self.assertEqual(result['status'],'failed')
                self.assertEqual(result['output_ids'],[1])
                self.assertTrue(any('timed out' in e or 'deadline' in e for e in result['errors']))
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_sampling_api_field_is_supported_by_source(self):
        # API name audit: suite params must be dataclass fields in this actual checkout.
        import ast
        from .benchmark.core import ROOT
        tree=ast.parse((ROOT/'python/sglang/srt/sampling/sampling_params.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SamplingParams')
        fields={n.target.id for n in cls.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)}
        suite,_=fixture()
        for case in suite['cases']:
            for turn in case['turns']+case['warmup_requests']:
                self.assertFalse(set(turn['sampling'])-fields)

    def test_replay_benchmark_has_no_profile_events(self):
        from .benchmark.replay import execute_interval, ReplayProfile
        runtime = SimpleNamespace(synchronize=lambda: None, stats_snapshot=lambda: {'hits': 1})
        events = [{'layer': 0, 'token': 0, 'routes': [[1]]}, {'layer': 1, 'token': 0, 'routes': [[1]]}]
        from unittest.mock import Mock
        cuda = Mock()
        execute = Mock()
        execute_interval(runtime, [1, 2], events, execute)
        cuda.Event.assert_not_called()
        self.assertEqual(execute.call_count, 2)
        cuda.Event.return_value.elapsed_time.return_value = 1.0
        profiler = ReplayProfile(cuda)
        execute_interval(runtime, [1, 2], events, execute, profiler)
        self.assertEqual(cuda.Event.call_count, 4)
        self.assertEqual(cuda.Event.return_value.record.call_count, 4)
        gpu, host, tokens = profiler.results(events, 2)
        self.assertEqual(len(gpu), 2)
        self.assertEqual(len(tokens), 1)

    def test_effective_environment_fingerprints_execution_settings(self):
        from .benchmark.core import subprocess_environment, machine_identity
        with tempfile.TemporaryDirectory() as directory, patch('tools.qwen38_cache_tests.benchmark.core.command', return_value={}):
            path = Path(directory)/'run.json'
            with patch.dict('os.environ', {'CUDA_LAUNCH_BLOCKING': '0', 'SGLANG_QWEN38_ELASTIC_L2': 'off'}, clear=True):
                before = machine_identity()
                resume_guard(path, before)
                env = subprocess_environment()
                self.assertEqual(env['CUDA_LAUNCH_BLOCKING'], '0')
            for key, value in [('CUDA_LAUNCH_BLOCKING','1'), ('SGLANG_QWEN38_ELASTIC_L2','on'),
                               ('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:64'), ('CUDA_DEVICE_MAX_CONNECTIONS','1')]:
                with patch.dict('os.environ', {'CUDA_LAUNCH_BLOCKING':'0','SGLANG_QWEN38_ELASTIC_L2':'off',key:value}, clear=True):
                    with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):
                        resume_guard(path, machine_identity())
            with patch.dict('os.environ', {'CUDA_LAUNCH_BLOCKING':'0','SGLANG_QWEN38_ELASTIC_L2':'off',
                                          'UNRELATED_NOTE':'changed','SECRET_TOKEN':'do-not-forward',
                                          'SGLANG_UNKNOWN_CACHE_SWITCH':'on','PYTHONPATH':'elsewhere'}, clear=True):
                self.assertEqual(machine_identity(), before)
                effective = subprocess_environment()
                self.assertNotIn('SECRET_TOKEN',effective)
                self.assertNotIn('SGLANG_UNKNOWN_CACHE_SWITCH',effective)
                self.assertNotEqual(effective['PYTHONPATH'],'elsewhere')

    def replay_fixture(self, root):
        from .benchmark.core import subprocess_environment
        suite, machine = fixture()
        suite['rounds']=1; suite['cases']=[suite['cases'][0]]
        suite['cases'][0]['turns'][0]['input_tokens']=1
        write(root/'short.json',[1]);write(root/'suite.json',suite)
        machine['manifest']=str(root/'manifest.json');write(root/'manifest.json',{})
        write(root/'machine.json',machine)
        events=[{'layer': l,'token': 0,'routes':[list(range(10))]} for l in range(48)]
        (root/'trace.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        meta=dict(kind='synthetic',events=48,trace_sha256=file_hash(root/'trace.jsonl'),
                  manifest_sha256=file_hash(root/'manifest.json'),model_revision=machine['model_revision'])
        write(root/'trace.meta.json',meta)
        args=SimpleNamespace(suite=str(root/'suite.json'),machine=str(root/'machine.json'),inputs=str(root),
            trace=str(root/'trace.jsonl'),trace_meta=str(root/'trace.meta.json'),warmup_events=0,correctness=False,
            profile=False,results=str(root/'results'),resume=True)
        worker=dict(status='ok',mode='benchmark',candidate='baseline04',trace_sha256=meta['trace_sha256'],warmup_events=0,
                    measured_event_range=[0,48],version=VERSION,effective_child_env=subprocess_environment(),
                    before={},after={},counter_delta={},effective_runtime={'stats_log_interval':0},
                    correctness={'status':'not_run'},wall_s=1.0)
        return args, worker, meta

    def test_replay_resume_rejects_foreign_and_malformed_units(self):
        from .benchmark.__main__ import replay_matrix
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args,worker,meta=self.replay_fixture(root)
            resume_guard(root/'results/run.json',{'fixture':'identity','extra':{'mode':'replay-benchmark'}})
            unit=dict(fingerprint=digest({'fixture':'identity','extra':{'mode':'replay-benchmark'}}),round=0,candidate='baseline04',order=0,pair='replay:0',
                      mode='benchmark',trace_sha256=meta['trace_sha256'],warmup_events=0,status='ok',result=worker)
            path=root/'results/r0-baseline04/unit.json'
            for key,value in [('fingerprint','foreign'),('candidate','wrong'),('round',1),('mode','profile'),
                              ('trace_sha256','wrong'),('result',None)]:
                bad=copy.deepcopy(unit);bad[key]=value;write(path,bad)
                with patch('tools.qwen38_cache_tests.benchmark.__main__.identity',return_value={'fixture':'identity','extra':{'mode':'replay-benchmark'}}), \
                     self.assertRaises(ValueError):
                    replay_matrix(args)
            write(path,unit)
            with patch('tools.qwen38_cache_tests.benchmark.__main__.identity',return_value={'fixture':'identity','extra':{'mode':'replay-benchmark'}}):
                self.assertEqual(replay_matrix(args),0)
            self.assertEqual(read(root/'results/summary.json')['wall_s']['baseline04']['median'],1.0)

    def test_replay_worker_success_requires_valid_report_and_recovers_orphan(self):
        from .benchmark.__main__ import replay_matrix
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args,worker,meta=self.replay_fixture(root)
            target=root/'results/r0-baseline04'
            def launch(argv, **kwargs):
                self.assertNotIn('SECRET_TOKEN',kwargs['env'])
                write(Path(argv[argv.index('--output')+1]), worker)
                return SimpleNamespace(returncode=0,pid=999999,wait=lambda **kw:0)
            with patch('tools.qwen38_cache_tests.benchmark.__main__.identity',return_value={'fixture':'identity','extra':{'mode':'replay-benchmark'}}), \
                 patch('tools.qwen38_cache_tests.benchmark.__main__.subprocess.Popen',side_effect=launch), \
                 patch('tools.qwen38_cache_tests.benchmark.__main__.stop_owned'):
                # Orphan report from interrupted work is discarded and worker starts fresh.
                write(target/'replay.json',{'status':'ok','wall_s':0.001})
                self.assertEqual(replay_matrix(args),0)
                self.assertEqual(read(target/'unit.json')['result']['wall_s'],1.0)
            (target/'unit.json').unlink()
            with patch('tools.qwen38_cache_tests.benchmark.__main__.identity',return_value={'fixture':'identity','extra':{'mode':'replay-benchmark'}}), \
                 patch('tools.qwen38_cache_tests.benchmark.__main__.subprocess.Popen',return_value=SimpleNamespace(returncode=0,wait=lambda **kw:0)), \
                 patch('tools.qwen38_cache_tests.benchmark.__main__.stop_owned'):
                self.assertEqual(replay_matrix(args),1)
            result=read(target/'unit.json')
            self.assertEqual(result['status'],'failed')
            self.assertIn('invalid worker result',result['error'])
            self.assertEqual(read(root/'results/summary.json')['wall_s']['baseline04']['n'],0)

    def test_replay_profile_never_enters_speed_summary(self):
        from .benchmark.__main__ import replay_matrix
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args,worker,meta=self.replay_fixture(root);args.profile=True
            worker.update(mode='profile',wall_s=None,diagnostic_wall_s=1.0)
            def launch(argv,**kwargs):
                self.assertIn('--profile',argv)
                write(Path(argv[argv.index('--output')+1]),worker)
                return SimpleNamespace(returncode=0,wait=lambda **kw:0)
            with patch('tools.qwen38_cache_tests.benchmark.__main__.identity',return_value={'fixture':'profile','extra':{'mode':'replay-profile'}}), \
                 patch('tools.qwen38_cache_tests.benchmark.__main__.subprocess.Popen',side_effect=launch), \
                 patch('tools.qwen38_cache_tests.benchmark.__main__.stop_owned'):
                self.assertEqual(replay_matrix(args),0)
            summary=read(root/'results/summary.json')
            self.assertEqual(summary['mode'],'replay-profile')
            self.assertEqual(summary['paired'],[])
            self.assertEqual(summary['wall_s']['baseline04']['n'],0)

    def test_delayed_response_headers_follow_total_deadline(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']));time.sleep(0.05)
                self.send_response(200);self.end_headers()
                self.wfile.write(b'data: {"output_ids":[1],"meta_info":{"completion_tokens":1,"prompt_tokens":4,"finish_reason":{"type":"length"}}}\n\n')
            def log_message(self,*args):pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                result=generate(f'http://127.0.0.1:{server.server_port}',[1]*4,
                    {'length_policy':'fixed','sampling':{'max_new_tokens':1}},1,Path(directory)/'result.json')
                self.assertEqual(result['status'],'ok')
                self.assertGreaterEqual(result['ttft_s'],0.05)
        finally:server.shutdown();server.server_close();thread.join()

    def test_successful_worker_report_validation_rejects_wrong_metadata(self):
        from .benchmark.__main__ import validate_replay_result
        with tempfile.TemporaryDirectory() as directory:
            args,worker,meta=self.replay_fixture(Path(directory))
            for key,value in [('status','failed'),('mode','profile'),('candidate','wrong'),
                              ('trace_sha256','wrong'),('version','oct13-a1'),('before',None),
                              ('wall_s',0),('wall_s',float('nan')),('effective_child_env',{})]:
                bad=copy.deepcopy(worker);bad[key]=value
                with self.assertRaises(ValueError):
                    validate_replay_result(bad,'benchmark','baseline04',meta['trace_sha256'],0,48,VERSION)

    def test_effective_server_environment_is_full_policy(self):
        from .benchmark.core import subprocess_environment
        suite,machine=fixture()
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ',{'CUDA_LAUNCH_BLOCKING':'1'},clear=True), \
             patch('tools.qwen38_cache_tests.benchmark.runner.subprocess.run',side_effect=TimeoutError('stop before launch')), \
             patch('tools.qwen38_cache_tests.benchmark.runner.telemetry',return_value={}):
            result=run_unit(schedule(suite)[0],suite,machine,directory,directory,'benchmark')
            self.assertEqual(result['enforced_environment'],subprocess_environment())
            self.assertEqual(result['enforced_environment']['CUDA_LAUNCH_BLOCKING'],'1')
