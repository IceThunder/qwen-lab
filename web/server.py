"""Start with scripts/web.sh; stdlib HTTP server, localhost only."""
import argparse
import signal
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from backend import JobManager, discover_models, validate_request
from neuronary import fetch_recipe, normalize

STATIC = Path(__file__).parent / 'static'
manager = JobManager()
from model_manager import ModelManager
model_manager = ModelManager()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and str(args[0]).startswith("GET /api/jobs/"):
            return
        super().log_message(fmt, *args)

    def allowed(self):
        authority = self.headers.get('Host', '')
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if authority not in allowed:
            self.reply(403, {'error': '只接受本机地址'})
            return False
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://{host}' for host in allowed}:
            self.reply(403, {'error': '不接受跨站请求'})
            return False
        return True

    def reply(self, code, data):
        raw = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if not self.allowed():
            return
        route = urlsplit(self.path).path
        if route == '/api/model-manager':
            return self.reply(200, model_manager.snapshot())
        if route == '/api/models':
            return self.reply(200, {'models': discover_models(), 'active_job': manager.active})
        if route.startswith('/api/jobs/'):
            try:
                return self.reply(200, manager.snapshot(route.rsplit('/', 1)[-1]))
            except KeyError:
                return self.reply(404, {'error': '实验不存在或已过期'})
        assets = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
                  '/style.css': ('style.css', 'text/css'),
                  '/i18n.js': ('i18n.js', 'text/javascript'),
                  '/i18n-catalog.js': ('i18n-catalog.js', 'text/javascript'),
                  '/model-manager.js': ('model-manager.js', 'text/javascript')}
        if route not in assets:
            return self.reply(404, {'error': '不存在的路径'})
        file, mime = assets[route]
        raw = (STATIC / file).read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', mime + '; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if not self.allowed():
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 1048576:
                return self.reply(413, {'error': '请求大小必须在 1 MB 内'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.reply(415, {'error': '只接受 JSON'})
            body = json.loads(self.rfile.read(length))
            route = urlsplit(self.path).path
            if route in ('/api/model-manager/inspect', '/api/model-manager/install', '/api/model-manager/pause', '/api/models/test-load'):
                if not isinstance(body, dict):
                    raise ValueError('请求必须为 JSON 对象')
                if route.endswith('/inspect'):
                    return self.reply(200, model_manager.inspect(body.get('url')))
                if route.endswith('/install'):
                    return self.reply(202, {'id': model_manager.start(body.get('id'))})
                if route.endswith('/pause'):
                    model_manager.pause(body.get('id'))
                    return self.reply(200, {'ok': True})
                req = validate_request({'model': body.get('model'), 'messages': [{'role': 'user', 'content': 'Reply with OK.'}],
                    'groups': [], 'compare': False, 'thinking': False, 'max_tokens': 16, 'context_limit': 2048, 'temperature': 0})
                req['load_test'] = True
                return self.reply(202, {'id': manager.submit(req)})
            if route == '/api/neuronary/preview':
                if not isinstance(body, dict):
                    raise ValueError('请求必须为 JSON 对象')
                return self.reply(200, normalize(fetch_recipe(body.get('url')), discover_models()))
            if route == '/api/chat':
                req = validate_request(body)
                return self.reply(202, {'id': manager.submit(req)})
            if route.startswith('/api/jobs/') and route.endswith('/cancel'):
                manager.cancel(route.split('/')[-2])
                return self.reply(200, {'ok': True})
            return self.reply(404, {'error': '不存在的路径'})
        except (ValueError, TypeError) as exc:
            self.reply(400, {'error': str(exc)})
        except RuntimeError as exc:
            self.reply(409, {'error': str(exc)})
        except KeyError:
            self.reply(404, {'error': '实验不存在'})
        except Exception:
            self.reply(500, {'error': '服务器内部错误，请检查终端日志'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    print(f'Qwen Lab: http://127.0.0.1:{args.port}  (Ctrl+C 停止)', flush=True)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    def terminate(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if manager.active:
            manager.cancel(manager.active)
        server.server_close()
        model_manager.close()
        manager.pool.shutdown(wait=True)

        manager.gguf.stop()
