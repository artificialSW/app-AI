# gunicorn.conf.py
import multiprocessing

# 서버 설정
bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1  # CPU 코어 수에 따라 자동 조정
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# 로그 설정
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 프로세스 설정
preload_app = True
timeout = 120
keepalive = 5

# 보안 설정
limit_request_line = 0
limit_request_field_size = 0