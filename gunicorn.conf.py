"""
Gunicorn Configuration for Production Deployment
=================================================
Optimized settings for serving the Flask application.
"""

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
backlog = 2048

# Worker processes
# For ML applications, use fewer workers due to memory constraints
workers = int(os.environ.get('GUNICORN_WORKERS', min(2, multiprocessing.cpu_count())))
worker_class = 'sync'
worker_connections = 1000
timeout = 120  # Longer timeout for ML inference
keepalive = 5

# Process naming
proc_name = 'brain_tumor_segmentation'

# Logging
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log to stderr
loglevel = os.environ.get('LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (uncomment for HTTPS)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'

# Hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    print("Starting Brain Tumor Segmentation Server...")


def on_reload(server):
    """Called when the server is reloaded."""
    print("Reloading server...")


def worker_int(worker):
    """Called when a worker receives SIGINT."""
    print(f"Worker {worker.pid} interrupted")


def worker_abort(worker):
    """Called when a worker receives SIGABRT."""
    print(f"Worker {worker.pid} aborted")


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    print(f"Worker spawned (pid: {worker.pid})")


def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    print(f"Worker {worker.pid} ready to serve requests")


def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    print(f"Worker {worker.pid} exited")
