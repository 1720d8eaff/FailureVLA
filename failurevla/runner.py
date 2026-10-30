"""Execute a caller-provided episode function and record its actual observations."""
import argparse
import importlib
from .experiment import METHODS, write_record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--runner', required=True, help='module:function accepting method config, returning one episode record')
    p.add_argument('--log', required=True)
    a = p.parse_args()
    module, name = a.runner.split(':', 1)
    run_episode = getattr(importlib.import_module(module), name)
    record = dict(run_episode(dict(METHODS[a.method])))
    record['method'] = a.method
    write_record(a.log, record)


if __name__ == '__main__':
    main()
