#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'safezone.settings.local')
    # 배포시에는 docker file 에서 delploy 로 변경 예정. manage.py 는 local 에서만 돌아가도록 설정
    # local 에서 배포용 연결시 아래 코드로 변경
    # os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'safezone.settings.deploy')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()