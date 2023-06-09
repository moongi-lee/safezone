from .base import *


def read_secret(secret_name):
    file = open('/run/secrets/' + secret_name)
    secret = file.read()
    secret = secret.rstrip().lstrip()
    file.close()
    return secret


# environ 설정 추가
env = environ.Env(
    DEBUG=(bool, False)
)

environ.Env.read_env(
    env_file= os.path.join(BASE_DIR, '.env')
)

# SECRET_KEY = env('SECRET_KEY')
SECRET_KEY = read_secret('DJANGO_SECRET_KEY')  # DOCKER SECRET 에서 가져옴


DEBUG = False # 배포환경에서 바뀌어야함.

ALLOWED_HOSTS = ["*"]


# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases
# db 어떻게 설정하는지 django 공식문서에서 설명 해줌

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql', # mariaDB 자체가 mysql 기반
        'NAME': 'safezone',
        'USER': 'safezone',
        # 'PASSWORD': 'ubuntu',
        'PASSWORD': read_secret('MYSQL_PASSWORD'),  # DOCKER SECRET 에서 가져옴
        'HOST': 'mariadb',  # container name
        'PORT': '3306',
    }
}


