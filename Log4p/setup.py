from setuptools import setup, find_packages

setup(
    name='Log4p',
    version='1.0.0',
    packages=find_packages(),
    author='芙宁娜',
    author_email='3072252442@qq.com',
    description='A logging library for Python',
    long_description=open('Readme.md','r',encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/KOKOMI12345/Log4p',
    install_requires=[
        'requests',
        'websockets',
        'colorlog',
        'httpx',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
