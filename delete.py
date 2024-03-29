import subprocess

# 需要卸载的依赖包列表
packages_to_uninstall = [
    'aiohttp', 'fastapi', 'ffmpy', 'fsspec', 'h11', 'httpx', 'jinja2', 'markdown-it-py',
    'matplotlib', 'numpy', 'orjson', 'pandas', 'paramiko', 'pillow', 'pycryptodome',
    'pydantic', 'pydub', 'python-multipart', 'pyyaml', 'requests', 'uvicorn', 'websockets',
    'aiosignal', 'attrs', 'frozenlist', 'multidict', 'yarl', 'starlette', 'typing-extensions',
    'annotated-types', 'pydantic-core', 'anyio', 'certifi', 'httpcore', 'idna', 'sniffio',
    'MarkupSafe', 'mdurl', 'linkify-it-py', 'mdit-py-plugins', 'contourpy', 'cycler',
    'fonttools', 'kiwisolver', 'packaging', 'pyparsing', 'python-dateutil', 'pytz', 'tzdata',
    'bcrypt', 'cryptography', 'pynacl', 'charset-normalizer', 'urllib3', 'click', 'colorama',
    'cffi', 'uc-micro-py', 'six', 'pycparser'
]

for package in packages_to_uninstall:
    subprocess.run(['pip', 'uninstall', '-y', package])

print("依赖包卸载完成！")
