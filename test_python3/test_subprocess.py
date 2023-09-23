import subprocess
out = subprocess.run(['rm', 'xx'], capture_output=True, text=True, check=False)

print(out.stdout, 'ERR', out.stderr)