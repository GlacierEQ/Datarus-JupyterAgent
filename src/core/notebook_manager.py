"""
Notebook Manager - Handles Docker container and Jupyter notebook management
"""

import os
import json
import time
import docker
import io
import tarfile
import requests
import socket
from io import BytesIO
import nbformat
from nbformat.v4 import new_notebook, new_output, new_code_cell, new_markdown_cell
from typing import Dict, List, Optional


class NotebookManager:
    """Manages Docker containers and Jupyter notebook execution"""
    
    def __init__(self, container_name: str, remote_serverip: str = None, config: Dict = None):
        # Initialize configuration
        self.remote_serverip = remote_serverip or "localhost"
        self.container_name = container_name
        self.volume_name = f"{container_name}_volume"
        self.container = None
        self.notebook = None
        self.notebook_name = None
        self.kernel_id = None
        self._cell_execution_count = 0
        self.config = config or {}
        self.token = None
        
        # Configure Docker client
        use_socket = os.getenv('USE_DOCKER_SOCKET', 'true').lower() == 'true'
        
        if use_socket or self.remote_serverip in ['localhost', '127.0.0.1']:
            # Use local Docker socket for better reliability
            print("Using local Docker socket")
            self.client = docker.from_env()
            self.jupyter_base_url = "http://localhost"
        else:
            # Use remote Docker TCP
            print(f"Using remote Docker at {self.remote_serverip}:2375")
            self.client = docker.DockerClient(base_url=f"tcp://{self.remote_serverip}:2375")
            self.jupyter_base_url = f"http://{self.remote_serverip}"
        
        # Port will be set when container is created
        self.jupyter_port = None
        self.jupyter_url = None
        
        self._ensure_volume()
        self._cleanup_existing_containers()

    def _cleanup_existing_containers(self):
        """Clean up any existing containers with the same name"""
        try:
            existing = self.client.containers.get(self.container_name)
            print(f"Found existing container {self.container_name}, removing...")
            existing.stop()
            existing.remove()
            time.sleep(2)
        except docker.errors.NotFound:
            pass
        except Exception as e:
            print(f"Warning during cleanup: {e}")

    def _find_available_port(self, start_port=8881):
        """Find an available port for the Jupyter server"""
        port = start_port
        max_attempts = 20
        
        for _ in range(max_attempts):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('', port))
                sock.close()
                
                # Also check if any Docker container is using this port
                containers = self.client.containers.list(all=True)
                port_in_use = False
                for container in containers:
                    if container.ports:
                        for container_port, bindings in container.ports.items():
                            if bindings:
                                for binding in bindings:
                                    if binding.get('HostPort') == str(port):
                                        port_in_use = True
                                        break
                
                if not port_in_use:
                    return port
                    
            except OSError:
                pass
            
            port += 1
        
        raise RuntimeError(f"Could not find available port in range {start_port}-{start_port+max_attempts}")

    def _ensure_volume(self):
        """Ensure Docker volume exists for persistent storage"""
        volumes = self.client.volumes.list()
        self.volume = None
        for vol in volumes:
            if vol.name == self.volume_name:
                self.volume = vol
                break
        if not self.volume:
            self.volume = self.client.volumes.create(name=self.volume_name)

    def _wait_for_jupyter(self, max_attempts=30):
        """Wait for Jupyter server to be ready"""
        print("Waiting for Jupyter server to start...")
        
        for attempt in range(max_attempts):
            try:
                # Get container logs to find the token
                logs = self.container.logs().decode('utf-8')
                
                # Look for the token in the logs
                import re
                token_match = re.search(r'token=([a-zA-Z0-9]+)', logs)
                if token_match:
                    self.token = token_match.group(1)
                    print(f"Found Jupyter token: {self.token}")
                
                # Try to connect to the server
                url = f"{self.jupyter_url}/api"
                headers = {}
                if self.token:
                    headers['Authorization'] = f'token {self.token}'
                
                response = requests.get(url, headers=headers, timeout=2)
                if response.status_code == 200:
                    print("Jupyter server is ready!")
                    return True
                    
            except (requests.exceptions.RequestException, docker.errors.APIError):
                pass
            
            time.sleep(1)
        
        raise RuntimeError("Jupyter server failed to start within timeout")

    def _ensure_kernel(self):
        """Ensure a kernel is running"""
        if not self.kernel_id:
            headers = {}
            if self.token:
                headers['Authorization'] = f'token {self.token}'
            
            response = requests.get(f"{self.jupyter_url}/api/kernels", headers=headers)
            response.raise_for_status()
            kernels = response.json()
            
            if kernels:
                self.kernel_id = kernels[0]['id']
            else:
                response = requests.post(f"{self.jupyter_url}/api/kernels", 
                                       json={"name": "python3"},
                                       headers=headers)
                response.raise_for_status()
                kernel_info = response.json()
                self.kernel_id = kernel_info['id']
                time.sleep(2)
        
        return self.kernel_id

    def ensure_container(self, docker_image="jupyter/tensorflow-notebook"):
        """Ensure container is running with proper port allocation"""
        containers = self.client.containers.list(all=True)
        
        for cont in containers:
            if cont.name == self.container_name:
                self.container = cont
                if self.container.status != "running":
                    print(f"Starting existing container {self.container_name}...")
                    self.container.start()
                    time.sleep(5)
                else:
                    print(f"Container {self.container_name} is already running")
                
                # Get the port from existing container
                if self.container.ports and '8888/tcp' in self.container.ports:
                    port_bindings = self.container.ports['8888/tcp']
                    if port_bindings:
                        self.jupyter_port = int(port_bindings[0]['HostPort'])
                        self.jupyter_url = f"{self.jupyter_base_url}:{self.jupyter_port}"
                        print(f"Using existing container on port {self.jupyter_port}")
                        self._wait_for_jupyter()
                
                return self.container.id
        
        # Find available port
        docker_config = self.config.get('docker', {})
        start_port = docker_config.get('jupyter_port', 8881)
        self.jupyter_port = self._find_available_port(start_port)
        self.jupyter_url = f"{self.jupyter_base_url}:{self.jupyter_port}"
        
        print(f"Creating new container {self.container_name} on port {self.jupyter_port}...")
        
        memory_limit = docker_config.get('memory_limit', '8g')
        cpu_quota = docker_config.get('cpu_quota', 80000)
        
        # Generate a random token for security
        import secrets
        self.token = secrets.token_hex(32)
        
        try:
            self.container = self.client.containers.run(
                docker_image,
                name=self.container_name,
                command=f"start-notebook.sh --NotebookApp.token='{self.token}' --NotebookApp.allow_origin='*'",
                ports={'8888/tcp': self.jupyter_port},
                detach=True,
                user='jovyan',
                volumes={self.volume_name: {'bind': '/home/jovyan/work', 'mode': 'rw'}},
                mem_limit=memory_limit,
                cpu_quota=cpu_quota,
                environment={
                    'JUPYTER_ENABLE_LAB': 'yes',
                    'JUPYTER_TOKEN': self.token
                },
                remove=False
            )
            print(f"Container created on port {self.jupyter_port}.")
            self._wait_for_jupyter()
            return self.container.id
            
        except docker.errors.APIError as e:
            if "port is already allocated" in str(e):
                print(f"Port {self.jupyter_port} became allocated, retrying with different port...")
                self.jupyter_port = self._find_available_port(self.jupyter_port + 1)
                self.jupyter_url = f"{self.jupyter_base_url}:{self.jupyter_port}"
                return self.ensure_container(docker_image)
            raise

    def _execute_code_direct(self, code: str, timeout: int = 60) -> List[Dict]:
        """Execute code directly in the kernel using jupyter_client"""
        self._ensure_kernel()
        
        # Script to execute code in the kernel
        exec_script = f"""
import json
import time
import jupyter_client
from queue import Empty

kernel_id = '{self.kernel_id}'

# Find the connection file
import glob
connection_files = glob.glob(f'/home/jovyan/.local/share/jupyter/runtime/kernel-{{kernel_id}}*.json')
if not connection_files:
    # Try alternative location
    connection_files = glob.glob(f'/tmp/kernel-{{kernel_id}}*.json')
if not connection_files:
    raise Exception(f"Kernel connection file not found for kernel {{kernel_id}}")

# Connect to the kernel
km = jupyter_client.BlockingKernelClient()
km.load_connection_file(connection_files[0])
km.start_channels()

# Execute the code
code = {repr(code)}
msg_id = km.execute(code, silent=False, store_history=True)

# Collect outputs
outputs = []
execution_count = None
timeout_seconds = {timeout}
start_time = time.time()

while True:
    if time.time() - start_time > timeout_seconds:
        outputs.append({{
            "output_type": "error",
            "ename": "TimeoutError",
            "evalue": "Cell execution timed out",
            "traceback": ["TimeoutError: Cell execution timed out after {{timeout_seconds}} seconds"]
        }})
        break
        
    try:
        msg = km.get_iopub_msg(timeout=1)
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
            
        msg_type = msg['header']['msg_type']
        content = msg['content']
        
        if msg_type == 'status' and content.get('execution_state') == 'idle':
            break
        elif msg_type == 'execute_input':
            execution_count = content.get('execution_count', 0)
        elif msg_type == 'stream':
            outputs.append({{
                'output_type': 'stream',
                'name': content.get('name', 'stdout'),
                'text': content.get('text', '')
            }})
        elif msg_type == 'execute_result':
            outputs.append({{
                'output_type': 'execute_result',
                'execution_count': content.get('execution_count', execution_count),
                'data': content.get('data', {{}})
            }})
        elif msg_type == 'display_data':
            outputs.append({{
                'output_type': 'display_data',
                'data': content.get('data', {{}})
            }})
        elif msg_type == 'error':
            outputs.append({{
                'output_type': 'error',
                'ename': content.get('ename', 'Error'),
                'evalue': content.get('evalue', ''),
                'traceback': content.get('traceback', [])
            }})
    except Empty:
        continue
    except Exception as e:
        outputs.append({{
            'output_type': 'error',
            'ename': type(e).__name__,
            'evalue': str(e),
            'traceback': [f"{{type(e).__name__}}: {{str(e)}}"]
        }})
        break

km.stop_channels()
print(json.dumps({{"outputs": outputs, "execution_count": execution_count}}))
"""
        
        # Execute the script in the container
        result = self.container.exec_run(
            ["python", "-c", exec_script],
            user='jovyan',
            stderr=True,
            stdout=True
        )
        
        if result.exit_code != 0:
            error_msg = result.output.decode('utf-8')
            return [{
                "output_type": "error",
                "ename": "KernelConnectionError",
                "evalue": f"Failed to execute code in kernel: {error_msg}",
                "traceback": [error_msg]
            }]
        
        # Parse the output
        output_str = result.output.decode('utf-8').strip()
        if output_str:
            try:
                result_data = json.loads(output_str)
                self._cell_execution_count = result_data.get('execution_count', self._cell_execution_count + 1)
                return result_data.get('outputs', [])
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw output
                return [{
                    "output_type": "stream",
                    "name": "stdout",
                    "text": output_str
                }]
        
        return []

    def _to_nb_output(self, o):
        """Convert output dictionary to notebook output format"""
        if isinstance(o, nbformat.NotebookNode):
            return o
        try:
            return nbformat.from_dict(o)
        except Exception:
            ot = o.get("output_type")
            if ot == "stream":
                return new_output(output_type="stream", name=o.get("name", "stdout"), text=o.get("text", ""))
            if ot == "execute_result":
                return new_output(output_type="execute_result", data=o.get("data", {}), execution_count=o.get("execution_count"))
            if ot == "display_data":
                return new_output(output_type="display_data", data=o.get("data", {}), metadata=o.get("metadata", {}))
            if ot == "error":
                return new_output(output_type="error", ename=o.get("ename", "Error"), evalue=o.get("evalue", ""), traceback=o.get("traceback", []))
            return nbformat.from_dict(o)

    def load_or_create_notebook(self, notebook_name: str):
        """Load existing notebook from container or create new one"""
        self.notebook_name = notebook_name
        self.reload_notebook_from_container()
        if self.notebook is None:
            self.notebook = new_notebook()
        return self.notebook, self.notebook_name

    def reload_notebook_from_container(self):
        """Reload notebook from Docker container"""
        notebook_path = f"/home/jovyan/work/{self.notebook_name}.ipynb"
        try:
            bits, stat = self.container.get_archive(notebook_path)
            tar_data = b''.join(bits)
            tar_file = tarfile.open(fileobj=BytesIO(tar_data))
            notebook_content = tar_file.extractfile(f'{self.notebook_name}.ipynb').read().decode('utf-8')
            self.notebook = nbformat.reads(notebook_content, as_version=4)
        except docker.errors.NotFound:
            self.notebook = None

    def save_notebook(self):
        """Save notebook to Docker container"""
        if not self.notebook or not self.container:
            return
        
        try:
            notebook_json = nbformat.writes(self.notebook).encode('utf-8')
            tar_buffer = io.BytesIO()
            
            with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
                tarinfo = tarfile.TarInfo(name=f'{self.notebook_name}.ipynb')
                tarinfo.size = len(notebook_json)
                tarinfo.uid = 1000
                tarinfo.gid = 100
                tarinfo.uname = 'jovyan'
                tarinfo.gname = 'users'
                tarinfo.mtime = int(time.time())
                tar.addfile(tarinfo, io.BytesIO(notebook_json))
            
            tar_buffer.seek(0)
            self.container.put_archive("/home/jovyan/work", tar_buffer)
        except Exception as e:
            print(f"Warning: Could not save notebook to container: {e}")

    def execute_cell(self, cell_index: int, timeout=60):
        """Execute a specific cell in the notebook"""
        if not self.notebook or cell_index >= len(self.notebook.cells):
            raise Exception("Invalid cell index")
        
        cell = self.notebook.cells[cell_index]
        if cell.cell_type != 'code':
            return 0, []
        
        raw_outputs = self._execute_code_direct(cell.source, timeout)
        cell.outputs = [self._to_nb_output(o) for o in raw_outputs]

        if not hasattr(cell, 'execution_count') or cell.execution_count is None:
            cell.execution_count = self._cell_execution_count

        self.save_notebook()

        # Check for errors
        has_error = any((o.get('output_type') if isinstance(o, dict) else o.get('output_type')) == 'error' for o in cell.outputs)
        return 1 if has_error else 0, raw_outputs

    def execute_last_cell(self, timeout=60):
        """Execute the last cell in the notebook"""
        if not self.notebook or len(self.notebook.cells) == 0:
            raise Exception("No cells in notebook")
        return self.execute_cell(len(self.notebook.cells) - 1, timeout)

    def create_step(self, step_name: str):
        """Create a new step in the notebook"""
        if not self.notebook:
            raise Exception("No notebook loaded.")
        header_cell = new_markdown_cell(source=f'## {step_name}')
        header_cell.metadata['step'] = step_name
        header_cell.metadata['role'] = 'step_header'
        self.notebook.cells.append(header_cell)
        self.save_notebook()

    def add_cell_to_step(self, cell_type: str, content: str, role: str, step_name: str, execute: bool = True):
        """Add a cell to a specific step"""
        if not self.notebook:
            raise Exception("No notebook loaded.")
        if cell_type == 'code':
            new_cell = new_code_cell(source=content)
            new_cell.execution_count = None
        elif cell_type == 'markdown':
            new_cell = new_markdown_cell(source=content)
        else:
            raise ValueError("Invalid cell type. Must be 'code' or 'markdown'.")
        new_cell.metadata['step'] = step_name
        new_cell.metadata['role'] = role
        self.notebook.cells.append(new_cell)
        self.save_notebook()
        
        if execute and cell_type == 'code':
            return self.execute_last_cell()
        
        return 0, []

    def get_outputs_by_step(self, step_name: str) -> Dict:
        """Get all outputs for a specific step"""
        if not self.notebook:
            raise Exception("No notebook loaded.")
        
        all_outputs = []
        for cell in self.notebook.cells:
            if cell.metadata.get('step') == step_name and cell.cell_type == 'code':
                all_outputs.extend(cell.get('outputs', []))

        stream_output = []
        combined_error = []
        has_error = False
        
        for output in all_outputs:
            if output['output_type'] == 'stream':
                text = ''.join(output['text']) if isinstance(output['text'], list) else output['text']
                stream_output.append(text)
            elif output['output_type'] == 'error':
                has_error = True
                # Include output before error for debugging
                if stream_output:
                    combined_error.append("Output before error:")
                    combined_error.extend(stream_output)
                    combined_error.append("")
                
                combined_error.append(f"Error: {output['ename']}: {output['evalue']}")
                
                if 'traceback' in output:
                    for line in output['traceback']:
                        if '-->' in line and 'ipython-input' in line:
                            combined_error.append(f"Line that caused error: {line.strip()}")
                        combined_error.append(line)
            elif output['output_type'] in ['execute_result', 'display_data']:
                if 'text/plain' in output.get('data', {}):
                    stream_output.append(output['data']['text/plain'])

        return {
            "stream": '\n'.join(stream_output) if not has_error else "",
            "error": '\n'.join(combined_error) if has_error else '\n'.join(stream_output),
            "has_error": has_error
        }

    def update_cell_content(self, step_name: str, cell_type: str, role: str, updated_content: str, execute: bool = True):
        """Update the content of a specific cell"""
        if not self.notebook:
            raise Exception("No notebook loaded.")
        
        cell_index = None
        for i, cell in enumerate(self.notebook.cells):
            if (cell.metadata.get('step') == step_name and 
                cell.cell_type == cell_type and 
                cell.metadata.get('role') == role):
                cell.source = updated_content
                cell_index = i
                break
        
        if cell_index is not None:
            self.save_notebook()
            if execute and cell_type == 'code':
                return self.execute_cell(cell_index)
            return 0, []
        
        return 1, ["Cell not found"]

    def save_notebook_to_file(self, path):
        """Save notebook to local file"""
        if self.notebook:
            with open(path, 'w') as f:
                nbformat.write(self.notebook, f)
            print(f"Notebook saved to: {path}")

    def cleanup(self):
        """Cleanup resources (container kept for inspection)"""
        if self.container:
            print(f"Container {self.container_name} kept for inspection")
            print(f"To remove: docker rm -f {self.container_name}")
            if self.jupyter_port:
                print(f"Jupyter available at: http://localhost:{self.jupyter_port}")
                if self.token:
                    print(f"Token: {self.token}")
