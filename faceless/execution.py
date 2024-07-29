import torch

from functools import lru_cache
import subprocess
import xml.etree.ElementTree as ElementTree
from typing import List, Any

from .typing import ValueAndUnit, ExecutionDevice

def apply_execution_provider_options(execution_providers: List[str] | None = None) -> List[Any]:
    execution_providers_with_options : List[Any] = []

    if execution_providers is None:
        execution_providers = get_default_providers()
    for execution_provider in execution_providers:
        if execution_provider == 'CUDAExecutionProvider':
            execution_providers_with_options.append((execution_provider,
            {
                'cudnn_conv_algo_search': 'EXHAUSTIVE' if use_exhaustive() else 'DEFAULT'
            }))
        else:
            execution_providers_with_options.append(execution_provider)
    return execution_providers_with_options

def get_default_providers() -> List[str]:
    if torch.cuda.is_available():
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif torch.backends.mps.is_available():
        return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def use_exhaustive() -> bool:
    execution_devices = detect_static_execution_devices()
    product_names = ('GeForce GTX 1630', 'GeForce GTX 1650', 'GeForce GTX 1660')

    return any(execution_device.get('product').get('name').startswith(product_names) for execution_device in execution_devices)


def run_nvidia_smi() -> subprocess.Popen[bytes]:
    commands = [ 'nvidia-smi', '--query', '--xml-format' ]
    return subprocess.Popen(commands, stdout = subprocess.PIPE)

@lru_cache(maxsize = None)
def detect_static_execution_devices() -> List[ExecutionDevice]:
    return detect_execution_devices()

def detect_execution_devices() -> List[ExecutionDevice]:
    execution_devices : List[ExecutionDevice] = []
    try:
        output, _ = run_nvidia_smi().communicate()
        root_element = ElementTree.fromstring(output)
    except Exception:
        root_element = ElementTree.Element('xml')

    for gpu_element in root_element.findall('gpu'):
        execution_devices.append(
        {
            'driver_version': root_element.find('driver_version').text,
            'framework':
            {
                'name': 'CUDA',
                'version': root_element.find('cuda_version').text,
            },
            'product':
            {
                'vendor': 'NVIDIA',
                'name': gpu_element.find('product_name').text.replace('NVIDIA ', ''),
                'architecture': gpu_element.find('product_architecture').text,
            },
            'video_memory':
            {
                'total': create_value_and_unit(gpu_element.find('fb_memory_usage/total').text),
                'free': create_value_and_unit(gpu_element.find('fb_memory_usage/free').text)
            },
            'utilization':
            {
                'gpu': create_value_and_unit(gpu_element.find('utilization/gpu_util').text),
                'memory': create_value_and_unit(gpu_element.find('utilization/memory_util').text)
            }
        })
    return execution_devices

def create_value_and_unit(text : str) -> ValueAndUnit:
    value, unit = text.split()
    value_and_unit : ValueAndUnit =\
    {
        'value': value,
        'unit': unit
    }

    return value_and_unit
