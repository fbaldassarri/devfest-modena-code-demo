# Code for DevFest Modena speech - 2024, the October 12th

**Set-up intel-npu-acceleration-library**

```
(base) user@host:~/$ conda create -n intel-npu-acceleration-library python=3.12 -c intel -y

(base) user@host:~/$ conda activate intel-npu-acceleration-library

(intel-npu-acceleration-library) user@host:~/$ wget https://github.com/intel/intel-npu-acceleration-library/archive/refs/tags/v1.3.0.tar.gz

(intel-npu-acceleration-library) user@host:~/$ tar -xvzf intel-npu-acceleration-library-1.3.0.tar.gz

(intel-npu-acceleration-library) user@host:~/intel-npu-acceleration-library-1.3.0$ pip install -r requirements.txt --upgrade

(intel-npu-acceleration-library) user@host:~/intel-npu-acceleration-library-1.3.0$ pip install -r dev_requirements.txt --upgrade

(intel-npu-acceleration-library) user@host:~/intel-npu-acceleration-library-1.3.0$ pip install .

(intel-npu-acceleration-library) user@host:~/intel-npu-acceleration-library-1.3.0$ pip install .[dev]

(intel-npu-acceleration-library) user@host:~/intel-npu-acceleration-library-1.3.0$ conda install -c anaconda libstdcxx-ng

(intel-npu-acceleration-library) user@host:~/intel-npu-acceleration-library-1.3.0$ pip install transformers==4.42.4
```

**Quantization**

Prepare the environment

```
(base) user@host:~/$ conda create -n auto-round python=3.12 -c intel -y

(base) user@host:~/$ conda activate auto-round 

(auto-round) user@host:~/$ conda install pytorch torchvision torchaudio cpuonly -c pytorch

(auto-round) user@host:~/$ wget https://github.com/intel/auto-round/archive/refs/tags/v0.3.tar.gz

(auto-round) user@host:~/$ tar -xvzf v0.3.tar.gz

(auto-round) user@host:~/$ cd auto-round-0.3

(auto-round) user@host:~/auto-round-0.3$ pip install -r requirements.txt

(auto-round) user@host:~/auto-round-0.3$ python setup.py install
```

Run scripts

```
(auto-round) user@host:~/$ git clone https://github.com/fbaldassarri/devfest-modena-code-demo.git

(auto-round) user@host:~/$ cd devfest-modena-code-demo

(auto-round) user@host:~/devfest-modena-code-demo$ pip install --upgrade transformers torch

(auto-round) user@host:~/devfest-modena-code-demo$ pip install huggingface[cli]

(auto-round) user@host:~/devfest-modena-code-demo$ huggingface-cli login

(auto-round) user@host:~/devfest-modena-code-demo$ python 02-quantize-llama3.2-3B-Instruct-autoround.py

(auto-round) user@host:~/devfest-modena-code-demo$ conda deactivate
```
**Inference**

Intel® Extension for PyTorch (IPEX) [windows 11]

```
(base) user@host:~/$ conda create -n ipex-xpu python=3.9 -y

(base) user@host:~/$ conda activate ipex-xpu

(ipex-xpu) user@host:~/$ conda install libuv -y

```
Install Intel® oneAPI Base Toolkit 2024.1
https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=window

```
(ipex-xpu) user@host:~/$ call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

(ipex-xpu) user@host:~/$ pip install https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_dev/xpu/torch-2.1.0a0%2Bgit04048c2-cp39-cp39-win_amd64.whl

(ipex-xpu) user@host:~/$ pip install https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_dev/xpu/intel_extension_for_pytorch-2.1.30%2Bgit6661060-cp39-cp39-win_amd64.whl

(ipex-xpu) user@host:~/$ git clone https://github.com/intel/intel-extension-for-transformers.git intel-extension-for-transformers -b xpu_lm_head 

(ipex-xpu) user@host:~/$ cd intel-extension-for-transformers 

(ipex-xpu) user@host:~/intel-extension-for-transformers$ pip install -v .

(ipex-xpu) user@host:~/intel-extension-for-transformers$ pip install transformers==4.35

(ipex-xpu) user@host:~/intel-extension-for-transformers$ pip install huggingface_hub==0.22

(ipex-xpu) user@host:~/intel-extension-for-transformers$ pip install lm_eval==0.4.2 --no-deps

(ipex-xpu) user@host:~/intel-extension-for-transformers$ pip install accelerate datasets diffusers

```

Optional for full neural-compressor installation: 

```
(ipex) user@host:~/devfest-modena-code-demo$ pip install  deprecated opencv-python pandas pycocotools scikit-learn
```
Install IPEX for Intel® Core™ Ultra Processors with Intel® Arc™ Graphics (MTL-H)

```
(ipex) user@host:~/devfest-modena-code-demo$ python -m pip install torch==2.3.1+cxx11.abi torchvision==0.18.1+cxx11.abi torchaudio==2.3.1+cxx11.abi intel-extension-for-pytorch==2.3.110+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/mtl/us/

```
Install IPEX for Intel® Core™ Core™ Processors, Intel® Xeon® processors, and Intel® Xeon® Scalable Processors

```
(ipex) user@host:~/devfest-modena-code-demo$ python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

(ipex) user@host:~/devfest-modena-code-demo$ python -m pip install intel-extension-for-pytorch

(ipex) user@host:~/devfest-modena-code-demo$ python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

(ipex) user@host:~/devfest-modena-code-demo$ pip install Werkzeug>=3.0.0

(ipex) user@host:~/devfest-modena-code-demo$ pip install transformers==4.45.0

(ipex) user@host:~/devfest-modena-code-demo$ pip install numpy==1.24.1

```




