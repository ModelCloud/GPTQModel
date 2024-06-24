from .base import BaseGPTQModel


# placer=holder only as dbrx original models are not supported
# supported dbrx_converted models can be found on https://hf.co/ModelCloud
class DbrxGPTQ(BaseGPTQModel):
    info = {"notes": "Dbrx is only supported using defused/converted models on https://hf.co/ModelCloud with `trust_remote_code=True`"}
