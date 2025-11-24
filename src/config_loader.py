import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


class ConfigLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._load_environment_variables()
    
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _load_environment_variables(self):
        load_dotenv()
        
        self._replace_env_vars(self.config)
    
    def _replace_env_vars(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = self._replace_env_vars(value)
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        return obj
    
    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_openai_config(self) -> Dict[str, Any]:
        return self.config.get('openai', {})
    
    def get_pinecone_config(self) -> Dict[str, Any]:
        return self.config.get('pinecone', {})
    
    def get_chunking_config(self) -> Dict[str, Any]:
        return self.config.get('chunking', {})
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        return self.config.get('retrieval', {})
    
    def get_docs_folder(self) -> str:
        return self.config.get('document_processing', {}).get('docs_folder', './docs')
    
    def get_supported_extensions(self) -> list:
        return self.config.get('document_processing', {}).get('supported_extensions', ['.txt', '.md'])
