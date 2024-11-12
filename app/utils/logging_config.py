import logging
import watchtower
import boto3
from datetime import datetime
from pathlib import Path

def setup_cloudwatch_logging(component_name: str):
    """
    Sets up both file and CloudWatch logging for a component
    
    Args:
        component_name: Name of the component (e.g., 'prompts_page', 'simulation_page')
    """
    logger = logging.getLogger(component_name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File logging setup
    base_dir = Path(__file__).resolve().parent.parent
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(
        logs_dir / f"{component_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    # CloudWatch logging setup
    cloudwatch_handler = watchtower.CloudWatchLogHandler(
        log_group_name='/sevenbots/application',
        log_stream_name=f'{component_name}-{datetime.now().strftime("%Y-%m-%d")}',
        boto3_client=boto3.client('logs')
    )
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Apply formatter to handlers
    file_handler.setFormatter(formatter)
    cloudwatch_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(cloudwatch_handler)
    logger.addHandler(logging.StreamHandler())  # Console output
    
    return logger 